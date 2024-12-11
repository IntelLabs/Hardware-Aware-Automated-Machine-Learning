from typing import List, Optional

import numpy as np
import torch

from peft.tuners.lora.gptq import QuantLinear


class SQFTQuantAwareLinear(QuantLinear):
    """
    A class that extends QuantLinear to support SQFT quantization-aware training with LoRA adapters.

    Attributes:
        scales (torch.Tensor): Scales for quantization.
        zeros (torch.Tensor): Zero points for quantization.
        g_idx (torch.Tensor): Group indices for quantization.
        wf (torch.Tensor): Weight factors for quantization.
        group_size (int): Size of the group for quantization.
        bits (int): Number of bits for quantization.
        base_weight (torch.Tensor): Base weight of the linear layer.
        sparse_adapter (bool): Flag to indicate if sparse adapter is used.
    """

    INT4_MAX = 15

    def __init__(
            self,
            base_layer,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            sparse_adapter: bool = False,
            **kwargs,
    ) -> None:
        """
        Initializes the SQFTQuantAwareLinear class.
        """
        super().__init__(base_layer, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights,
                         use_rslora, use_dora, **kwargs)
        self.scales = None
        self.zeros = None
        self.g_idx = None
        self.wf = None
        self.group_size = None
        self.bits = None

        self.base_weight = None
        self.sparse_adapter = sparse_adapter

    def init_params(self, custom_weight: Optional[torch.Tensor] = None) -> None:
        """
        Initializes the parameters for quantization.

        This function is referenced from:
        https://github.com/AutoGPTQ/AutoGPTQ/blob/866b4c8c2cbb893f1156cb6c114625bba2e4d7c5/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py#L296-L316

        Args:
            custom_weight (Optional[torch.Tensor]): Custom weight for initialization.
        """
        self.group_size = self.quant_linear_module.group_size
        self.g_idx = self.quant_linear_module.g_idx
        self.wf = self.quant_linear_module.wf.to(self.quant_linear_module.qzeros.device)
        self.bits = self.quant_linear_module.bits
        self.scales = self.quant_linear_module.scales

        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.quant_linear_module.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        zeros = zeros + 1
        zeros = torch.bitwise_and(
            zeros, (2 ** self.bits) - 1
        )
        self.zeros = zeros.reshape(-1, zeros.shape[1] * zeros.shape[2])

        if custom_weight is not None:
            self.base_weight = custom_weight
        else:
            weight = torch.bitwise_right_shift(
                torch.unsqueeze(self.quant_linear_module.qweight, 1).expand(-1, 32 // self.bits, -1),
                self.wf.unsqueeze(-1),
            ).to(torch.int16 if self.quant_linear_module.bits == 8 else torch.int8)
            int_weight = torch.bitwise_and(weight, (2 ** self.bits) - 1)
            int_weight = int_weight.reshape(-1, int_weight.shape[-1])
            self.base_weight = self.dequantize_weight(int_weight).t().contiguous()

    def matmul_dqweight(self, input_tensor: torch.Tensor, dequantized_weight: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication with dequantized weight.

        Args:
            input_tensor (torch.Tensor): Input tensor.
            dequantized_weight (torch.Tensor): Dequantized weight tensor.

        Returns:
            torch.Tensor: Result of the matrix multiplication.
        """
        output_shape = input_tensor.shape[:-1] + (self.quant_linear_module.outfeatures,)
        input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])
        result = torch.matmul(input_tensor, dequantized_weight)
        result = result.reshape(output_shape)
        result = result + self.quant_linear_module.bias if self.quant_linear_module.bias is not None else result
        return result

    def dequantize_weight(self, int_weight: torch.Tensor) -> torch.Tensor:
        """
        Dequantizes the weight tensor.

        Args:
            int_weight (torch.Tensor): Integer weight tensor.

        Returns:
            torch.Tensor: Dequantized weight tensor.
        """
        scales = self.scales.reshape(-1, 1, self.scales.shape[-1])
        zeros = self.zeros.reshape(-1, 1, self.zeros.shape[-1])

        int_weight = int_weight.reshape(-1, self.group_size, int_weight.shape[-1])
        dequantized_weight = scales * (int_weight - zeros)

        dequantized_weight = dequantized_weight.reshape(
            dequantized_weight.shape[0] * dequantized_weight.shape[1],
            dequantized_weight.shape[2]
        )
        return dequantized_weight

    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Quantizes the weight tensor.

        Args:
            weight (torch.Tensor): Weight tensor.

        Returns:
            torch.Tensor: Quantized weight tensor.
        """
        scale_zeros = self.zeros * self.scales
        int_weight = torch.round(
            (weight.t().contiguous() + scale_zeros[self.g_idx]) / self.scales[self.g_idx]
        ).clamp(0, self.INT4_MAX)
        return int_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        result_dtype = x.dtype
        assert len(self.active_adapters) == 1
        active_adapter = self.active_adapters[0]
        assert active_adapter in self.lora_A.keys()
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        scaling = self.scaling[active_adapter]
        dropout = self.lora_dropout[active_adapter]

        base_weight = self.base_weight
        lora_A_weight = lora_A.weight
        lora_B_weight = lora_B.weight
        adapter_weight = torch.matmul(lora_B_weight, lora_A_weight) * scaling
        if self.sparse_adapter:
            mask = (base_weight != 0).detach()
            adapter_weight = adapter_weight * mask
        merged_weight = base_weight + adapter_weight
        merged_dqweight = merged_weight.t().contiguous() + (
                self.dequantize_weight(self.quantize_weight(merged_weight)) - merged_weight.t().contiguous()
        ).detach()
        x = x.to(merged_dqweight.dtype)
        result = self.matmul_dqweight(dropout(x), merged_dqweight)
        result = result.to(result_dtype)

        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merges the adapter weights into the base weight.

        Reference:
        https://github.com/AutoGPTQ/AutoGPTQ/blob/866b4c8c2cbb893f1156cb6c114625bba2e4d7c5/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py#L135-L140
        """
        base_weight = self.base_weight
        active_adapter = self.active_adapters[0]
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        scaling = self.scaling[active_adapter]
        adapter_weight = torch.matmul(lora_B.weight, lora_A.weight) * scaling
        if self.sparse_adapter:
            mask = (base_weight != 0).detach()
            adapter_weight = adapter_weight * mask
        merged_weight = base_weight + adapter_weight
        merged_intweight = self.quantize_weight(merged_weight)

        device = merged_intweight.device
        intweight_np = merged_intweight.cpu().numpy().astype(np.uint32)
        i = 0
        row = 0
        qweight = np.zeros((int(intweight_np.shape[0] / 32 * self.bits), intweight_np.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight_np[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")
        qweight = qweight.astype(np.int32)
        qweight = torch.from_numpy(qweight).to(device)

        self.quant_linear_module.qweight.data = qweight
        self.merged_adapters.append(active_adapter)
