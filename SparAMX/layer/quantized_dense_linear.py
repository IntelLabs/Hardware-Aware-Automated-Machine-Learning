import math
from sympy import count_ops
from torch import Tensor, nn
from torch.autograd import Function
import torch
import numpy as np
from bitsandbytes.nn.modules import Linear8bitLt
from torch.nn.parameter import Parameter

from sparamx import quantized_dense_linear
import ctypes

class AlignedMemory:
    def __init__(self, size, alignment=64):
        self.size = size
        self.alignment = alignment
        self.buffer = ctypes.create_string_buffer(size + alignment - 1)
        offset = ctypes.addressof(self.buffer) % alignment
        self.aligned_address = ctypes.addressof(self.buffer) + alignment - offset if offset else ctypes.addressof(self.buffer)

    def get_ptr(self):
        return self.aligned_address

def create_aligned_tensor(shape, dtype, cdtype, alignment=64):
    # return torch.empty(shape, dtype=dtype)
    size = torch.empty(shape, dtype=dtype).numel() * dtype.itemsize
    mem = AlignedMemory(size, alignment)
    return torch.from_numpy(np.frombuffer(
        (ctypes.c_byte * size).from_address(mem.get_ptr()), dtype=np.dtype(cdtype)
    )).reshape(shape).clone()


class QuantizedDenseLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super(torch.nn.Linear, self).__init__() # We'll do the initialization ourselves.
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False) # AHMED modified here, adding requires_grad=False
        self.w_scales = nn.Parameter(torch.empty((out_features), dtype=torch.bfloat16, requires_grad=False))

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def set_all_zeros(self, weights):
        self.weight_values.data = torch.Tensor([])
        metadata_size           = math.ceil(weights.numel() / 32.)
        self.weight_metadata    = nn.Parameter(torch.zeros(metadata_size, dtype=torch.int32), requires_grad = False)

    def forward(self, input: Tensor) -> Tensor:
        # print("[Info]: Entering custom linear implementation")
        # import pdb; pdb.set_trace()
        # reshaped_input = input.reshape(-1, input.shape[-1]).to(torch.bfloat16)
        # return torch.ones(reshaped_input.shape[0],self.out_cols, dtype=torch.bfloat16)


        # import pdb; pdb.set_trace()
        # Calculate max for each token separately
        # keepdim=True preserves dimension for broadcasting
        token_abs_max = torch.max(torch.abs(input), dim=2, keepdim=True)[0]
        
        # Calculate scaling factor for each token
        # Shape: (num_tokens, 1)
        token_scales = torch.minimum(
            127.0 / token_abs_max,
            torch.ones_like(token_abs_max)
        )        
        # Quantize each token using its own scaling factor
        # Broadcasting will apply each scale to its corresponding token
        input_int8 = torch.round(input * token_scales).to(torch.int8)
    
        # input_abs_max = torch.max(torch.abs(input))
        # input_scale = 127.0 / input_abs_max
        # # input_scale = 0.5
        
        # # Quantize input to INT8
        # input_int8 = torch.round(input * input_scale).to(torch.int8)
        
        # import pdb; pdb.set_trace()
        # Call the custom kernel with quantized input
        result = quantized_dense_linear.forward(
            input_int8, 
            self.weight,
            self.bias
        )
        
        scale_down = token_scales.to(self.w_scales.dtype) * self.w_scales.data

        # Dequantize the result back to bfloat16
        # Note: The output scale needs to account for both input and weight scaling
        # Assuming weights are already properly scaled in the kernel
        return result / scale_down
            


        # with torch.autograd.profiler.emit_itt():
        # import pdb; pdb.set_trace()
        return quantized_dense_linear.forward(input, self.weight)
        # return quantized_sparse_linear.forward(input, self.weight_metadata, self.weight_values, self.weight_values_is, self.out_cols, self.bias)


    def quantize_weights(self, weights):
        # import pdb; pdb.set_trace()
        # Calculate max for each token separately
        # keepdim=True preserves dimension for broadcasting
        token_abs_max = torch.max(torch.abs(weights), dim=1, keepdim=True)[0]
        
        # Calculate scaling factor for each token
        # Shape: (num_tokens, 1)
        token_scales = torch.minimum(
                127.0 / token_abs_max,
                torch.ones_like(token_abs_max)
            )
 
        # Quantize each token using its own scaling factor
        # Broadcasting will apply each scale to its corresponding token
        quantized_weights = torch.round(weights * token_scales).to(torch.int8)
        # import pdb; pdb.set_trace()
        self.w_scales.data = token_scales.squeeze().to(torch.bfloat16)
        return quantized_weights

    @classmethod
    def from_linear(cls, nn_linear_inst: nn.Linear, num_threads=32, shallow=False):
        # import pdb; pdb.set_trace()
        # TODO: not an efficient implementation, we will create another copy of parameters, imagine Billion scale llm!
        # see llm_pipeline.py for a "Wrapper" way
        new_inst = cls(nn_linear_inst.in_features, 
                       nn_linear_inst.out_features, 
                       nn_linear_inst.bias is not None, 
                       next(nn_linear_inst.parameters()).device, 
                       next(nn_linear_inst.parameters()).dtype)
        if shallow:
            return new_inst
        # import pdb; pdb.set_trace()
        # new_inst.weight.data = nn_linear_inst.weight.data
        # print(f"Before weight conversion: {nn_linear_inst.weight.data}")
        # Convert to AMX format: Group each 2 consecutive elements, then transpose then reshape to have the same original shape.
        # import pdb; pdb.set_trace()
        weights = new_inst.quantize_weights(nn_linear_inst.weight.data)
        # weights = nn_linear_inst.weight.data.to(torch.int8)
        weights = weights.view(nn_linear_inst.weight.data.shape[0], -1, 4).transpose(0, 1)
        weights = weights.reshape(nn_linear_inst.weight.data.shape).contiguous()

        # # import pdb; pdb.set_trace()
        # if weights.shape[1] > 64:
        #     weights = weights.view(weights.shape[0], -1, )

        new_inst.weight.data = weights


        if new_inst.bias is not None:
            new_inst.bias.data = nn_linear_inst.bias.data
        return new_inst