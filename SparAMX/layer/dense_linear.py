import math
from torch import Tensor, nn
from torch.autograd import Function
import torch
import os

from sparamx import dense_linear

class DenseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)


    # we only override the forward to map to our custom kernel
    # backward will fallback to nn.Linear.backward()

    def forward(self, input: Tensor) -> Tensor:
        # print("[Info]: Entering custom linear implementation")
        # import pdb; pdb.set_trace()
        reshaped_input = input.reshape(-1, input.shape[-1]).to(torch.bfloat16)
        # import pdb; pdb.set_trace()
        output = torch.zeros(input.shape[0], self.weight.shape[0], dtype=torch.float32, device=input.device).contiguous()
        dense_linear.forward(reshaped_input, self.weight.to(torch.bfloat16), output, self.bias)
        return output.reshape(input.shape[0], -1).to(torch.bfloat16)

    def matmul(self, mat):
        # import pdb; pdb.set_trace()
        # with torch.autograd.profiler.emit_itt():
            # print(self.weight_values.size())
        return dense_linear.matmul(mat, self.weight, self.bias)


    def save_state(self, path):
        # Create path if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'weight': self.weight,
        }, path)

    @classmethod
    def load_from_file(cls, path):
        state = torch.load(path)
        new_inst = cls(state['weight'].shape[0], state['weight'].shape[1], False, 'cpu', torch.bfloat16)
        new_inst.weight = state['weight']
        # Make sure weight is cache aligned
        if new_inst.weight.data_ptr() % 64 != 0:  # Use data_ptr() instead of ctypes.addressof()
            new_inst.weight = nn.Parameter(new_inst.weight.data.clone(), requires_grad=False)
        return new_inst

    @classmethod
    def from_linear(cls, nn_linear_inst: nn.Linear, quiet = True, num_threads= 64, shallow=False):
        # TODO: not an efficient implementation, we will create another copy of parameters, imagine Billion scale llm!
        # see llm_pipeline.py for a "Wrapper" way
        new_inst = cls(nn_linear_inst.in_features, 
                       nn_linear_inst.out_features,
                       nn_linear_inst.bias is not None, 
                       next(nn_linear_inst.parameters()).device, 
                       next(nn_linear_inst.parameters()).dtype)
        
        # new_inst.weight.data = nn_linear_inst.weight.data
        if not(quiet):
            print(f"Before weight conversion: {nn_linear_inst.weight.data}")
        # Convert to AMX format: Group each 2 consecutive elements, then transpose then reshape to have the same original shape.
        new_inst.weight.data = nn_linear_inst.weight.data.view(nn_linear_inst.weight.data.shape[0], -1, 2).transpose(0, 1).reshape(nn_linear_inst.weight.data.shape).to(torch.bfloat16).contiguous()
        # ^ The above handles corner cases for the below.
        # new_inst.weight.data = nn_linear_inst.weight.data.view(-1, 2, 2).transpose(0, 1).reshape(nn_linear_inst.weight.data.shape)
        if not(quiet):
            print(f"After weight conversion: {new_inst.weight.data}")
        if new_inst.bias is not None:
            new_inst.bias.data = nn_linear_inst.bias.data
        return new_inst

    @classmethod
    def batched_matmul(cls, mat1, mat2):
        """Performs Batched Matrix Multiplication using the sparse kernel"""
        input = mat1
        weight = mat2.transpose(2, 3)
        computer = cls.from_batched_weights(weight)
        output = computer.matmul(input)
        return output


    @classmethod
    def batched_matmul_naive(cls, mat1, mat2):
        batch_size, num_heads, out_rows, inner_dim = mat1.size()
        res = torch.empty(batch_size, num_heads, out_rows, mat2.size(-1), device=mat1.device, dtype=mat1.dtype)
        mat2t = mat2.transpose(2, 3)
        for b in range(batch_size):
            for h in range(num_heads):
                input = mat1[b, h]
                weight = mat2t[b, h]
                computer = cls.from_weights(weight)
                output = computer(input)
                res[b, h] = output
        return res

    @classmethod
    def from_weights(cls, orig_weights, num_threads = 32, shallow=False):
        weights = orig_weights.view(orig_weights.shape[0], -1, 2).transpose(0, 1).reshape(orig_weights.shape).to(torch.bfloat16).contiguous()
        new_inst = cls(weights.shape[1], weights.shape[0], False, 'cpu', torch.bfloat16)
        new_inst.weight.data = weights

        return new_inst

    @classmethod
    def from_batched_weights(cls, orig_weights, num_threads = 32, shallow=False):
        weights = orig_weights.view(orig_weights.shape[0], orig_weights.shape[1], orig_weights.shape[2], -1, 2).transpose(2, 3).reshape(orig_weights.shape).to(torch.bfloat16).contiguous()
        new_inst = cls(weights.shape[3], weights.shape[2], False, 'cpu', torch.bfloat16)
        new_inst.weight.data = weights

        return new_inst