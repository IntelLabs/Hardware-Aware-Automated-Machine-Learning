import math
from torch import Tensor, nn
from torch.autograd import Function
import torch

import custom_onednn_linear


class OneDnnLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        # import pdb; pdb.set_trace()
        self.onednn_primitive = custom_onednn_linear.get_onednn_descriptor(1,in_features, out_features)

    # we only override the forward to map to our custom kernel
    # backward will fallback to nn.Linear.backward()

    def forward(self, input: Tensor) -> Tensor:
        # print("[Info]: Entering custom linear implementation")
        # import pdb; pdb.set_trace()
        return custom_onednn_linear.forward(self.onednn_primitive, input, self.weight, self.bias)

    @classmethod
    def from_linear(cls, nn_linear_inst: nn.Linear, shallow=False):
        # TODO: not an efficient implementation, we will create another copy of parameters, imagine Billion scale llm!
        # see llm_pipeline.py for a "Wrapper" way
        new_inst = cls(nn_linear_inst.in_features, 
                       nn_linear_inst.out_features, 
                       nn_linear_inst.bias is not None, 
                       next(nn_linear_inst.parameters()).device, 
                       next(nn_linear_inst.parameters()).dtype)
        
        # new_inst.weight.data = nn_linear_inst.weight.data
        print(f"Before weight conversion: {nn_linear_inst.weight.data}")
        # Convert to AMX format: Group each 2 consecutive elements, then transpose then reshape to have the same original shape.
        new_inst.weight.data = nn_linear_inst.weight.data
        # ^ The above handles corner cases for the below.
        # new_inst.weight.data = nn_linear_inst.weight.data.view(-1, 2, 2).transpose(0, 1).reshape(nn_linear_inst.weight.data.shape)
        print(f"After weight conversion: {new_inst.weight.data}")
        if new_inst.bias is not None:
            new_inst.bias.data = nn_linear_inst.bias.data
        return new_inst