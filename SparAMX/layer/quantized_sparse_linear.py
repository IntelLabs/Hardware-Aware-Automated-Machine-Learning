import math
from sympy import count_ops
from torch import Tensor, nn
from torch.autograd import Function
import torch
import numpy as np
from bitsandbytes.nn.modules import Linear8bitLt
from torch.nn.parameter import Parameter

from sparamx import quantized_sparse_linear
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


def bits_to_int32(bit_tensor):
    # Calculate the number of 32-bit integers we'll need
    num_ints = (bit_tensor.size(0) + 31) // 32

    # Pad the bit tensor with zeros to make its length a multiple of 32
    padded_bits = torch.nn.functional.pad(bit_tensor, (0, num_ints * 32 - bit_tensor.size(0)))

    # Reshape the padded tensor to have 32 bits per row
    reshaped = padded_bits.view(-1, 32)

    # Convert each row of 32 bits to an integer
    powers_of_two = 2 ** torch.arange(32, dtype=torch.int64, device=bit_tensor.device)
    int32_tensor = (reshaped * powers_of_two).sum(dim=1, dtype=torch.int32)

    return int32_tensor

class QuantizedSparseLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super(torch.nn.Linear, self).__init__() # We'll do the initialization ourselves.
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        metadata_size = math.ceil(out_features * in_features / 32.)
        self.weight_metadata    = nn.Parameter(create_aligned_tensor((metadata_size,), torch.int32, np.int32), requires_grad=False)
        self.weight_values      = nn.Parameter(create_aligned_tensor((0,), torch.int8, np.int8), requires_grad = False)
        self.weight_values_is   = nn.Parameter(create_aligned_tensor((0,), torch.int32, np.int32), requires_grad = False) # Using int32 results in a weird bug where the CPP extension reads an off by 1 number.
        self.out_cols           = out_features
        self.w_scales           = nn.Parameter(torch.empty((out_features), dtype=torch.bfloat16, requires_grad=False))
        del self.weight

    def populate_sparse_weights(self, weights):
        bitmap_tensor = (weights != 0)
        bitmap_list = bitmap_tensor.reshape(-1)
        self.weight_metadata.data = bits_to_int32(bitmap_list)
        self.weight_values.data = weights[bitmap_tensor]

    def split_weights_to_sets(self, num_sets):
        num_sets = min(num_sets, math.ceil(self.out_cols / 16.))
        # assert (self.out_cols / 16) % num_sets == 0, f"num_sets ({num_sets}) does not divide out_cols ({self.out_cols}) evenly."

        if int(self.weight_values.numel()) == 0:
            self.weight_values_is.data = torch.Tensor([0] * num_sets)
            return
        assert self.weight_metadata.numel() % num_sets == 0, "Number of sets doesn't divide weights equally."
        metadata_elements_per_set = self.weight_metadata.numel() // num_sets
        self.weight_values_is.data = torch.empty(num_sets, dtype=torch.int32)
        prev = 0
        for i in range(num_sets):
            num_elements = 0
            for j in range(metadata_elements_per_set):
                cnt = bin(int(self.weight_metadata.data[i * metadata_elements_per_set + j]) & 0xFFFFFFFF).count('1')
                num_elements+= cnt
            # print(f'Num Weights in thread {i}: {num_elements}')
            self.weight_values_is.data[i] = prev
            # print(f'I: {i}, Num Elements: {num_elements}, Prev: {prev}')
            prev+= num_elements

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
        
        # Call the custom kernel with quantized input
        result = quantized_sparse_linear.forward(
            input_int8, 
            self.weight_metadata, 
            self.weight_values, 
            self.weight_values_is, 
            self.out_cols, 
            self.bias
        )
        
        scale_down = token_scales.to(self.w_scales.dtype) * self.w_scales.data

        # Dequantize the result back to bfloat16
        # Note: The output scale needs to account for both input and weight scaling
        # Assuming weights are already properly scaled in the kernel
        return result / scale_down
            


        # with torch.autograd.profiler.emit_itt():
        # import pdb; pdb.set_trace()
        return quantized_sparse_linear.forward(input, self.weight_metadata, self.weight_values, self.weight_values_is, self.out_cols, self.bias)


    def quantize_weights(self, weights):
        # import pdb; pdb.set_trace()
        # Calculate max for each token separately
        # keepdim=True preserves dimension for broadcasting
        weight_abs_max = torch.max(torch.abs(weights), dim=1, keepdim=True)[0]
        
        # Calculate scaling factor for each token
        # Shape: (num_tokens, 1)
        weight_scales = torch.minimum(
                127.0 / weight_abs_max,
                torch.ones_like(weight_abs_max)
            )        
        # Quantize each token using its own scaling factor
        # Broadcasting will apply each scale to its corresponding token
        quantized_weights = torch.round(weights * weight_scales).to(torch.int8)
        # import pdb; pdb.set_trace()
        self.w_scales.data = weight_scales.squeeze().to(torch.bfloat16)
        return quantized_weights

    @classmethod
    def from_linear(cls, nn_linear_inst: nn.Linear, num_threads=32, shallow=False):
        # import pdb; pdb.set_trace()
        num_threads = min(num_threads, math.ceil(nn_linear_inst.out_features / 16.))
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
        weights = weights.view(nn_linear_inst.weight.data.shape[0], -1, 4).transpose(0, 1)
        weights = weights.reshape(weights.shape[0], -1).contiguous()
        # import pdb; pdb.set_trace()
        # if weights.shape[1] > 32:
        #     weights = weights.view(weights.shape[0], -1, )
        # weights = torch.arange()
        
        if weights.shape[1] > 64:
            # Will have multiple tiles in the out_cols dimension. Need to reorder the
            # weights to be suitable for consumption, putting items of each tile next to each other
            # because that's how the Kernel consumes them.
            tile_width = 64
            tile_height = 16
            if weights.shape[1] // num_threads > tile_width:
                grouped_col_tiles = True
            else:
                grouped_col_tiles = False
            weights = weights.view(weights.shape[0], -1, tile_width).transpose(0, 1)
            if grouped_col_tiles:
                weights = weights.view(weights.shape[0], weights.shape[1] // tile_height, tile_height, weights.shape[2]).transpose(0,1)
                weights = weights.view(weights.shape[0], weights.shape[1] // 2, 2, weights.shape[2], weights.shape[3]).transpose(0,1)

        # import pdb; pdb.set_trace()

        # TODO: Custom implementation. Must be removed for accurate output

        # new_inst.set_all_zeros(weights)
        new_inst.populate_sparse_weights(weights)
        # import pdb; pdb.set_trace();
        new_inst.split_weights_to_sets(num_threads)
        # import pdb; pdb.set_trace();
        # print(f"After weight conversion: {weights}")
        if new_inst.bias is not None:
            new_inst.bias.data = nn_linear_inst.bias.data
        return new_inst