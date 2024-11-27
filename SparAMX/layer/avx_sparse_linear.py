import math
from sympy import count_ops
from torch import Tensor, nn
from torch.autograd import Function
import torch
import numpy as np
import os
import ctypes

from sparamx import avx_sparse_linear

# CHANGE BELOW FOR GROUP SIZE
NUM_GROUPS = 32

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

def bits_to_int32_batched(bit_tensor):
    # Calculate the number of 32-bit integers we'll need
    num_ints = (bit_tensor.size(2) + 31) // 32

    # Pad the bit tensor with zeros to make its length a multiple of 32
    padded_bits = torch.nn.functional.pad(bit_tensor, (0, num_ints * 32 - bit_tensor.size(2)))

    # Reshape the padded tensor to have 32 bits per row
    reshaped = padded_bits.view(padded_bits.shape[0], padded_bits.shape[1], -1, 32)

    # Convert each row of 32 bits to an integer
    powers_of_two = 2 ** torch.arange(32, dtype=torch.int64, device=bit_tensor.device)
    int32_tensor = (reshaped * powers_of_two).sum(dim=3, dtype=torch.int32)

    return int32_tensor

class AvxSparseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        metadata_size = math.ceil(out_features * in_features / 32.)
        self.weight_metadata    = nn.Parameter(create_aligned_tensor((metadata_size,), torch.int32, np.int32), requires_grad=False)
        self.weight_values      = nn.Parameter(create_aligned_tensor((0,), torch.bfloat16, np.float16), requires_grad = False)
        self.weight_values_is   = nn.Parameter(create_aligned_tensor((0,), torch.int32, np.int32), requires_grad = False) # Using int32 results in a weird bug where the CPP extension reads an off by 1 number.
        self.weight_values_bs   = None
        self.out_cols           = out_features
        del self.weight

    def save_state(self, path):
        # Create path if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'weight_metadata': self.weight_metadata,
            'weight_values': self.weight_values,
            'weight_values_is': self.weight_values_is,
            'weight_values_bs': self.weight_values_bs,
            'out_cols': self.out_cols
        }, path)

    @classmethod
    def load_from_file(cls, path):
        state = torch.load(path)
        new_inst = cls(state['weight_metadata'].shape[0], state['out_cols'], False, 'cpu', torch.bfloat16)
        new_inst.weight_metadata = state['weight_metadata']
        new_inst.weight_values = state['weight_values']
        new_inst.weight_values_is = state['weight_values_is']
        new_inst.weight_values_bs = state['weight_values_bs']
        new_inst.out_cols = state['out_cols']
        return new_inst

    def populate_sparse_weights(self, weights):
        bitmap_tensor = (weights != 0)
        bitmap_list = bitmap_tensor.reshape(-1)
        self.weight_metadata.data = bits_to_int32(bitmap_list)
        self.weight_values.data = weights[bitmap_tensor]

    def populate_sparse_weights_batched(self, weights):
        bitmap_tensor = (weights != 0)
        bitmap_list = bitmap_tensor.reshape(bitmap_tensor.shape[0], bitmap_tensor.shape[1], -1)
        self.weight_metadata.data = bits_to_int32_batched(bitmap_list)
        self.weight_values.data = weights[bitmap_tensor]
        self.weight_values_bs = torch.empty(bitmap_tensor.shape[0], bitmap_tensor.shape[1], dtype=torch.int32)
        offset = 0
        for i in range(bitmap_tensor.shape[0]):
            for j in range(bitmap_tensor.shape[1]):
                # Store the offset from the start of the weight_values tensor for each batch
                self.weight_values_bs[i, j] = offset
                offset += torch.sum(bitmap_tensor[i, j]).item()

    def split_weights_to_sets(self, num_sets):
        num_sets = min(num_sets, math.ceil(self.out_cols / 16.))
        assert (self.out_cols / 16) % num_sets == 0, "num_sets does not divide out_cols evenly."

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

    def split_weights_to_sets_batched(self, num_sets):
        # import pdb; pdb.set_trace()
        num_sets = min(num_sets, math.ceil(self.out_cols / 16.))
        assert (self.out_cols // 16) % num_sets == 0, f"num_sets ({num_sets}) does not divide out_cols ({self.out_cols}) evenly."
        self.weight_values_is.data = torch.empty(self.weight_metadata.shape[0], self.weight_metadata.shape[1], num_sets, dtype=torch.int32)
        for i in range(self.weight_metadata.shape[0]):
            for j in range(self.weight_metadata.shape[1]):
                # import pdb; pdb.set_trace()
                assert self.weight_metadata[i][j].numel() % num_sets == 0, f"Number of sets {num_sets} doesn't divide weights {self.weight_metadata[i][j].numel()} equally."
                metadata_elements_per_set = self.weight_metadata[i][j].numel() // num_sets
                prev = 0
                for k in range(num_sets):
                    num_elements = 0
                    for l in range(metadata_elements_per_set):
                        cnt = bin(int(self.weight_metadata[i][j].data[k * metadata_elements_per_set + l]) & 0xFFFFFFFF).count('1')
                        num_elements+= cnt
                    self.weight_values_is.data[i][j][k] = prev
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

        # import pdb;pdb.set_trace()
        # with torch.autograd.profiler.emit_itt():
        return avx_sparse_linear.forward(input, self.weight_metadata, self.weight_values, self.weight_values_is, self.out_cols)

    def sparse_matmul(self, mat):
        # import pdb; pdb.set_trace()
        return avx_sparse_linear.matmul(mat, self.weight_metadata, self.weight_values, self.weight_values_bs, self.weight_values_is, self.out_cols, self.bias)


    @classmethod
    def batched_matmul(cls, mat1, mat2):
        """Performs Batched Matrix Multiplication using the sparse kernel"""
        input = mat1
        weight = mat2.transpose(2, 3)
        computer = cls.from_batched_weights(weight)
        output = computer.sparse_matmul(input)
        return output
        batch_size, num_heads, out_rows, inner_dim = mat1.size()
        res = torch.empty(batch_size, num_heads, out_rows, mat2.size(-1), device=mat1.device, dtype=mat1.dtype)
        for b in range(batch_size):
            for h in range(num_heads):
                input = mat1[b, h]
                weight = mat2[b, h]
                computer = cls.from_weights(weight.transpose(0, 1))
                output = computer(input.unsqueeze(0)).squeeze(0)
                res[b, h] = output
        return res


    @classmethod
    def from_batched_weights(cls, orig_weights, num_threads=32, shallow=False):

        num_threads = min(num_threads, math.ceil(orig_weights.shape[2] / 16.))
        # import pdb; pdb.set_trace()
        # TODO: not an efficient implementation, we will create another copy of parameters, imagine Billion scale llm!
        # see llm_pipeline.py for a "Wrapper" way
        new_inst = cls(orig_weights.shape[3], orig_weights.shape[2], False, 'cpu', torch.bfloat16)

        if shallow:
            return new_inst
        # import pdb; pdb.set_trace()
        # new_inst.weight.data = nn_linear_inst.weight.data
        # print(f"Before weight conversion: {nn_linear_inst.weight.data}")
        # Convert to AMX format: Group each 2 consecutive elements, then transpose then reshape to have the same original shape.
        # import pdb; pdb.set_trace()
        # weights = nn_linear_inst.weight.data
        weights = orig_weights.view(orig_weights.shape[0], orig_weights.shape[1], orig_weights.shape[2], -1, 2).transpose(2, 3).to(torch.bfloat16)
        weights = weights.reshape(weights.shape[0], weights.shape[1], weights.shape[2], -1).contiguous()

        num_groups = min(NUM_GROUPS, max(weights.shape[3] // 32 // num_threads, 1))
        NUM_ELEMENTS_PER_ROW = 32 * num_groups
        # Reshape weights to have 32 elements per row
        batch, seq, rows, cols = weights.shape
        padded_cols = ((cols + NUM_ELEMENTS_PER_ROW - 1) // NUM_ELEMENTS_PER_ROW) * NUM_ELEMENTS_PER_ROW
        padded_weights = torch.nn.functional.pad(weights, (0, padded_cols - cols))
        reshaped = padded_weights.view(batch, seq, rows, -1, NUM_ELEMENTS_PER_ROW)

        # Transpose and reshape to get the desired order
        reordered = reshaped.transpose(2, 3).reshape(batch, seq, -1, NUM_ELEMENTS_PER_ROW)

        # Flatten the tensor and trim to original size
        weights = reordered.flatten()[:batch * seq * rows * cols].reshape(batch, seq, rows, cols).contiguous()
        # # import pdb; pdb.set_trace()
        # # if weights.shape[1] > 32:
        # #     weights = weights.view(weights.shape[0], -1, )
        # # weights = torch.arange()

        # if weights.shape[1] > 32:
        #     # Will have multiple tiles in the out_cols dimension. Need to reorder the
        #     # weights to be suitable for consumption, putting items of each tile next to each other
        #     # because that's how the Kernel consumes them.
        #     tile_width = 32
        #     tile_height = 16
        #     if weights.shape[1] // num_threads > tile_width:
        #         grouped_col_tiles = True
        #     else:
        #         grouped_col_tiles = False
        #     weights = weights.view(weights.shape[0], -1, tile_width).transpose(0, 1)
        #     if grouped_col_tiles:
        #         weights = weights.view(weights.shape[0], weights.shape[1] // tile_height, tile_height, weights.shape[2]).transpose(0,1)
        #         weights = weights.view(weights.shape[0], weights.shape[1] // 2, 2, weights.shape[2], weights.shape[3]).transpose(0,1)

        # import pdb; pdb.set_trace()

        # TODO: Custom implementation. Must be removed for accurate output
        # new_inst.set_all_zeros(weights)
        new_inst.populate_sparse_weights_batched(weights)
        # import pdb; pdb.set_trace();
        new_inst.split_weights_to_sets_batched(num_threads)

        return new_inst

    @classmethod
    def from_weights(cls, orig_weights, num_threads=32, shallow=False):

        num_threads = min(num_threads, math.ceil(orig_weights.shape[0] / 16.))
        # import pdb; pdb.set_trace()
        # TODO: not an efficient implementation, we will create another copy of parameters, imagine Billion scale llm!
        # see llm_pipeline.py for a "Wrapper" way
        new_inst = cls(orig_weights.shape[1], orig_weights.shape[0], False, 'cpu', torch.bfloat16)

        if shallow:
            return new_inst
        # import pdb; pdb.set_trace()
        # new_inst.weight.data = nn_linear_inst.weight.data
        # print(f"Before weight conversion: {nn_linear_inst.weight.data}")
        # Convert to AMX format: Group each 2 consecutive elements, then transpose then reshape to have the same original shape.
        # import pdb; pdb.set_trace()
        # weights = nn_linear_inst.weight.data
        weights = orig_weights.view(orig_weights.shape[0], -1, 2).transpose(0, 1).to(torch.bfloat16)
        weights = weights.reshape(weights.shape[0], -1).contiguous()

        num_groups = min(NUM_GROUPS, max(weights.shape[1] // 32 // num_threads, 1))
        NUM_ELEMENTS_PER_ROW = 32 * num_groups
        # Reshape weights to have 32 elements per row
        rows, cols = weights.shape
        padded_cols = ((cols + NUM_ELEMENTS_PER_ROW - 1) // NUM_ELEMENTS_PER_ROW) * NUM_ELEMENTS_PER_ROW
        padded_weights = torch.nn.functional.pad(weights, (0, padded_cols - cols))
        reshaped = padded_weights.view(rows, -1, NUM_ELEMENTS_PER_ROW)

        # Transpose and reshape to get the desired order
        reordered = reshaped.transpose(0, 1).reshape(-1, NUM_ELEMENTS_PER_ROW)

        # Flatten the tensor and trim to original size
        weights = reordered.flatten()[:rows * cols].reshape(rows, cols).contiguous()
        # # import pdb; pdb.set_trace()
        # # if weights.shape[1] > 32:
        # #     weights = weights.view(weights.shape[0], -1, )
        # # weights = torch.arange()

        # if weights.shape[1] > 32:
        #     # Will have multiple tiles in the out_cols dimension. Need to reorder the
        #     # weights to be suitable for consumption, putting items of each tile next to each other
        #     # because that's how the Kernel consumes them.
        #     tile_width = 32
        #     tile_height = 16
        #     if weights.shape[1] // num_threads > tile_width:
        #         grouped_col_tiles = True
        #     else:
        #         grouped_col_tiles = False
        #     weights = weights.view(weights.shape[0], -1, tile_width).transpose(0, 1)
        #     if grouped_col_tiles:
        #         weights = weights.view(weights.shape[0], weights.shape[1] // tile_height, tile_height, weights.shape[2]).transpose(0,1)
        #         weights = weights.view(weights.shape[0], weights.shape[1] // 2, 2, weights.shape[2], weights.shape[3]).transpose(0,1)

        # import pdb; pdb.set_trace()

        # TODO: Custom implementation. Must be removed for accurate output
        # new_inst.set_all_zeros(weights)
        new_inst.populate_sparse_weights(weights)
        # import pdb; pdb.set_trace();
        new_inst.split_weights_to_sets(num_threads)
        # import pdb; pdb.set_trace();
        # print(f"After weight conversion: {weights}")
        return new_inst


    @classmethod
    def from_linear(cls, nn_linear_inst: nn.Linear, num_threads=32, shallow=False):

        num_threads = min(num_threads, math.ceil(nn_linear_inst.out_features / 16.))
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
        # weights = nn_linear_inst.weight.data
        weights = nn_linear_inst.weight.data.view(nn_linear_inst.weight.data.shape[0], -1, 2).transpose(0, 1).to(torch.bfloat16)
        weights = weights.reshape(weights.shape[0], -1).contiguous()

        num_groups = min(NUM_GROUPS, max(weights.shape[1] // 32 // num_threads, 1))
        NUM_ELEMENTS_PER_ROW = 32 * num_groups
        # Reshape weights to have 32 elements per row
        rows, cols = weights.shape
        padded_cols = ((cols + NUM_ELEMENTS_PER_ROW - 1) // NUM_ELEMENTS_PER_ROW) * NUM_ELEMENTS_PER_ROW
        padded_weights = torch.nn.functional.pad(weights, (0, padded_cols - cols))
        reshaped = padded_weights.view(rows, -1, NUM_ELEMENTS_PER_ROW)

        # Transpose and reshape to get the desired order
        reordered = reshaped.transpose(0, 1).reshape(-1, NUM_ELEMENTS_PER_ROW)

        # Flatten the tensor and trim to original size
        weights = reordered.flatten()[:rows * cols].reshape(rows, cols).contiguous()
        # # import pdb; pdb.set_trace()
        # # if weights.shape[1] > 32:
        # #     weights = weights.view(weights.shape[0], -1, )
        # # weights = torch.arange()

        # if weights.shape[1] > 32:
        #     # Will have multiple tiles in the out_cols dimension. Need to reorder the
        #     # weights to be suitable for consumption, putting items of each tile next to each other
        #     # because that's how the Kernel consumes them.
        #     tile_width = 32
        #     tile_height = 16
        #     if weights.shape[1] // num_threads > tile_width:
        #         grouped_col_tiles = True
        #     else:
        #         grouped_col_tiles = False
        #     weights = weights.view(weights.shape[0], -1, tile_width).transpose(0, 1)
        #     if grouped_col_tiles:
        #         weights = weights.view(weights.shape[0], weights.shape[1] // tile_height, tile_height, weights.shape[2]).transpose(0,1)
        #         weights = weights.view(weights.shape[0], weights.shape[1] // 2, 2, weights.shape[2], weights.shape[3]).transpose(0,1)

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
