import torch
from torch import nn
import torch.quantization as quant
import pytest
import itertools

from layer.avx_sparse_linear import AvxSparseLinear 
torch.set_num_threads(1)

def set_weights_to_zero(weights, percentage):
    # Calculate the number of weights to set to 0
    num_weights = weights.numel()
    num_zeros = int(num_weights * percentage / 100.0)
    
    # Create a mask tensor
    mask = torch.ones_like(weights)
    
    # Randomly select indices to set to 0
    zero_indices = torch.randperm(num_weights)[:num_zeros]
    
    # Set the selected indices in the mask to 0
    mask.view(-1)[zero_indices] = 0
    
    # Apply the mask to the weights
    return weights * mask


test_data = [
    (1,16,16),
    (1,32,32),
    (1,128,128),
    (1,1024,1024),
    (1,128,32),
    (1,1024,32),
    (1,128,64),
    (1,128,1024),
    (1,128,16),
]

@pytest.mark.parametrize("out_rows,inner_dim,out_cols", test_data)
def test_matmul_using_kernel(out_rows, inner_dim, out_cols):
    input = torch.randn(out_rows, inner_dim, dtype=torch.bfloat16)
    weights = torch.randn(inner_dim, out_cols, dtype=torch.bfloat16)
    ans = torch.matmul(input, weights)
    computer = AvxSparseLinear.from_weights(weights.transpose(0, 1))
    output = computer(input.unsqueeze(0)).squeeze(0)
    # import pdb; pdb.set_trace()
    torch.testing.assert_close(ans, output)

test_data = [
    (1,1024,1024,1),
    (1,1024,1024,20),
    (1,1024,1024,50),
    (1,1024,1024,90),
]

@pytest.mark.parametrize("out_rows,inner_dim,out_cols,percentage", test_data)
def test_sparse_matmul_using_kernel(out_rows, inner_dim, out_cols, percentage):
    input = torch.randn(out_rows, inner_dim, dtype=torch.bfloat16)
    weights = torch.randn(inner_dim, out_cols, dtype=torch.bfloat16)
    weights = set_weights_to_zero(weights, percentage)
    ans = torch.matmul(input, weights)
    computer = AvxSparseLinear.from_weights(weights.transpose(0, 1))
    output = computer(input.unsqueeze(0)).squeeze(0)
    # import pdb; pdb.set_trace()
    torch.testing.assert_close(ans, output)



test_data = [
    (1, 32, 16, 16),
    # (1, 32, 2, 16), # Need to fix the kernel to handle this case. It's caused by filling multiple rows in the same iteration which assumes that all rows will be filled.
    (1, 32, 32, 16),
    (1, 32, 16, 128),
    (1, 32, 128, 128),
    (1, 32, 256, 128),
    (1, 32, 512, 128),
    (1, 32, 1024, 128),
    (1, 128, 16, 4096),
]

@pytest.mark.parametrize("batch_size,num_heads,seq_len,head_dim", test_data)
def test_attention_matrix_computation(batch_size, num_heads, seq_len, head_dim):
    query_states = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    ans = torch.matmul(query_states, key_states.transpose(2,3))
    output = AvxSparseLinear.batched_matmul(query_states, key_states.transpose(2,3))
    torch.testing.assert_close(ans, output)

test_data = [
    (1, 32, 16, 128, 10),
    (1, 32, 16, 128, 20),
    (1, 32, 16, 128, 50),
    (1, 32, 16, 128, 90),
    (1, 32, 1024, 128, 10),
    (1, 32, 1024, 128, 20),
    (1, 32, 1024, 128, 50),
    (1, 32, 1024, 128, 90),
]

@pytest.mark.parametrize("batch_size,num_heads,seq_len,head_dim,percentage", test_data)
def test_sparse_attention_matrix_computation(batch_size, num_heads, seq_len, head_dim, percentage):
    query_states = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key_states = set_weights_to_zero(key_states, 50)
    ans = torch.matmul(query_states, key_states.transpose(2,3))
    output = AvxSparseLinear.batched_matmul(query_states, key_states.transpose(2,3))
    torch.testing.assert_close(ans, output)

