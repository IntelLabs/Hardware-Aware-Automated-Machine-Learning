import torch
import torch.quantization as quant
import pytest
import itertools
import torch.nn.functional as F


from layer.sparse_linear import SparseLinear 

# Constants
TILE_ROW_SIZE = 32

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
    # (16,16,16), # Too small for our assumptions of handling big cases.
    (32,32,32),
    (128,128,128),
    (1024,1024,1024),
    (32,128,32),
    (32,1024,32),
    (32,128,64),
    (32,128,1024),
    (1024,128,16),
]

@pytest.mark.parametrize("out_rows,inner_dim,out_cols", test_data)
def test_matmul_using_kernel(out_rows, inner_dim, out_cols):
    input = torch.randn(out_rows, inner_dim, dtype=torch.bfloat16)
    weights = torch.randn(inner_dim, out_cols, dtype=torch.bfloat16)
    ans = torch.matmul(input, weights)
    computer = SparseLinear.from_weights(weights.transpose(0, 1))
    output = computer(input.unsqueeze(0)).squeeze(0)
    # import pdb; pdb.set_trace()
    torch.testing.assert_close(ans, output)

test_data = [
    (1024,1024,1024,1),
    (1024,1024,1024,20),
    (1024,1024,1024,50),
    (1024,1024,1024,90),
]

@pytest.mark.parametrize("out_rows,inner_dim,out_cols,percentage", test_data)
def test_sparse_matmul_using_kernel(out_rows, inner_dim, out_cols, percentage):
    input = torch.randn(out_rows, inner_dim, dtype=torch.bfloat16)
    weights = torch.randn(inner_dim, out_cols, dtype=torch.bfloat16)
    weights = set_weights_to_zero(weights, percentage)
    ans = torch.matmul(input, weights)
    computer = SparseLinear.from_weights(weights.transpose(0, 1))
    output = computer(input.unsqueeze(0)).squeeze(0)
    # import pdb; pdb.set_trace()
    torch.testing.assert_close(ans, output)



test_data = [
    # (1, 32, 16, 16),
    # (1, 32, 2, 16), # Need to fix the kernel to handle this case. It's caused by filling multiple rows in the same iteration which assumes that all rows will be filled.
    # (1, 32, 32, 16),
    # (1, 32, 16, 128),
    (1, 32, 128, 128),
    (1, 32, 256, 128),
    (1, 32, 512, 128),
    (1, 32, 1024, 128),
    # (16, 128, 16, 4096),
]

@pytest.mark.parametrize("batch_size,num_heads,seq_len,head_dim", test_data)
def test_attention_matrix_computation(batch_size, num_heads, seq_len, head_dim):
    query_states = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    ans = torch.matmul(query_states, key_states.transpose(2,3))
    output = SparseLinear.batched_matmul(query_states, key_states.transpose(2,3))
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
    output = SparseLinear.batched_matmul(query_states, key_states.transpose(2,3))
    torch.testing.assert_close(ans, output)


test_data = [
    (1, 32, 64, 128, 10),
    (1, 32, 64, 128, 20),
    (1, 32, 64, 128, 50),
    (1, 32, 64, 128, 90),
    (1, 32, 1024, 128, 10),
    (1, 32, 1024, 128, 20),
    (1, 32, 1024, 128, 50),
    (1, 32, 1024, 128, 90),
]

@pytest.mark.parametrize("batch_size,num_heads,seq_len,head_dim,percentage", test_data)
def test_sparse_attention_values_computation(batch_size, num_heads, seq_len, head_dim, percentage):
    attention_values = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=torch.bfloat16)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    value_states = set_weights_to_zero(value_states, percentage)
    ans = torch.matmul(attention_values, value_states)
    output = SparseLinear.batched_matmul(attention_values, value_states)
    torch.testing.assert_close(ans, output)

# Add these new test cases
test_data_extra_weights = [
    # (batch_size, num_heads, seq_len, head_dim, extra_cols, percentage)
    (1, 32, 128, 128, 16, 20),
    (1, 32, 256, 128, 32, 50),
    (1, 32, 512, 128, 64, 90),
]

@pytest.mark.parametrize("batch_size,num_heads,seq_len,head_dim,extra_cols,percentage", test_data_extra_weights)
def test_sparse_attention_with_extra_weights_as_cols(batch_size, num_heads, seq_len, head_dim, extra_cols, percentage):
    # Create main tensors
    query_states = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key_states = set_weights_to_zero(key_states, percentage)
    
    # Create extra weights tensor
    extra_weights = torch.randn(batch_size, num_heads, extra_cols, head_dim, dtype=torch.bfloat16)
    
    new_key_states = torch.cat([key_states, extra_weights], dim=2)
    # Compute expected result
    ans = torch.matmul(query_states, new_key_states.transpose(2,3))

    prepared_extra_weights = prepare_extra_weights(extra_weights, is_key=True)
    
    # Compute actual result using SparseLinear
    output = SparseLinear.batched_matmul(
        query_states, 
        key_states.transpose(2,3), 
        extra_weights=prepared_extra_weights,
        is_key=True
    )
    
    # Assert results match
    torch.testing.assert_close(ans, output)

def prepare_extra_weights(extra_weights, is_key=False):
    if is_key:
        # Clone the input tensor
        res = extra_weights.clone()
        
        # Get original shapes
        batch, heads, seq_len, dim = res.shape
        
        # First reshape maintaining the original sequence dimension
        res = res.view(batch, heads, seq_len, dim // 2, 2)
        
        # Transpose the sequence and dim//2 dimensions
        res = res.permute(0, 1, 3, 2, 4)
        
        # Final reshape
        res = res.reshape(batch, heads, dim, seq_len)
        
        # Convert to bfloat16 and ensure contiguous memory layout
        res = res.to(torch.bfloat16).contiguous()
        
        return res
    return extra_weights

# Add a simpler test case for debugging
def test_simple_extra_weights():
    # Small dimensions for easier debugging
    batch_size, num_heads, seq_len, head_dim, extra_cols = 1, 1, 64, 32, 8
    
    # Create tensors with simple patterns for debugging
    query_states = torch.ones(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key_states = torch.ones(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    extra_weights = torch.randn(batch_size, num_heads, extra_cols, head_dim, dtype=torch.bfloat16)
    # Change items of the last column to 0
    # extra_weights[0,0,-1] = 1
    # extra_weights[0,0,0,2] = 0
    
    # Compute expected result
    # main_result = torch.matmul(query_states, key_states.transpose(2,3))
    # extra_result = torch.matmul(query_states, extra_weights.transpose(2,3))

    new_key_states = torch.cat([key_states, extra_weights], dim=2)
    ans = torch.matmul(query_states, new_key_states.transpose(2,3))

    # ans = torch.cat([main_result, extra_result], dim=-1)
    
    prepared_extra_weights = prepare_extra_weights(extra_weights, is_key=True)
    # prepared_extra_weights = extra_weights.transpose(2,3)
    # import pdb; pdb.set_trace()
    # breakpoint()

    # Compute actual result
    output = SparseLinear.batched_matmul(
        query_states,
        key_states.transpose(2,3),
        extra_weights=prepared_extra_weights,
        is_key=True
    )
    # import pdb; pdb.set_trace()
    # Assert results match
    torch.testing.assert_close(ans, output)

def prepare_extra_values(extra_values):
    # Pad the values to have full tiles
    # A full tile has TILE_SIZE rows
    # import pdb; pdb.set_trace()
    res = extra_values.clone() # TODO: Can we optimize this?
    if extra_values.size(2) % TILE_ROW_SIZE != 0:
        res = F.pad(extra_values, (0, 0, 0, TILE_ROW_SIZE - extra_values.size(2) % TILE_ROW_SIZE))
    res = res.transpose(2, 3)
    # reorder elements so that each word of the sequence is in the same row
    batch, heads, dim, seq_len = res.shape
        
    # First reshape maintaining the original sequence dimension
    res = res.view(batch, heads, dim, seq_len // 2, 2)
    
    # Transpose the sequence and dim//2 dimensions
    res = res.permute(0, 1, 3, 2, 4)
    
    # Final reshape
    res = res.reshape(batch, heads, seq_len, dim)
    
    # Convert to bfloat16 and ensure contiguous memory layout
    res = res.to(torch.bfloat16).contiguous()
    return res


# Add these new test cases for value states
test_data_extra_weights_value = [
    # (batch_size, num_heads, seq_len, head_dim, extra_rows, percentage)
    (1, 32, 128, 128, 16, 20),
    (1, 32, 256, 128, 32, 50),
    (1, 32, 512, 128, 64, 90),
    (1, 32, 1024, 128, 1024, 90),
]

@pytest.mark.parametrize("batch_size,num_heads,seq_len,head_dim,extra_rows,percentage", test_data_extra_weights_value)
def test_sparse_attention_with_extra_weights_as_rows(batch_size, num_heads, seq_len, head_dim, extra_rows, percentage):
    # Create attention scores (result of QK multiplication)
    attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len + extra_rows, dtype=torch.bfloat16)
    # Create value states and extra weights
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    value_states = set_weights_to_zero(value_states, percentage)
    extra_weights = torch.randn(batch_size, num_heads, extra_rows, head_dim, dtype=torch.bfloat16)
    
    # Concatenate along the sequence length dimension (dim=2)
    new_value_states = torch.cat([value_states, extra_weights], dim=2)
    
    # Compute expected result
    ans = torch.matmul(attention_scores, new_value_states)
    
    # Compute actual result using SparseLinear
    output = SparseLinear.batched_matmul(
        attention_scores, 
        value_states, 
        extra_weights=prepare_extra_values(extra_weights),
        is_key=False
    )
    
    # Assert results match
    torch.testing.assert_close(ans, output)

# Add a simpler test case for debugging value states
def test_simple_extra_weights_value():
    # Small dimensions for easier debugging
    batch_size, num_heads, seq_len, head_dim, extra_rows = 1, 1, 2048, 1024, 256
    
    # Create attention scores with simple pattern
    attention_scores = torch.randn(batch_size, num_heads, 1, seq_len + extra_rows, dtype=torch.bfloat16)
    # Add extra_rows to the sequence length but have their values as zeros in attention scores.
    # attention_scores = torch.cat([attention_scores, torch.randn(batch_size, num_heads, 1, extra_rows, dtype=torch.bfloat16)], dim=3)
    
    # Create value states and extra weights with simple patterns
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    extra_weights = torch.randn(batch_size, num_heads, extra_rows, head_dim, dtype=torch.bfloat16)


    
    # Concatenate along sequence length dimension
    new_value_states = torch.cat([value_states, extra_weights], dim=2)
    
    # Compute expected result
    ans = torch.matmul(attention_scores, new_value_states)
    
    # Compute actual result
    output = SparseLinear.batched_matmul(
        attention_scores,
        value_states,
        # extra_weights=None,
        extra_weights=prepare_extra_values(extra_weights),
        is_key=False
    )
    

    # import pdb; pdb.set_trace()
    # Assert results match
    torch.testing.assert_close(ans, output)


