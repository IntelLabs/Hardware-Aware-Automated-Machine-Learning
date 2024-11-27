import torch
from torch import nn
import torch.quantization as quant
import pytest

from layer.quantized_dense_linear import QuantizedDenseLinear 

# torch.set_num_threads(1)

def replace_zeros_with_random(tensor):
    # Create a mask for zero elements
    zero_mask = tensor == 0
    # import pdb; pdb.set_trace()
    # Generate random numbers with the same shape as the input tensor
    random_tensor = torch.randint_like(tensor,-127,128)
    
    # Replace zeros with random numbers
    tensor[zero_mask] = random_tensor[zero_mask]
    # import pdb; pdb.set_trace()

    return tensor

def set_weights_to_zero(layer, percentage):
    # Get the weight tensor
    weights = layer.weight.data
    
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
    layer.weight.data *= mask

def perform_zero_input_test(batch_size, out_rows, inner_dim, out_cols):
    one_layer_net_torch_stock = nn.Linear(in_features=inner_dim, out_features=out_cols, bias=False, dtype=torch.bfloat16)
    one_layer_net_torch_stock.weight.data = torch.randint(-127,128, (out_cols,inner_dim), dtype=torch.bfloat16)
    # Make sure all weights are representable in bfloat16 since that's what AMX will operate on.
    one_layer_net_torch_extcpp = QuantizedDenseLinear.from_linear(one_layer_net_torch_stock)
    x = torch.zeros(batch_size, out_rows, inner_dim, dtype=torch.bfloat16)
    with torch.no_grad():
        o_torch_stock = one_layer_net_torch_stock(x)
        o_torch_extcpp = one_layer_net_torch_extcpp(x.to(torch.int8))
    # import pdb; pdb.set_trace()

    assert torch.equal(o_torch_stock, o_torch_extcpp)    

def perform_random_input_test(batch_size, out_rows, inner_dim, out_cols):
    one_layer_net_torch_stock = nn.Linear(in_features=inner_dim, out_features=out_cols, bias=False, dtype=torch.bfloat16)
    one_layer_net_torch_stock.weight.data = torch.randint(-127,128, (out_cols,inner_dim), dtype=torch.bfloat16)
    one_layer_net_torch_stock.weight.data = replace_zeros_with_random(one_layer_net_torch_stock.weight.data)
    # Make sure all weights are representable in bfloat16 since that's what AMX will operate on.
    one_layer_net_torch_extcpp = QuantizedDenseLinear.from_linear(one_layer_net_torch_stock)
    x = torch.randint(-127,128,(batch_size, out_rows, inner_dim), dtype=torch.bfloat16)
    with torch.no_grad():
        o_torch_stock = one_layer_net_torch_stock(x)
        o_torch_extcpp = one_layer_net_torch_extcpp(x.to(torch.int8))

    torch.testing.assert_close(o_torch_extcpp, o_torch_stock)
    # assert torch.equal(o_torch_stock, o_torch_extcpp)    


def perform_random_input_test_with_sparsity(batch_size, out_rows, inner_dim, out_cols, sparsity_percentage):
    one_layer_net_torch_stock = nn.Linear(in_features=inner_dim, out_features=out_cols, bias=False, dtype=torch.bfloat16)
    one_layer_net_torch_stock.weight.data = torch.randint(-127,128, (out_cols,inner_dim), dtype=torch.bfloat16)
    # Make sure all weights are representable in bfloat16 since that's what AMX will operate on.
    set_weights_to_zero(one_layer_net_torch_stock, sparsity_percentage)
    one_layer_net_torch_extcpp = QuantizedDenseLinear.from_linear(one_layer_net_torch_stock)
    x = torch.randint(-127,128,(batch_size, out_rows, inner_dim), dtype=torch.bfloat16)
    with torch.no_grad():
        o_torch_stock = one_layer_net_torch_stock(x)
        o_torch_extcpp = one_layer_net_torch_extcpp(x.to(torch.int8))

    torch.testing.assert_close(o_torch_extcpp, o_torch_stock)
    # assert torch.equal(o_torch_stock, o_torch_extcpp)  

test_data = [
    (1, 1, 64, 16),
    (1, 1, 128, 16),
    (1, 1, 256, 16),
    (1, 1, 512, 16),
    (1, 1, 4096, 16),
    (1, 16, 64, 16),
    (1, 16, 128, 16),
    (1, 16, 256, 16),
    (1, 16, 512, 16),
    (1, 16, 4096, 16),
    (1, 512, 64, 16),
    (1, 512, 128, 16),
    (1, 512, 256, 16),
    (1, 512, 512, 16),
    (1, 512, 4096, 16),
    (1, 512, 4096, 512),
    (1, 512, 4096, 1024),

]

@pytest.mark.parametrize("batch_size,out_rows,inner_dim,out_cols", test_data)
def test_zero_inputs_even_d(batch_size, out_rows,inner_dim,out_cols):
    perform_zero_input_test(batch_size, out_rows,inner_dim,out_cols)

test_data = [
    (1, 1, 64, 16),
    (1, 1, 128, 16),
    (1, 1, 256, 16),
    (1, 1, 512, 16),
    (1, 1, 4096, 16),
    (1, 16, 64, 16),
    (1, 16, 128, 16),
    (1, 16, 256, 16),
    (1, 16, 512, 16),
    (1, 16, 4096, 16),
    (1, 512, 64, 16),
    (1, 512, 128, 16),
    (1, 512, 256, 16),
    (1, 512, 512, 16),
    (1, 512, 4096, 16),
    (1, 512, 4096, 512),
    (1, 512, 4096, 1024),

]

@pytest.mark.parametrize("batch_size,out_rows,inner_dim,out_cols", test_data)
@pytest.mark.skip(reason="Odd Inner Dimension is not yet supported because each 2 elements are put in the same row.")
def test_zero_inputs_odd_d(batch_size,out_rows,inner_dim,out_cols):
    perform_zero_input_test(batch_size,out_rows,inner_dim,out_cols)

test_data = [
    (1, 1, 64, 16),
    (1, 1, 128, 16),
    (1, 1, 256, 16),
    (1, 1, 512, 16),
    (1, 1, 4096, 16),
    (1, 16, 64, 16),
    (1, 16, 128, 16),
    (1, 16, 256, 16),
    (1, 16, 512, 16),
    (1, 16, 4096, 16),
    (1, 512, 64, 16),
    (1, 512, 128, 16),
    (1, 512, 256, 16),
    (1, 512, 512, 16),
    (1, 512, 4096, 16),
    (1, 512, 4096, 512),
    (1, 512, 4096, 1024),

]

@pytest.mark.parametrize("batch_size,out_rows,inner_dim,out_cols", test_data)
def test_random_inputs_even_d(batch_size,out_rows,inner_dim,out_cols):
    perform_random_input_test(batch_size,out_rows,inner_dim,out_cols)

test_data = [
    (1, 1, 64, 16),
    (1, 1, 128, 16),
    (1, 1, 256, 16),
    (1, 1, 512, 16),
    (1, 1, 4096, 16),
    (1, 16, 64, 16),
    (1, 16, 128, 16),
    (1, 16, 256, 16),
    (1, 16, 512, 16),
    (1, 16, 4096, 16),
    (1, 512, 64, 16),
    (1, 512, 128, 16),
    (1, 512, 256, 16),
    (1, 512, 512, 16),
    (1, 512, 4096, 16),
    (1, 512, 4096, 512),
    (1, 512, 4096, 1024),

    # (1, 1, 4096 * 4, 4096),
    # (1, 1, 4096 * 4, 4096), # Added 3 times because this case before using int64 type would randomly fail.
    # (1, 1, 4096 * 4, 4096),
]

@pytest.mark.parametrize("batch_size,out_rows,inner_dim,out_cols", test_data)
# @pytest.mark.skip(reason="More than 1 tile is not supported yet.")
def test_random_inputs_more_than_a_tile_all_full(batch_size,out_rows,inner_dim,out_cols):
    perform_random_input_test(batch_size,out_rows,inner_dim,out_cols)

test_data = [
    (1, 1, 64, 16),
    (1, 1, 128, 16),
    (1, 1, 256, 16),
    (1, 1, 512, 16),
    (1, 1, 4096, 16),
    (1, 16, 64, 16),
    (1, 16, 128, 16),
    (1, 16, 256, 16),
    (1, 16, 512, 16),
    (1, 16, 4096, 16),
    (1, 512, 64, 16),
    (1, 512, 128, 16),
    (1, 512, 256, 16),
    (1, 512, 512, 16),
    (1, 512, 4096, 16),
    (1, 512, 4096, 512),
    (1, 512, 4096, 1024),

]

@pytest.mark.parametrize("batch_size,out_rows,inner_dim,out_cols", test_data)
@pytest.mark.skip(reason="Partial tiles are not supported yet.")
def test_random_inputs_more_than_a_tile_partial_tiles(batch_size,out_rows,inner_dim,out_cols):
    perform_random_input_test(batch_size,out_rows,inner_dim,out_cols)


test_data = [
    (1, 1, 64, 16),
    (1, 1, 128, 16),
    (1, 1, 256, 16),
    (1, 1, 512, 16),
    (1, 1, 4096, 16),
    (1, 16, 64, 16),
    (1, 16, 128, 16),
    (1, 16, 256, 16),
    (1, 16, 512, 16),
    (1, 16, 4096, 16),
    (1, 512, 64, 16),
    (1, 512, 128, 16),
    (1, 512, 256, 16),
    (1, 512, 512, 16),
    (1, 512, 4096, 16),
    (1, 512, 4096, 512),
    (1, 512, 4096, 1024),
    # llama layers
    (16, 1, 4096, 4096),
    (16, 1, 4096, 1024),
    (16, 1, 4096, 14336),
    (16, 1, 14336, 4096),
    # (16, 1024, 1024*4, 1024*4),
]

@pytest.mark.parametrize("batch_size,out_rows,inner_dim,out_cols", test_data)
# @pytest.mark.skip(reason="More than 1 tile is not supported yet.")
def test_random_inputs_more_than_a_tile_with_20p_sparsity(batch_size,out_rows,inner_dim,out_cols):
    perform_random_input_test_with_sparsity(batch_size,out_rows,inner_dim,out_cols,20)

@pytest.mark.parametrize("batch_size,out_rows,inner_dim,out_cols", test_data)
# @pytest.mark.skip(reason="More than 1 tile is not supported yet.")
def test_random_inputs_more_than_a_tile_with_50p_sparsity(batch_size,out_rows,inner_dim,out_cols):
    perform_random_input_test_with_sparsity(batch_size,out_rows,inner_dim,out_cols,50)


@pytest.mark.parametrize("batch_size,out_rows,inner_dim,out_cols", test_data)
# @pytest.mark.skip(reason="More than 1 tile is not supported yet.")
def test_random_inputs_more_than_a_tile_with_80p_sparsity(batch_size,out_rows,inner_dim,out_cols):
    perform_random_input_test_with_sparsity(batch_size,out_rows,inner_dim,out_cols,80)

@pytest.mark.parametrize("batch_size,out_rows,inner_dim,out_cols", test_data)
# @pytest.mark.skip(reason="More than 1 tile is not supported yet.")
def test_random_inputs_more_than_a_tile_with_100p_sparsity(batch_size,out_rows,inner_dim,out_cols):
    perform_random_input_test_with_sparsity(batch_size,out_rows,inner_dim,out_cols,100)

