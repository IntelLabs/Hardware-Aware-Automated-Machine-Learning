import torch
from torch import nn
import torch.quantization as quant
import pytest

from layer.onednn_linear import OneDnnLinear 


def perform_zero_input_test(out_rows, inner_dim, out_cols):
    one_layer_net_torch_stock = nn.Linear(in_features=inner_dim, out_features=out_cols, bias=False)
    # Make sure all weights are representable in bfloat16 since that's what AMX will operate on.
    one_layer_net_torch_stock.weight.data = one_layer_net_torch_stock.weight.data.to(torch.bfloat16).to(torch.float)
    one_layer_net_torch_extcpp = OneDnnLinear.from_linear(one_layer_net_torch_stock)
    x = torch.zeros(out_rows, inner_dim, dtype=torch.bfloat16).to(torch.float)
    with torch.no_grad():
        o_torch_stock = one_layer_net_torch_stock(x)
        o_torch_extcpp = one_layer_net_torch_extcpp(x)

    assert torch.equal(o_torch_stock, o_torch_extcpp)    

def perform_random_input_test(out_rows, inner_dim, out_cols):
    one_layer_net_torch_stock = nn.Linear(in_features=inner_dim, out_features=out_cols, bias=False)
    # Make sure all weights are representable in bfloat16 since that's what AMX will operate on.
    one_layer_net_torch_stock.weight.data = one_layer_net_torch_stock.weight.data.to(torch.bfloat16).to(torch.float)
    one_layer_net_torch_extcpp = OneDnnLinear.from_linear(one_layer_net_torch_stock)
    x = torch.randn(out_rows, inner_dim, dtype=torch.bfloat16).to(torch.float)
    with torch.no_grad():
        o_torch_stock = one_layer_net_torch_stock(x)
        o_torch_extcpp = one_layer_net_torch_extcpp(x)
    print(o_torch_extcpp)
    print(o_torch_stock)
    torch.testing.assert_close(o_torch_extcpp, o_torch_stock)
    # assert torch.equal(o_torch_stock, o_torch_extcpp)    


test_data = [
    (1, 2, 1),
    (2, 2, 2),
    (3, 2, 3),
    (4, 2, 4),
    (5, 2, 5),
    (1, 4, 8),
    (2, 4, 12),
    (4, 4, 14),
    (6, 4, 16),
]

@pytest.mark.parametrize("out_rows,inner_dim,out_cols", test_data)
def test_zero_inputs_even_d(out_rows,inner_dim,out_cols):
    perform_zero_input_test(out_rows,inner_dim,out_cols)

test_data = [
    (1, 1, 1),
]

@pytest.mark.parametrize("out_rows,inner_dim,out_cols", test_data)
@pytest.mark.skip(reason="Old Inner Dimension is not yet supported because each 2 elements are put in the same row.")
def test_zero_inputs_odd_d(out_rows,inner_dim,out_cols):
    perform_zero_input_test(out_rows,inner_dim,out_cols)

test_data = [
    (1, 2, 36),
]

@pytest.mark.parametrize("out_rows,inner_dim,out_cols", test_data)
def test_zero_inputs_more_than_a_tile(out_rows,inner_dim,out_cols):
    perform_zero_input_test(out_rows,inner_dim,out_cols)


test_data = [
    (1, 2, 1),
    (2, 2, 2),
    (3, 2, 3),
    (4, 2, 4),
    (5, 2, 5),
    (1, 4, 2),
    (2, 4, 12),
    (4, 4, 14),
    (6, 4, 16),
    (8, 4, 16),
    (1, 8, 2),
    (2, 8, 12),
    (4, 8, 14),
    (6, 8, 16),
    (8, 8, 16),    
    (1, 16, 2),
    (2, 16, 12),
    (4, 16, 14),
    (6, 16, 16),
    (8, 16, 16),
    (1, 32, 2),
    (2, 32, 12),
    (4, 32, 14),
    (6, 32, 16),
    (16, 32, 16), 
]

@pytest.mark.parametrize("out_rows,inner_dim,out_cols", test_data)
def test_random_inputs_even_d(out_rows,inner_dim,out_cols):
    perform_random_input_test(out_rows,inner_dim,out_cols)

test_data = [
    (32, 2, 1),
    (32, 2, 8),
    (32, 2, 16),
    (32, 4, 1),
    (32, 4, 16),
    (32, 32, 1),
    (32, 32, 16),
    (64, 32, 16),
    (128, 32, 16),
    (512, 32, 16),
    (1, 2, 32),
    (2, 2, 32),
    (4, 2, 32),
    (32, 2, 32),
    (512, 2, 32),
    (1, 2, 64),
    (1, 2, 512),
    (128, 2, 128),
    (1, 4, 32),
    (32, 4, 32),
    (128, 4, 128),
    (64, 32, 64),
    (1, 64, 1),
    (2, 64, 1),
    (2, 64, 2),
    (16, 64, 16),
    (32, 64, 32),
    (128, 128, 128),
    (1024, 1024, 128),
    # (16384, 16384, 16384),
]

@pytest.mark.parametrize("out_rows,inner_dim,out_cols", test_data)
# @pytest.mark.skip(reason="More than 1 tile is not supported yet.")
def test_random_inputs_more_than_a_tile_all_full(out_rows,inner_dim,out_cols):
    perform_random_input_test(out_rows,inner_dim,out_cols)

test_data = [
    (18, 2, 1),
]

@pytest.mark.parametrize("out_rows,inner_dim,out_cols", test_data)
@pytest.mark.skip(reason="Partial tiles are not supported yet.")
def test_random_inputs_more_than_a_tile_partial_tiles(out_rows,inner_dim,out_cols):
    perform_random_input_test(out_rows,inner_dim,out_cols)
