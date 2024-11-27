import torch
from torch import nn
import time

from layer.onednn_linear import OneDnnLinear 
from statistics import median

torch.set_num_threads(1)
d=14336
o=4096

execute_custom_layer = True
execute_stock_layer = False

one_layer_net_torch_stock = nn.Linear(in_features=d, out_features=o, bias=False)

# Make sure all weights are representable in bfloat16.
one_layer_net_torch_stock.weight.data = one_layer_net_torch_stock.weight.data.to(torch.bfloat16).to(torch.float)
# one_layer_net_torch_stock.weight.data = torch.tensor([[0.,0]])
# import pdb; pdb.set_trace()
print("\n[Info]: Creating one layer neural network with nn.Linear")
print(f"one_layer_net_torch_stock:\n{one_layer_net_torch_stock}")

one_layer_net_torch_extcpp = OneDnnLinear.from_linear(one_layer_net_torch_stock)
print("\n[Info]: Creating one layer neural network with DenseFC")
print(f"one_layer_net_torch_extcpp:\n{one_layer_net_torch_extcpp}")

N=1
x = torch.randn(N, d, dtype=torch.bfloat16).to(torch.float)
# x_bf = x.to(torch.bfloat16)
# x = torch.tensor([[1., 0]])
# import pdb; pdb.set_trace()
print(f"\n[Info]: Creating test input x ({N}, {d})")
print(f"x: {x.shape}\n{x}")

times_stock = []
times_amx = []
with torch.no_grad():
    for i in range(5): # warmup
        x = torch.randn(N, d, dtype=torch.bfloat16).to(torch.float)
        if execute_stock_layer:
            o_torch_stock = one_layer_net_torch_stock(x)
        if execute_custom_layer:
            o_torch_extcpp = one_layer_net_torch_extcpp(x)
    for i in range(1000): # measurement
        x = torch.randn(N, d, dtype=torch.bfloat16).to(torch.float)
        start = time.time()
        if execute_stock_layer:
            o_torch_stock = one_layer_net_torch_stock(x)
        time_stock = time.time() - start
        times_stock.append(time_stock)
        start = time.time()
        if execute_custom_layer:
            o_torch_extcpp = one_layer_net_torch_extcpp(x)
        time_amx = time.time() - start
        times_amx.append(time_amx)

# print(f"\none_layer_net_torch_stock(x): {o_torch_stock.shape}\n{o_torch_stock}")
# print(f"\none_layer_net_torch_extcpp(x): {o_torch_extcpp.shape}\n{o_torch_extcpp}")
if execute_stock_layer and execute_custom_layer:
    torch.testing.assert_close(o_torch_extcpp, o_torch_stock)

if execute_stock_layer:
    print(f"Time Stock: {median(times_stock)}\n")
if execute_custom_layer:
    print(f"Time AMX: {median(times_amx)}\n")
