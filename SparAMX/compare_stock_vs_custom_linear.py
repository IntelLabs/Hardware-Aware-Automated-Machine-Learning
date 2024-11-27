import torch
from torch import nn
import time

from layer.dense_linear import DenseLinear 
from statistics import median

torch.set_num_threads(1)
d=4096
o=1024

one_layer_net_torch_stock = nn.Linear(in_features=d, out_features=o, bias=False)

# Make sure all weights are representable in bfloat16.
one_layer_net_torch_stock.weight.data = one_layer_net_torch_stock.weight.data.to(torch.bfloat16).to(torch.float)
# one_layer_net_torch_stock.weight.data = torch.tensor([[1.,1.,2,1], [5.,1,1,1]])
print("\n[Info]: Creating one layer neural network with nn.Linear")
print(f"one_layer_net_torch_stock:\n{one_layer_net_torch_stock}")

one_layer_net_torch_extcpp = DenseLinear.from_linear(one_layer_net_torch_stock)
print("\n[Info]: Creating one layer neural network with DenseFC")
print(f"one_layer_net_torch_extcpp:\n{one_layer_net_torch_extcpp}")

N=1
x = torch.randn(N, d, dtype=torch.bfloat16).to(torch.float)
x_bf = x.to(torch.bfloat16)
# x = torch.tensor([[1., 1, 1, 1]])
print(f"\n[Info]: Creating test input x ({N}, {d})")
print(f"x: {x.shape}\n{x}")

times_stock = []
times_amx = []
with torch.no_grad():
    for i in range(5): # warmup
        x = torch.randn(N, d, dtype=torch.bfloat16).to(torch.float)
        o_torch_stock = one_layer_net_torch_stock(x)
        o_torch_extcpp = one_layer_net_torch_extcpp(x)
    for i in range(1000): # measurement
        x = torch.randn(N, d, dtype=torch.bfloat16).to(torch.float)
        start = time.time()
        o_torch_stock = one_layer_net_torch_stock(x)
        time_stock = time.time() - start
        times_stock.append(time_stock)
        start = time.time()
        o_torch_extcpp = one_layer_net_torch_extcpp(x)
        time_amx = time.time() - start
        times_amx.append(time_amx)

print(f"\none_layer_net_torch_stock(x): {o_torch_stock.shape}\n{o_torch_stock}")
print(f"\none_layer_net_torch_extcpp(x): {o_torch_extcpp.shape}\n{o_torch_extcpp}")

# print(f"\none_layer_net_torch_stock(x) ==  one_layer_net_torch_extcpp(x) ???\n{torch.testing.assert_close(o_torch_extcpp, o_torch_stock)}")
print(f"Time Stock: {median(times_stock)}, Time AMX: {median(times_amx)}")

