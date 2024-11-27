import torch
from torch import nn
import time
import os

# Set OpenMP environment variable for thread affinity
os.environ['OMP_PROC_BIND'] = 'true'  # Bind threads to cores
os.environ['OMP_PLACES'] = 'cores'  # Each thread will be bound to a core

from layer.sparse_linear import SparseLinear 
from statistics import median

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


num_layers = 4
sparsity_percentage = 50
# torch.set_num_threads(1)
d=4096
o=14336
stock_layers = []
extcpp_layers = []
for _ in range(num_layers):
    one_layer_net_torch_stock = nn.Linear(in_features=d, out_features=o, bias=False)
    set_weights_to_zero(one_layer_net_torch_stock, sparsity_percentage)
    # Make sure all weights are representable in bfloat16.f
    one_layer_net_torch_stock.weight.data = one_layer_net_torch_stock.weight.data.to(torch.bfloat16)
    stock_layers.append(one_layer_net_torch_stock)
    one_layer_net_torch_extcpp = SparseLinear.from_linear(one_layer_net_torch_stock)
    extcpp_layers.append(one_layer_net_torch_extcpp)



# one_layer_net_torch_stock.weight.data = torch.tensor([[0.] * 2 + [1] * 2] * 32)
print("\n[Info]: Creating one layer neural network with nn.Linear")
print(f"one_layer_net_torch_stock:\n{one_layer_net_torch_stock}")

print("\n[Info]: Creating one layer neural network with DenseFC")
print(f"one_layer_net_torch_extcpp:\n{one_layer_net_torch_extcpp}")
B=1
N=1
x = torch.randn(B, N, d, dtype=torch.bfloat16).to(torch.float)
# x = torch.tensor([[1.] * 4])
x_bf = x.to(torch.bfloat16)

print(f"\n[Info]: Creating test input x ({N}, {d})")
print(f"x: {x.shape}\n{x}")

times_stock = []
times_amx = []
with torch.no_grad():
    for i in range(1): # warmup
        initial_x = torch.randn(B, N, d, dtype=torch.bfloat16)
        x = initial_x
        for layer in stock_layers:
            x = layer(initial_x)
        x = initial_x
        for layer in extcpp_layers:
            x = layer(initial_x)

    for i in range(1): # measurement
        initial_x = torch.randn(B, N, d, dtype=torch.bfloat16)
        x = initial_x
        start = time.time()
        for layer in stock_layers:
            x = layer(initial_x)
        o_torch_stock = x
        time_stock = time.time() - start
        times_stock.append(time_stock)
        x = initial_x
        import pdb; pdb.set_trace()
        for _ in range(1000):
            start = time.time()
            for layer in extcpp_layers:
                x = layer(initial_x)
            o_torch_extcpp = x
            time_amx = time.time() - start
            times_amx.append(time_amx)

# torch.set_printoptions(threshold=100000000000)

print(f"\none_layer_net_torch_stock(x): {o_torch_stock.shape}\n{o_torch_stock}")
print(f"\none_layer_net_torch_extcpp(x): {o_torch_extcpp.shape}\n{o_torch_extcpp}")

print(f"\none_layer_net_torch_stock(x) ==  one_layer_net_torch_extcpp(x) ???\n{torch.equal(o_torch_stock, o_torch_extcpp)}")
print(f"Time Stock: {median(times_stock)}, Time AMX: {median(times_amx)}")
