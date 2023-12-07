# from nas_101_api.model_fetchall import Network
# from nasbench import api
from nas_201_api import NASBench201API as API
import numpy as np
import random 
from dataset_utils import *
import argparse
from torchsummary import summary
from models import get_cell_based_tiny_net 

'''
Dataframe details for evaluation:
# score_set will have per layer score of every network
# sample accuracies will be the NASBench-101 accuracies of the networks

score_set = {}
score_scalar_set = {}
Net_Sample (0 --> Number of networks for tau metric) [sample]
    score = []
    Layer_ID (0 --> Number of layers in the sample)  [id]
        score.append(eval_layer(individual, sample, id))
    score_set[sample] = score
    score_scalar_set.append(sum(score)/num_layers)

kendall_tau(score_scalar_set, sample_accuracies)
'''
batch_size = 1
NASBENCH_TFRECORD = 'NAS-Bench-201-v1_1-096897.pth'
api = API(NASBENCH_TFRECORD, verbose=False)
NUM_NETS = 500
dataset = 'ImageNet16-120'
if dataset=='cifar100':
    train_loader = get_cifar100_train_loader(batch_size=batch_size)
elif dataset=='cifar10':
    train_loader = get_train_loader(batch_size=batch_size)
elif dataset=='ImageNet16-120':
    train_loader = get_imagenet_train_loader(batch_size=batch_size)

    
args = get_args()
test = False
print("NASBench TFRecord loaded.")

# A simple hook that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

# Sample NUM_NETS different architectures
'''
./networks/
    ./0.pt
        'accuracy'
        'layer0'
            'wt': [..]
            'wt_grad_data': [..]
            .
            .
            'act_grad_data_perturb': [..]
        'layer1'
            'wt': [..]
            'wt_grad_data': [..]
            .
            .
            'act_grad_data_perturb': [..]
        .
        .
        'layern.pt'
            'wt': [..]
            'wt_grad_data': [..]
            .
            .
            'act_grad_data_perturb': [..]
    ./1
    ./2
    ./3
    ./4
    .
    .
    .
    ./100
'''
### Notes ###
# Binary address space for accessing memory and previous math ops
# About 22 net statistic addresses, rest for math op outputs
# [22 static]  [20 wtaddr] [20 inpaddr] [20 preactaddr] [20 actaddr] [20 scalar] : 122 addresses
# static addr: [wt, wtgraddata, wtgradnoise, wtgraddataperturb,
            # inpactdata, inpactnoise, inpactdataperturb, inpactgraddata, inpactgradnoise, inpactgraddataperturb,
            # preactdata, preactnoise, preactdataperturb, preactgraddata, preactgradnoise, preactgraddataperturb,
            # actdata, actnoise, actdataperturb, actgraddata, actgradnoise, actgraddataperturb]
# Conv3x3BnReLU has: 
#  Conv3x3BnRelu(
#   (conv3x3): ConvBnRelu(
#     (conv_bn_relu): Sequential(
    #   !!! INPUT
#       (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #   !!! CONV_OUT
#       (2): ReLU()
    #   !!! ReLU
#     )
#   )
# )

TYPE_STATS = ['DATA', 'NOISE', 'PERTURB']
STAT = ['INPUT', 'CONV_OUT', 'ReLU']
master_state_dict = {}
# rand_idx_list  = [random.randint(0, len(api)-1) for _ in range(NUM_NETS)]
i_reg  = 0

while i_reg  < NUM_NETS:
    rand_num = random.randint(0, len(api)-1)
    state_dict = {}
    config = api.get_net_config(rand_num, 'ImageNet16-120')
    results = api.get_more_info(rand_num, 'ImageNet16-120', hp=200)
    
    state_dict["net_info"] = results 
    net = get_cell_based_tiny_net(config)
    print("Progress: ", i_reg)
    print(state_dict["net_info"])
    data_sample = next(iter(train_loader))[0].unsqueeze(0)

    # First, save all weights of the network
    #'Conv2d', 'BatchNorm2d', 'ReLU'
    # Note that for CIFAR100 networks, this isn't entirely correct
    # There are some downsample type layers that we are skipping I suppose
    # Also, order has changed here :/ 
# NASBENCH101
# Conv3x3BnReLU had: 
#  Conv3x3BnRelu(
#   (conv3x3): ConvBnRelu(
#     (conv_bn_relu): Sequential(
    #   !!! INPUT
#       (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #   !!! CONV_OUT
#       (2): ReLU()
    #   !!! ReLU
#     )
#   )
# )
# NASBENCH201 
# (0): ReLU()
    # !!! INPUT
# (1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # !!! CONV_OUT
# (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#

    conv_module_list = [x for x in net.modules() if x.__class__.__name__.__contains__('ReLUConvBN')]
    layer_module_list = []
    for x in conv_module_list:
        layer_module_list.append([y for y in x.modules() if y.__class__.__name__ == "Conv2d" or y.__class__.__name__ == "BatchNorm2d" or y.__class__.__name__  =="ReLU"])
    flat_layer_module_list = [item for sublist in layer_module_list for item in sublist]
    # Now, remove the first item (Which is always a relu)
    assert(flat_layer_module_list[0].__class__.__name__=='ReLU')
    # Removes first relu
    flat_layer_module_list  = flat_layer_module_list[1:]
    # Removes last layer :|
    flat_layer_module_list = flat_layer_module_list[:-2]
    # Should we also append a relu layer in front of it?
    # Rather, fetch the relu after the last layer.
    # But, there is an average pool after the last batch norm 
    # [TODO] So, for now perhaps we can remove the last layer from consideration (Which is VERY  BAD)

# Forward pass hook : Take ConvBnReLU modules, add forward hooks at  conv2d.input, batchnorm2d.output, relu.output
# Backward pass hook: Take ConvBnReLU modules, add backward hooks at conv2d.input, batchnorm2d.output, relu.output
    num_tot_layers = len(flat_layer_module_list)
    num_conv_layers = num_tot_layers//3
    # STAGE: Data
    # Create hooks for all Conv layers.
    hookF = [Hook(layer) for layer in flat_layer_module_list]
    hookB = [Hook(layer, backward=True) for layer in flat_layer_module_list]
    # Get Forward Pass Hook
    # fix data min max scaling
    out, _ = net(data_sample.float())
    print(data_sample.shape)
    # Get Backward Pass Hook
    out.backward(torch.ones_like(out))
    for i in range(num_conv_layers):
        state_dict['layer' + str(i)] = {}

    for i in range(num_tot_layers):
        if i%3 == 0:
            state_dict['layer'+str(i//3)]['wt'] = flat_layer_module_list[i].weight.clone().detach()
            state_dict['layer'+str(i//3)]['wtgraddata'] = flat_layer_module_list[i].weight.grad.clone().detach()
            state_dict['layer'+str(i//3)]['inpactdata'] = hookF[i].input[0].clone().detach()
            state_dict['layer'+str(i//3)]['inpactgraddata'] = hookB[i].input[0].clone().detach()
        if i%3 == 1:
            state_dict['layer'+str(i//3)]['preactdata'] = hookF[i].output.clone().detach()
            state_dict['layer'+str(i//3)]['preactgraddata'] = hookB[i].output[0].clone().detach()
        if i%3 == 2:
            state_dict['layer'+str(i//3)]['actdata'] = hookF[i].output.clone().detach()
            state_dict['layer'+str(i//3)]['actgraddata'] = hookB[i].output[0].clone().detach()
    net.zero_grad()
    # STAGE: Noise
    # Create hooks for all Conv layers.
    hookF = [Hook(layer) for layer in flat_layer_module_list]
    hookB = [Hook(layer, backward=True) for layer in flat_layer_module_list]
    # Get Forward Pass Hook
    out, _ = net(torch.randn(batch_size, 3, 16, 16))
    # Get Backward Pass Hook
    out.backward(torch.ones_like(out))

    for i in range(num_tot_layers):
        if i%3 == 0:
            state_dict['layer'+str(i//3)]['wtgradnoise'] = flat_layer_module_list[i].weight.grad.clone().detach()
            state_dict['layer'+str(i//3)]['inpactnoise'] = hookF[i].input[0].clone().detach()
            state_dict['layer'+str(i//3)]['inpactgradnoise'] = hookB[i].input[0].clone().detach()
        if i%3 == 1:
            state_dict['layer'+str(i//3)]['preactnoise'] = hookF[i].output.clone().detach()
            state_dict['layer'+str(i//3)]['preactgradnoise'] = hookB[i].output[0].clone().detach()
        if i%3 == 2:
            state_dict['layer'+str(i//3)]['actnoise'] = hookF[i].output.clone().detach()
            state_dict['layer'+str(i//3)]['actgradnoise'] = hookB[i].output[0].clone().detach()
    net.zero_grad()
    # STAGE: Perturb  
    # Create hooks for all Conv layers.
    hookF = [Hook(layer) for layer in flat_layer_module_list]
    hookB = [Hook(layer, backward=True) for layer in flat_layer_module_list]
    # Get Forward Pass Hook
    out, _ = net(data_sample.float() + 0.01**0.5*torch.randn(batch_size, 3, 16, 16))
    # Get Backward Pass Hook
    out.backward(torch.ones_like(out))

    for i in range(num_tot_layers):
        if i%3 == 0:
            state_dict['layer'+str(i//3)]['wtgradperturb'] = flat_layer_module_list[i].weight.grad.clone().detach() - state_dict['layer'+str(i//3)]['wtgraddata']
            state_dict['layer'+str(i//3)]['inpactperturb'] = hookF[i].input[0].clone().detach() - state_dict['layer'+str(i//3)]['inpactdata']
            state_dict['layer'+str(i//3)]['inpactgradperturb'] = hookB[i].input[0].clone().detach() - state_dict['layer'+str(i//3)]['inpactgraddata']
        if i%3 == 1:
            state_dict['layer'+str(i//3)]['preactperturb'] = hookF[i].output.clone().detach() - state_dict['layer'+str(i//3)]['preactdata']
            state_dict['layer'+str(i//3)]['preactgradperturb'] = hookB[i].output[0].clone().detach() - state_dict['layer'+str(i//3)]['preactgraddata']
        if i%3 == 2:
            state_dict['layer'+str(i//3)]['actperturb'] = hookF[i].output.clone().detach() - state_dict['layer'+str(i//3)]['actdata']
            state_dict['layer'+str(i//3)]['actgradperturb'] = hookB[i].output[0].clone().detach() - state_dict['layer'+str(i//3)]['actgraddata']

    master_state_dict[i_reg] = state_dict
    i_reg += 1

# [master_state_dict[0]['layer0'][x].shape for x in master_state_dict[0]['layer0'].keys()]


import sys

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

print(get_size(master_state_dict))

eval_dataset_path = 'net_stats_dataset_' + str(NUM_NETS) + '_NN_22_stats_NASBench201_' + dataset  + '.pt'
test_dataset_path = 'net_stats_dataset_' + str(NUM_NETS) + '_NN_22_stats_NASBench201_' + dataset  + '_test.pt'
if test==True:
    torch.save(master_state_dict, test_dataset_path)
else:
    torch.save(master_state_dict, eval_dataset_path)