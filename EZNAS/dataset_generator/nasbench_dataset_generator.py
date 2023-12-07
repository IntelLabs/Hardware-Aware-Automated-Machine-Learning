from nas_101_api.model import Network
from nasbench import api
import numpy as np
import random 
from dataset_utils import *
import argparse
from torchsummary import summary


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
train_loader = get_train_loader(batch_size=batch_size)

test = True

NASBENCH_TFRECORD = 'nasbench_only108.tfrecord'
INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3, CONV1X1]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix
NUM_VERTICES = 7
MAX_EDGES = 9
args = get_args()
nasbench = api.NASBench(NASBENCH_TFRECORD)
print("NASBench TFRecord loaded.")

NUM_NETS = 500
net_list = []
net_list = get_networks(NUM_NETS, nasbench, ALLOWED_OPS, INPUT, OUTPUT)

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
TYPE_STATS = ['DATA', 'NOISE', 'PERTURB']
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
STAT = ['INPUT', 'CONV_OUT', 'ReLU']
# exit(0)
master_state_dict = {}
for idx, (net_mat, ops_list) in enumerate(net_list):
    state_dict = {}
    ops = [INPUT] + [ALLOWED_OPS[int(''.join(ops_list[i:i+2]), 2)] for i in range(0, 10, 2)] + [OUTPUT] 
    model_spec = api.ModelSpec(matrix=net_mat, ops=ops)        
    net = Network(model_spec, args)     
    state_dict["net_info"] = nasbench.query(model_spec)   
    net = Network(model_spec, args)
    print(state_dict["net_info"])
    # summary(net, (3, 32, 32))
    data_sample = next(iter(train_loader))
    # First, save all weights of the network
    #'Conv2d', 'BatchNorm2d', 'ReLU'
    conv_module_list = [x for x in net.modules() if x.__class__.__name__.__contains__('Conv1x1BnRelu') or x.__class__.__name__.__contains__('Conv3x3BnRelu')]
    layer_module_list = []
    for x in conv_module_list:
        layer_module_list.append([y for y in x.modules() if y.__class__.__name__ == "Conv2d" or y.__class__.__name__ == "BatchNorm2d" or y.__class__.__name__  =="ReLU"])
    flat_layer_module_list = [item for sublist in layer_module_list for item in sublist]
    # exit(0)

# Forward pass hook : Take ConvBnReLU modules, add forward hooks at  conv2d.input, batchnorm2d.output, relu.output
# Backward pass hook: Take ConvBnReLU modules, add backward hooks at conv2d.input, batchnorm2d.output, relu.output
    num_conv_layers = len(conv_module_list)
    num_tot_layers = len(flat_layer_module_list)
    # STAGE: Data
    # Create hooks for all Conv layers.
    hookF = [Hook(layer) for layer in flat_layer_module_list]
    hookB = [Hook(layer, backward=True) for layer in flat_layer_module_list]
    # Get Forward Pass Hook
    out, _ = net(data_sample[0])
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
    out, _ = net(torch.randn(batch_size, 3, 32, 32))
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
    out, _ = net(data_sample[0] + 0.01**0.5*torch.randn(batch_size, 3, 32, 32))
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

    master_state_dict[idx] = state_dict

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

eval_dataset_path = 'net_stats_dataset_' + str(NUM_NETS) + '_NN_22_stats_CIFAR_10.pt'
test_dataset_path = 'net_stats_dataset_' + str(NUM_NETS) + '_NN_22_stats_CIFAR_10_test.pt'
if test==True:
    torch.save(master_state_dict, test_dataset_path)
else:
    torch.save(master_state_dict, eval_dataset_path)