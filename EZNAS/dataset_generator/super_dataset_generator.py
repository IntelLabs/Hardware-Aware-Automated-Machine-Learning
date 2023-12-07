# from nas_101_api.model_fetchall import Network
# from nasbench import api
from nas_201_api import NASBench201API as API
import numpy as np
import random 
from dataset_utils import *
import argparse
from torchsummary import summary
from models import get_cell_based_tiny_net 
from nasspace import *
import argparse
import torch.nn as nn


args = get_args()
args.nasspace = args.nds_space

# dataset_path = 'net_stat_NN_22_stats_' + dataset + "_" + search_space + "_" + nds_space + ".pt"
dataset = args.dataset
search_space = args.search_space

# if dataset=='ImageNet':
#     NUM_NETS = 115
# else:
#     NUM_NETS = 500

# For NASBench201, we will generate 1000 networks on each data-set
# For NDS CIFAR10, we will generate 100 networks 
# For NDS ImageNet, we will generate 100 networks

if dataset == 'ImageNet':
    NUM_NETS = 20
elif dataset == 'cifar10' and search_space=='NDS':
    NUM_NETS = 200
else:
    NUM_NETS = 1000


nds_space = '' if search_space=='NASBench201' else args.nds_space

if dataset=='ImageNet':
    nds_space += '_in'
if search_space=='NDS':
    assert(dataset !='cifar100' or dataset != 'ImageNet16-120')


dataset_path = '/data3/yakhauri/datasets/net_stat_NN_22_stats_' + dataset + "_" + search_space + "_" + nds_space + "_" + str(NUM_NETS) + ".pt"

batch_size = 1
if dataset=='cifar100':
    train_loader = get_cifar100_train_loader(batch_size=batch_size)
elif dataset=='cifar10':
    train_loader = get_train_loader(batch_size=batch_size)
elif dataset=='ImageNet16-120':
    train_loader = get_imagenet_train_loader(batch_size=batch_size)
elif dataset=='ImageNet':
    train_loader = get_imagenet_224_train_loader(batch_size=batch_size)
    


if nds_space!='':
    network_sampler = get_search_space(args)
else:
    NASBENCH_TFRECORD = 'NAS-Bench-201-v1_1-096897.pth'
    api = API(NASBENCH_TFRECORD, verbose=False)
    
# A simple hook that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
            # self.hook = module.register_full_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    # def close(self):
    #     self.hook.remove()

def initialize_module(m):
  if isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if m.bias is not None:
      nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.constant_(m.weight, 1)
    if m.bias is not None:
      nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.Linear):
    nn.init.normal_(m.weight, 0, 0.01)
    nn.init.constant_(m.bias, 0)

TYPE_STATS = ['DATA', 'NOISE', 'PERTURB']
STAT = ['INPUT', 'CONV_OUT', 'ReLU']
master_state_dict = {}
# rand_idx_list  = [random.randint(0, len(api)-1) for _ in range(NUM_NETS)]
i_reg  = 0
while i_reg  < NUM_NETS:
    try:
        state_dict = {}
        if search_space!='NASBench201':
            rand_num = random.randint(0, network_sampler.__len__()-1)
            results = {"test-accuracy": network_sampler.get_final_accuracy(rand_num, 's', 's')}
            state_dict["net_info"] = results 
            net = network_sampler.get_network(rand_num)
        
        else:
            rand_num = random.randint(0, len(api)-1)
            results = api.get_more_info(rand_num, dataset, hp=200)
            state_dict["net_info"] = results 
            config = api.get_net_config(rand_num, dataset)
            net = get_cell_based_tiny_net(config)
        #  Initialization strategy of network
        for m in net.modules():
            initialize_module(m)
        
        print("Progress: ", i_reg)
        print(state_dict["net_info"])
        data_sample = next(iter(train_loader))[0]
        if len(data_sample.shape)==3:
            assert(data_sample.shape[0]==3)
            data_sample = data_sample.unsqueeze(0)

        flat_layer_module_list = [y for y in net.modules() if y.__class__.__name__ == "Conv2d" or y.__class__.__name__ == "BatchNorm2d" or y.__class__.__name__  =="ReLU"]

        flat_layer_module_list_cleaned = []
        idx = 0
        while idx < len(flat_layer_module_list):
            # Uncomment to remove layers of type Conv2D (7,1) --> Conv2D (1,7) (This is actually incorrect, fix this later if using)
            # if flat_layer_module_list[idx].__class__.__name__=='Conv2d' and flat_layer_module_list[idx+1].__class__.__name__=='Conv2d' and flat_layer_module_list.kernel_size[0]==flat_layer_module_list.kernel_size[1]:
            if flat_layer_module_list[idx].__class__.__name__=='Conv2d' and flat_layer_module_list[idx+1].__class__.__name__=='Conv2d':
                flat_layer_module_list_cleaned.append(flat_layer_module_list[idx])
                idx = idx+2
            # Skip the entire layer if it does not have batchnorm2d as the next layer (avoid avgpool->conv2d type layers etc)
            elif flat_layer_module_list[idx].__class__.__name__=='Conv2d' and flat_layer_module_list[idx+1].__class__.__name__!='BatchNorm2d':
                idx = idx+1
            else:
                flat_layer_module_list_cleaned.append(flat_layer_module_list[idx])
                idx = idx + 1
        flat_layer_module_list = flat_layer_module_list_cleaned
        flat_layer_module_list = flat_layer_module_list[2:]
        if search_space=='NASBench201':
            flat_layer_module_list = flat_layer_module_list[:-1]
            if flat_layer_module_list[-1].__class__.__name__=='BatchNorm2d' and flat_layer_module_list[-2].__class__.__name__=='BatchNorm2d':
                flat_layer_module_list = flat_layer_module_list[:-1]
        # Here, we have removed first layer (due to no ReLU at start) and removed the last ReLU if it exists
        num_tot_layers = len(flat_layer_module_list)
        num_conv_layers = num_tot_layers//3
        for i in range(num_conv_layers):
            state_dict['layer' + str(i)] = {}
        assert(num_tot_layers%3 == 0)
        # STAGE: Data
        # Create hooks for all Conv layers.
        hookF = [Hook(layer) for layer in flat_layer_module_list]
        hookB = [Hook(layer, backward=True) for layer in flat_layer_module_list]
        # Get Forward Pass Hook
        # fix data min max scaling
        # Make sure input requires gradient
        # data_sample.requires_grad=True
        out, _ = net(data_sample.float())
        print(data_sample.shape)
        # Get Backward Pass Hook
        out.backward(torch.ones_like(out))
        for i in range(num_tot_layers):
            if i%3 == 0:
                state_dict['layer'+str(i//3)]['inpactdata'] = hookF[i].input[0].clone().detach()
                state_dict['layer'+str(i//3)]['inpactgraddata'] = hookB[i].input[0].clone().detach()
            if i%3 == 1:
                state_dict['layer'+str(i//3)]['wt'] = flat_layer_module_list[i].weight.clone().detach()
                state_dict['layer'+str(i//3)]['wtgraddata'] = flat_layer_module_list[i].weight.grad.clone().detach()
                state_dict['layer'+str(i//3)]['preactdata'] = hookF[i].input[0].clone().detach()
                state_dict['layer'+str(i//3)]['preactgraddata'] = hookB[i].input[0].clone().detach()
            if i%3 == 2:
                state_dict['layer'+str(i//3)]['actdata'] = hookF[i].output.clone().detach()
                state_dict['layer'+str(i//3)]['actgraddata'] = hookB[i].output[0].clone().detach()
        net.zero_grad()
        print("Data stage done")
        # STAGE: Noise
        # Create hooks for all Conv layers.
        hookF = [Hook(layer) for layer in flat_layer_module_list]
        hookB = [Hook(layer, backward=True) for layer in flat_layer_module_list]
        # Get Forward Pass Hook
        out, _ = net(torch.randn(data_sample.shape))
        # Get Backward Pass Hook
        out.backward(torch.ones_like(out))

        for i in range(num_tot_layers):
            if i%3 == 0:
                state_dict['layer'+str(i//3)]['inpactnoise'] = hookF[i].input[0].clone().detach()
                state_dict['layer'+str(i//3)]['inpactgradnoise'] = hookB[i].input[0].clone().detach()
            if i%3 == 1:
                state_dict['layer'+str(i//3)]['wtgradnoise'] = flat_layer_module_list[i].weight.grad.clone().detach()
                state_dict['layer'+str(i//3)]['preactnoise'] = hookF[i].input[0].clone().detach()
                state_dict['layer'+str(i//3)]['preactgradnoise'] = hookB[i].input[0].clone().detach()
            if i%3 == 2:
                state_dict['layer'+str(i//3)]['actnoise'] = hookF[i].output.clone().detach()
                state_dict['layer'+str(i//3)]['actgradnoise'] = hookB[i].output[0].clone().detach()
        net.zero_grad()
        print("Noise stage done")
        # STAGE: Perturb  
        # Create hooks for all Conv layers.
        hookF = [Hook(layer) for layer in flat_layer_module_list]
        hookB = [Hook(layer, backward=True) for layer in flat_layer_module_list]
        # Get Forward Pass Hook
        out, _ = net(data_sample.float() + 0.01**0.5*torch.randn(data_sample.shape))
        # Get Backward Pass Hook
        out.backward(torch.ones_like(out))

        for i in range(num_tot_layers):
            if i%3 == 0:
                state_dict['layer'+str(i//3)]['inpactperturb'] = hookF[i].input[0].clone().detach() - state_dict['layer'+str(i//3)]['inpactdata']
                state_dict['layer'+str(i//3)]['inpactgradperturb'] = hookB[i].input[0].clone().detach() - state_dict['layer'+str(i//3)]['inpactgraddata']
            if i%3 == 1:
                state_dict['layer'+str(i//3)]['wtgradperturb'] = flat_layer_module_list[i].weight.grad.clone().detach() - state_dict['layer'+str(i//3)]['wtgraddata']
                state_dict['layer'+str(i//3)]['preactperturb'] = hookF[i].input[0].clone().detach() - state_dict['layer'+str(i//3)]['preactdata']
                state_dict['layer'+str(i//3)]['preactgradperturb'] = hookB[i].input[0].clone().detach() - state_dict['layer'+str(i//3)]['preactgraddata']
            if i%3 == 2:
                state_dict['layer'+str(i//3)]['actperturb'] = hookF[i].output.clone().detach() - state_dict['layer'+str(i//3)]['actdata']
                state_dict['layer'+str(i//3)]['actgradperturb'] = hookB[i].output[0].clone().detach() - state_dict['layer'+str(i//3)]['actgraddata']
        print("Perturb stage done")


        master_state_dict[i_reg] = state_dict
        i_reg += 1
    except Exception as e:
        print("Exception encountered: ", e)


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


torch.save(master_state_dict, dataset_path)



# search_space = 'NASBench201'
        # (0): ReLUConvBN(
        #   (op): Sequential(
        #     (0): ReLU()
        #     (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   )
        # )
# search_space = 'NDS'
        #   (op): Sequential(
        #     (0): ReLU()
        #     (1): Conv2d(128, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=128, bias=False)
        #     (2): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (4): ReLU()
        #     (5): Conv2d(128, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=128, bias=False)
        #     (6): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   )
        ############################ PROBLEM BELOW ################################################        
        #   (0): ReLU()
        #   (1): Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False)
        #   (2): Conv2d(128, 128, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)
        #   (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ############################ PROBLEM ABOVE ################################################  

        
# dataset = 'cifar10'
# dataset = 'cifar100'
# dataset = 'ImageNet16-120'
# dataset = 'ImageNet'


# nds_space = '' if search_space=='NASBench201' else 'nds_amoeba' # This has 1,7 --> 7,1
# nds_space = '' if search_space=='NASBench201' else 'nds_darts' # This has 1,7 --> 7,1
# nds_space = '' if search_space=='NASBench201' else 'nds_enas' # Doesnt seem to have 1, 7 --> 7,1
# nds_space = '' if search_space=='NASBench201' else 'nds_pnas' # Doesnt seem to have 1,7 --> 7,1 
# nds_space = '' if search_space=='NASBench201' else 'nds_nasnet' # This has 1,7 --> 7,1  


# if nds_space == '', structure is : ReLU --> Conv2d --> BN 
# if nds_space != '', structure sometimes is : ReLU --> (Conv2d --> Conv2d) --> BN

# NDS-Amoeba CIFAR10    Amoeba      nds_amoeba
    # Many layers with: ReLU-Conv2d-Conv2d-BN
    # Only two with ReLU-Conv2d-BN
# NDS-DARTS CIFAR10     DARTS       nds_darts
    # Many layers with: ReLU-Conv2d-Conv2d-BN
    # Only two with ReLU-Conv2d-BN
# NDS-ENAS CIFAR10      ENAS        nds_enas
    # Many layers with: ReLU-Conv2d-Conv2d-BN
    # Only two with ReLU-Conv2d-BN
# NDS-PNAS CIFAR10      PNAS        nds_pnas
    # Many layers with: ReLU-Conv2d-Conv2d-BN
    # Only two with ReLU-Conv2d-BN
# NDS-NASNet CIFAR10    NASNet      nds_nasnet
    # Many layers with: ReLU-Conv2d-Conv2d-BN
    # Only two with ReLU-Conv2d-BN


    
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
##### DEPRECATED NASBench101 #####
# static addr: [wt, wtgraddata, wtgradnoise, wtgraddataperturb,
            # inpactdata, inpactnoise, inpactdataperturb, inpactgraddata, inpactgradnoise, inpactgraddataperturb,
            # preactdata, preactnoise, preactdataperturb, preactgraddata, preactgradnoise, preactgraddataperturb,
            # actdata, actnoise, actdataperturb, actgraddata, actgradnoise, actgraddataperturb]
#### NEW ####
#       ReLU    -->     Conv    -->     BN
#   ^                ^                      ^
#  relu.input       conv2d.input          bn.output
# inpactdata        preactdata            actdata
# static addr: [wt, wtgraddata, wtgradnoise, wtgraddataperturb,
            # inpactdata, inpactnoise, inpactdataperturb, inpactgraddata, inpactgradnoise, inpactgraddataperturb,
            # preactdata, preactnoise, preactdataperturb, preactgraddata, preactgradnoise, preactgraddataperturb,
            # actdata, actnoise, actdataperturb, actgraddata, actgradnoise, actgraddataperturb]

#### NEW ####
#       ReLU    -->     Conv    -->     BN
#   ^                ^                      ^
#  relu.input       conv2d.input          bn.output
# inpactdata        preactdata            actdata
# static addr: [wt, wtgraddata, wtgradnoise, wtgraddataperturb,
            # inpactdata, inpactnoise, inpactdataperturb, inpactgraddata, inpactgradnoise, inpactgraddataperturb,
            # preactdata, preactnoise, preactdataperturb, preactgraddata, preactgradnoise, preactgraddataperturb,
            # actdata, actnoise, actdataperturb, actgraddata, actgradnoise, actgraddataperturb]
# new
# Static addr: [inpactdata, inpactgraddata, wt, wtgraddata, preactdata, preactgraddata, actdata, actgraddata, 
                # inpactnoise, inpactgradnoise, wtgradnoise, preactnoise, preactgradnoise, actnoise, actgradnoise, 
                # inpactperturb, inpactgradperturb, wtgradperturb, preactperturb, preactgradperturb, actperturb, actgradperturb]


                    # Scan every Conv2d, if next element if a Conv2d as well, remove it. 
    # [TODO] or [NOTE] --> There are layers with ReLU --> Conv2D --> Conv2D --> BN
    #                       for such layers, we remove the second Conv2D. 
    #                       not ideal, but doing for early program discovery
    #  there are also some which have Conv2D (7,1) --> Conv2D (1,7) 
    #  We can also consider removing these type of layers entirely for shape matches
    # kz = [x for x in net.modules() if x.__class__.__name__=='Conv2d'][0].kernel_size
    # [x for x in net.modules() if x.__class__.__name__=='Conv2d' and x.kernel_size[0] != x.kernel_size[1]]
    # assert(kz[0] == kz[1])


    
    # Forward pass hook : Take ConvBnReLU modules, add forward hooks at  conv2d.input, batchnorm2d.output, relu.output
    # Backward pass hook: Take ConvBnReLU modules, add backward hooks at conv2d.input, batchnorm2d.output, relu.output