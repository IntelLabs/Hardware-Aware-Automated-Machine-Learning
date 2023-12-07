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
from inference_data_helpers import  *
from tqdm import tqdm
# dataset: cifar10, cifar100, ImageNet16-120, ImageNet
# search_space: NASBench201, NDS
# network_space: amoeba, darts, enas, nasnet, pnas
# Make a class which holds all data-loaders
class NetStatisticsFetcher():
    def __init__(self, batch_size=1):
        super(NetStatisticsFetcher, self).__init__()
        assert batch_size==1
        NASBENCH_TFRECORD = 'NAS-Bench-201-v1_1-096897.pth'
        # Data-loaders
        print("Loading data-loaders...")
        self.cifar100_train_loader = get_cifar100_train_loader(batch_size=batch_size)
        self.cifar10_train_loader = get_train_loader(batch_size=batch_size)
        self.imagenet_16_train_loader = get_imagenet_train_loader(batch_size=batch_size)
        self.imagenet_train_loader = get_imagenet_224_train_loader(batch_size=batch_size)
        # Network-instantiators 
        # Instantiate a network sampler for EACH dataset (CIFAR10 and CIFAR100) and network space on NDS
        # CIFAR10 NDS
        print("Loading NDS Sampler...")
        choices = ['nds_amoeba', 'nds_darts', 'nds_enas', 'nds_pnas', 'nds_nasnet']
        args = get_args()
        args.nasspace = choices[0]
        self.network_sampler_amoeba = get_search_space(args)
        args.nasspace = choices[1]
        self.network_sampler_darts = get_search_space(args)
        args.nasspace = choices[2]
        self.network_sampler_enas = get_search_space(args)
        args.nasspace = choices[3]
        self.network_sampler_pnas = get_search_space(args)
        args.nasspace = choices[4]
        self.network_sampler_nasnet = get_search_space(args)
        # ImageNet NDS
        choices = ['nds_amoeba_in', 'nds_darts_in', 'nds_enas_in', 'nds_pnas_in', 'nds_nasnet_in']
        args.nasspace = choices[0]
        self.network_sampler_amoeba_in = get_search_space(args)
        args.nasspace = choices[1]
        self.network_sampler_darts_in = get_search_space(args)
        args.nasspace = choices[2]
        self.network_sampler_enas_in = get_search_space(args)
        args.nasspace = choices[3]
        self.network_sampler_pnas_in = get_search_space(args)
        args.nasspace = choices[4]
        self.network_sampler_nasnet_in = get_search_space(args)
        # Single sampler for NASBench201 is sufficient
        print("Loading NASBench201 Sampler...")
        self.api = API(NASBENCH_TFRECORD, verbose=False)

    def get_inference_data(self, dataset, search_space, network_space):
        if dataset!='ImageNet16-120':
            train_loader = eval('self.' + dataset.lower() + '_train_loader')
        else:
            train_loader = self.imagenet_16_train_loader
        master_state_dict = {}
        i_reg = 0 
        while i_reg < 1:
            try:
                state_dict = {}
                if search_space=='NDS':
                    network_sampler = eval('self.network_sampler_' + network_space)
                    rand_num = random.randint(0, network_sampler.__len__()-1)
                    results = {"test-accuracy": network_sampler.get_final_accuracy(rand_num, 's', 's')}
                    state_dict["net_info"] = results 
                    net = network_sampler.get_network(rand_num)
                else:
                    api = self.api
                    rand_num = random.randint(0, len(api)-1)
                    results = api.get_more_info(rand_num, dataset, hp=200)
                    state_dict["net_info"] = results 
                    config = api.get_net_config(rand_num, dataset)
                    net = get_cell_based_tiny_net(config)
                    # Now, we have the net and the state_dict with accuracies defined.
                for m in net.modules():
                    initialize_module(m)
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
                out, _ = net(data_sample.float())
                
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
                i_reg = i_reg
        return master_state_dict


    def get_many_net_inference_data(self, dataset, search_space, network_space, num_nets):
        if dataset!='ImageNet16-120':
            train_loader = eval('self.' + dataset.lower() + '_train_loader')
        else:
            train_loader = self.imagenet_16_train_loader
        master_state_dict = {}
        i_reg = 0 
        pbar = tqdm(total = num_nets)
        print("Generating Dataset: ", dataset, "\nSearch Space: ", search_space, "\nNetwork Space: ", network_space, "\nNumber of Samples: ", num_nets)
        while i_reg < num_nets:
            try:
                state_dict = {}
                if search_space=='NDS':
                    network_sampler = eval('self.network_sampler_' + network_space)
                    rand_num = random.randint(0, network_sampler.__len__()-1)
                    results = {"test-accuracy": network_sampler.get_final_accuracy(rand_num, 's', 's')}
                    state_dict["net_info"] = results 
                    net = network_sampler.get_network(rand_num)
                else:
                    api = self.api
                    rand_num = random.randint(0, len(api)-1)
                    results = api.get_more_info(rand_num, dataset, hp=200)
                    state_dict["net_info"] = results 
                    config = api.get_net_config(rand_num, dataset)
                    net = get_cell_based_tiny_net(config)
                    # Now, we have the net and the state_dict with accuracies defined.
                for m in net.modules():
                    initialize_module(m)
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
                out, _ = net(data_sample.float())
                
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
                pbar.update(1)
            except Exception as e:
                i_reg = i_reg
        pbar.close()
        return master_state_dict
