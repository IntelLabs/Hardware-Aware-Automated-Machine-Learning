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
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

args = get_args()
args.nasspace = args.nds_space
dataset = args.dataset
search_space = args.search_space

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

if nds_space!='':
    network_sampler = get_search_space(args)
else:
    NASBENCH_TFRECORD = 'NAS-Bench-201-v1_1-096897.pth'
    api = API(NASBENCH_TFRECORD, verbose=False)

master_state_dict = {}
i_reg  = 0
# exit(0)
acc_list = []
flops_list = []
params_list = []
# exit(0)
from tqdm import tqdm
for i in tqdm(range(NUM_NETS)):
    if search_space!='NASBench201':
        rand_num = random.randint(0, network_sampler.__len__()-1)
        results = {"test-accuracy": network_sampler.get_final_accuracy(rand_num, 's', 's')}
        acc_list.append(network_sampler.get_final_accuracy(rand_num, 's', 's'))
        params_list.append(network_sampler.data[rand_num]['params'])
        flops_list.append(network_sampler.data[rand_num]['flops'])
    else:
        rand_num = random.randint(0, len(api)-1)
        results = api.get_more_info(rand_num, dataset, hp=200)
        acc_list.append(results['test-accuracy'])
        params_list.append(api.get_cost_info(rand_num, dataset)['params'])
        flops_list.append(api.get_cost_info(rand_num, dataset)['flops'])

# Calculate FLOPS KDT and Pearson
tau, p_score = stats.kendalltau(np.asarray(flops_list), np.asarray(acc_list))
pearson, _ = stats.pearsonr(np.asarray(flops_list), np.asarray(acc_list))
plt.title('FLOPs Pearson: ' + str(pearson)[:6] + "; KendallTau: " + str(tau)[:6])
plt.scatter(acc_list, flops_list)
plt.xlabel('Accuracy')
plt.ylabel('Score')
plt.savefig('./flops_param_graphs2/' +"flops_"+ str(dataset) +"_"+ str(search_space) +"_"+ str(nds_space) + ".png")
plt.clf()
# Calculate Params KDT and Pearson
tau, p_score = stats.kendalltau(np.asarray(params_list), np.asarray(acc_list))
pearson, _ = stats.pearsonr(np.asarray(params_list), np.asarray(acc_list))
plt.title('Params Pearson: ' + str(pearson)[:6] + "; KendallTau: " + str(tau)[:6])
plt.scatter(acc_list, params_list)
plt.xlabel('Accuracy')
plt.ylabel('Score')
plt.savefig('./flops_param_graphs2/' +"params_"+ str(dataset) +"_"+ str(search_space) +"_"+ str(nds_space) + ".png")
plt.clf()