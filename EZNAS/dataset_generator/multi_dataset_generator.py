import torch
import os
import random
from tqdm import tqdm

SUBSAMPLE_NETS = 20
NUM_SAMPLING_EVAL = 4

# Create 4 files with _a, _b, _c, _d
# Each of them to have 30 networks.
# Do nothing for imagenet
fnames = ['net_stat_NN_22_stats_cifar100_NASBench201__1000.pt', 'net_stat_NN_22_stats_cifar10_NASBench201__1000.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_amoeba_200.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_darts_200.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_enas_200.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_nasnet_200.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_pnas_200.pt', 'net_stat_NN_22_stats_ImageNet16-120_NASBench201__1000.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_amoeba_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_darts_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_enas_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_nasnet_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_pnas_in_20.pt']
fnames = ['/data2/yakhauri/datasets/' + x for x in fnames if x.__contains__('ImageNet_')==False]

for master_file_name in tqdm(fnames):
    master_dict = torch.load(master_file_name)
    num_nets_available = int(master_file_name.split("_")[-1][:-3])
    start_idxs = random.sample(range(num_nets_available//SUBSAMPLE_NETS), NUM_SAMPLING_EVAL)
    # If there are 200 networks and 20 subsampling nets 
    # We will sample 4 numbers from 0 to 9
    # Multiply each of these with SUBSAMPLE_NETS to get the starting idx
    start_idxs = [x*SUBSAMPLE_NETS for x in start_idxs]
    # [0, 20, 60, 180]
    for i in tqdm(range(NUM_SAMPLING_EVAL)):
        sub_dict = {}
        for j in range(SUBSAMPLE_NETS):
            sub_dict[j] = master_dict[start_idxs[i]+j]
        generated_file_name = master_file_name[:-3]
        generated_file_name = generated_file_name + "_subs_" + str(SUBSAMPLE_NETS) + "_id_" + str(i) +".pt"
        torch.save(sub_dict, generated_file_name)


