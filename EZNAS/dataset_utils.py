import argparse
import numpy as np
import random 
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
# Secondary Imports
from xautodl.datasets.get_dataset_with_transform import get_datasets
from xautodl.datasets.DownsampledImageNet import ImageNet16
from nasbench import api
    
def get_imagenet_224_train_loader(batch_size=4):
    valdir = os.path.join(os.environ['ilsvrc2012_PATH'], 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)
    return val_loader

def get_imagenet_train_loader(batch_size=4):
    mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    std = [x / 255 for x in [63.22, 61.26, 65.09]]
    lists = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(16, padding=2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    ]
    train_transform = transforms.Compose(lists)
    xshape = (1, 3, 16, 16)
    # use path from 'IN16120_PATH as environment variable
    train_data = ImageNet16(os.environ['IN16120_PATH'], True, train_transform, 120)
    assert len(train_data) == 151700
    return train_data

def get_cifar100_train_loader(batch_size=4):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./cifardata', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    return trainloader



def get_train_loader(batch_size=4):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./cifardata', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)
    return trainloader


def get_networks(NUM_NETS, nasbench, ALLOWED_OPS, INPUT, OUTPUT):
    net_list = []
    while(len(net_list)<NUM_NETS):
        # print(len(net_list), end=", ")
        try:
            n = 7
            individual = [int(random.random()>0.7) for _ in range(25)]
            test_matrix =  np.asarray([[0, 1, 0, 0, 0, 0, 0],    # input layer
                            [0, 0, 1, 0, 0, 0, 0],    # 1x1 conv
                            [0, 0, 0, 1, 0, 0, 0],    # 3x3 conv
                            [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
                            [0, 0, 0, 0, 0, 1, 0],    # 5x5 conv (replaced by two 3x3's)
                            [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
                            [0, 0, 0, 0, 0, 0, 0]])
            triu = np.triu_indices(n, 2) # Find upper right indices of a triangular nxn matrix
            test_matrix[triu] = individual[:15] # Assign list values to upper right matrix
            indv = [str(x) for  x in individual]
            ops = [INPUT] + [ALLOWED_OPS[int(''.join(indv[15+i:17+i]), 2)] for i in range(0, 10, 2)] + [OUTPUT] 
            model_spec = api.ModelSpec(matrix=test_matrix, ops=ops)             
            val_acc = nasbench.query(model_spec)['validation_accuracy']
            net_list.append((test_matrix, [str(x) for  x in individual[15:]]))
        except:
            pass
    return net_list


def get_eznas_trainloader(batch_size, dataset):
    if dataset == "cifar100":
        train_loader = get_cifar100_train_loader(batch_size=batch_size)
    elif dataset == "cifar10":
        train_loader = get_train_loader(batch_size=batch_size)
    elif dataset == "ImageNet16-120":
        train_loader = get_imagenet_train_loader(batch_size=batch_size)
    elif dataset == "ImageNet":
        train_loader = get_imagenet_224_train_loader(batch_size=batch_size)
    return train_loader

def get_args():
    parser = argparse.ArgumentParser(description='NAS Without Training')
    parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
    parser.add_argument('--api_loc', default='nasbench_only108.tfrecord',
                        type=str, help='path to API')
    parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
    parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
    parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
    parser.add_argument('--nasspace', default='nasbench101', type=str, help='the nas search space to use')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
    parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
    parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
    parser.add_argument('--init', default='', type=str)
    parser.add_argument('--GPU', default='0', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--trainval', action='store_true')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
    parser.add_argument('--n_samples', default=-1, type=int)
    parser.add_argument('--n_runs', default=500, type=int)
    parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
    parser.add_argument('--num_labels', default=10, type=int, help='#classes (nasbench101)')


    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120', 'ImageNet'], type=str, help='Data-set to be used')
    parser.add_argument('--search_space', default='NASBench201', choices=['NASBench201', 'NATSBench', 'NDS'], type=str, help='Search space')
    parser.add_argument('--nds_space', default='', choices=['nds_amoeba', 'nds_darts', 'nds_enas', 'nds_pnas', 'nds_nasnet'], type=str, help='NDS space specification')
    args = parser.parse_args()
    return args