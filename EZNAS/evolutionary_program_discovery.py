import random
import operator
# from zeroshotautoml.logs.Jun_21_2021_15h_47m_33s.evolutionary_program_discovery import EVOLUTION_DATASET
from deap import creator, base, tools, algorithms, gp
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import time
from absl import app
import multiprocessing
from nasbench import api
from nas_101_api.model import Network
import argparse
# from op_mutations import *
from gp_func_defs import *
import scipy.stats as stats
from evolutionary_utils import load_individuals, selTournamentSansDuplicates, selTournamentSansPerfDuplicates, reduce_tree_repr
import yaml
### Graphviz Section ###
import pygraphviz as pgv
import itertools   
import math
from datetime import datetime, date
import shutil, os
import logging
import matplotlib.pyplot as plt
import matplotlib
from itertools import repeat
from tqdm import tqdm
import pickle
import toolz
from pprint import pprint
matplotlib.use('Agg')
now = datetime.now()
today = date.today()

# Metric to optimize over
# METRIC = 'pearson'
METRIC = 'kendalltau'

#########################################
# Evolve on x data-sets and search spaces
#   During evolution, evaluate on any single dict randomly
#   later, we can try evaluating on all dicts and taking least correlation
# Test on y data-sets and search spaces
#   run the graphing of best algorithms in a loop for all sets
#########################################


#  33G net_stat_NN_22_stats_cifar100_NASBench201__1000.pt
#  33G net_stat_NN_22_stats_cifar10_NASBench201__1000.pt
#  42G net_stat_NN_22_stats_cifar10_NDS_nds_amoeba_200.pt
#  34G net_stat_NN_22_stats_cifar10_NDS_nds_darts_200.pt
#  30G net_stat_NN_22_stats_cifar10_NDS_nds_enas_200.pt
#  32G net_stat_NN_22_stats_cifar10_NDS_nds_nasnet_200.pt
#  46G net_stat_NN_22_stats_cifar10_NDS_nds_pnas_200.pt
#  14G net_stat_NN_22_stats_ImageNet16-120_NASBench201__1000.pt
# 153G net_stat_NN_22_stats_ImageNet_NDS_nds_amoeba_in_20.pt
# 136G net_stat_NN_22_stats_ImageNet_NDS_nds_darts_in_20.pt
# 184G net_stat_NN_22_stats_ImageNet_NDS_nds_enas_in_20.pt
# 166G net_stat_NN_22_stats_ImageNet_NDS_nds_nasnet_in_20.pt
# 190G net_stat_NN_22_stats_ImageNet_NDS_nds_pnas_in_20.pt

# At the full scale evolution and testing stage (graphing)
# Load data-set once, evaluate all individuals on the same dataset
# Thus, below two dictionaries to be available and set to the dataset in question
# # Assign current_evol_full_dict to each dict in EVOLUTION_DATASET, and use that in graphing pearson
# global current_evol_full_dict
# # Assign current_test_full_dict to each dict in TEST_DATASET, and use that in graphing pearson
# global current_test_full_dict
# global current_dkey
# Assign current_evol_full_dict to each dict in EVOLUTION_DATASET, and use that in graphing pearson
# current_evol_full_dict = {}
# # Assign current_test_full_dict to each dict in TEST_DATASET, and use that in graphing pearson
# current_test_full_dict = {}
# current_dkey = ''
# Program validity dataset
PROG_VAL_DSET = ['net_stat_NN_22_stats_cifar10_NASBench201__1000.pt']
# Evolution dataset
# EVOLUTION_DATASET = ['cifar10_NDS_nds_nasnet', 'cifar10_NASBench201', 'ImageNet16-120_NASBench201']
# EVOLUTION_DATASET = ['net_stat_NN_22_stats_cifar10_NASBench201__1000.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_pnas_200.pt']
EVOLUTION_DATASET = ['net_stat_NN_22_stats_cifar10_NASBench201__1000.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_amoeba_200.pt','net_stat_NN_22_stats_cifar10_NDS_nds_darts_200.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_nasnet_200.pt']
# Test dataset
# TEST_DATASET = ['cifar10_NDS_nds_nasnet', 'cifar10_NASBench201', 'ImageNet16-120_NASBench201']
# TEST_DATASET = ['net_stat_NN_22_stats_cifar100_NASBench201__1000.pt']
TEST_DATASET = ['net_stat_NN_22_stats_cifar100_NASBench201__1000.pt',
            'net_stat_NN_22_stats_cifar10_NDS_nds_enas_200.pt', 
             'net_stat_NN_22_stats_ImageNet16-120_NASBench201__1000.pt',  'net_stat_NN_22_stats_cifar10_NDS_nds_pnas_200.pt',
            'net_stat_NN_22_stats_ImageNet_NDS_nds_amoeba_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_darts_in_20.pt', 
            'net_stat_NN_22_stats_ImageNet_NDS_nds_enas_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_nasnet_in_20.pt', 
            'net_stat_NN_22_stats_ImageNet_NDS_nds_pnas_in_20.pt']
# 
# TEST_DATASET = ['net_stat_NN_22_stats_cifar100_NASBench201__1000.pt', 'net_stat_NN_22_stats_cifar10_NASBench201__1000.pt', 
#             'net_stat_NN_22_stats_cifar10_NDS_nds_amoeba_200.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_darts_200.pt', 
#             'net_stat_NN_22_stats_cifar10_NDS_nds_enas_200.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_nasnet_200.pt', 
#             'net_stat_NN_22_stats_cifar10_NDS_nds_pnas_200.pt', 'net_stat_NN_22_stats_ImageNet16-120_NASBench201__1000.pt', 
#             'net_stat_NN_22_stats_ImageNet_NDS_nds_amoeba_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_darts_in_20.pt', 
#             'net_stat_NN_22_stats_ImageNet_NDS_nds_enas_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_nasnet_in_20.pt', 
#             'net_stat_NN_22_stats_ImageNet_NDS_nds_pnas_in_20.pt']
# TEST_DATASET = ['net_stat_NN_22_stats_ImageNet_NDS_nds_darts_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_enas_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_nasnet_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_pnas_in_20.pt']


suffix_logfile = METRIC + "_evol_" + "_".join(EVOLUTION_DATASET) + "_test_" + "_".join(TEST_DATASET)
current_time = now.strftime("%Hh_%Mm_%Ss")
if os.path.exists("logs")==False:
    os.mkdir("logs")

date = today.strftime("%b_%d_%Y")
# logfolder = "logs/" + date + "_" + current_time + "_" + METRIC + "_evol_" + EVOLUTION_DATASET + "_test_" + TEST_DATASET
logfolder = "logs/" + date + "_" + current_time
os.mkdir(logfolder)
fz = open(logfolder + "run_information.txt", 'w')
fz.write(suffix_logfile)
fz.close()
if os.path.exists(logfolder + '/trees_explored/') == False:
    os.mkdir(logfolder + "/trees_explored/")
if os.path.exists(logfolder + '/graphs/') == False:
    os.mkdir(logfolder + "/graphs/")
if os.path.exists(logfolder + '/models/') == False:
    os.mkdir(logfolder + "/models/")

files_to_copy = [x for x in os.listdir() if x.__contains__('.py') or x.__contains__('.yaml')] 
for f in files_to_copy:
    shutil.copy(f, logfolder)

#### Basic Configuration ####
with open("evol_config.yaml", "r") as f:
    config = yaml.load(f)
# Integer based math op identification
NUM_MATH_OPS = config['NUM_MATH_OPS']
INT_MIN, INT_MAX = 0, NUM_MATH_OPS
# Number of network statistics available
STATIC_ADDRS = config['STATIC_ADDRS']
NUM_ADDRS = STATIC_ADDRS + config['NUM_DYNAMIC_ADDR_SPACES']*config['LEN_IND_ADDR']
# Number of generations of evolutionary algorithm to run
NGEN=config['NGEN']
### Population --> lambda_ (mutation or mate) --> evaluate lambda_ --> MU selection (By randomly sampling TOURSIZE individuals and choosing by some criteria)
###      ^ ______________________________________________________________|
# Population size
POPSIZE = config['POPSIZE']
# Tournament size
TOURSIZE = config['TOURSIZE']
# MU The number of individuals to select for the next generation.
MU = config['MU']
# Number of children to produce at each generation
lambda_ = config['lambda_']
# Crossover probability
CXPR = config['CXPR']
# Mutation probability
MUTPR = config['MUTPR']
# Number of processes to launch
nproc = config['nproc']
# Evaluation data-set details
NUM_NETS = config['NUM_NETS']
# Test any individual on only 10 networks. 
SUBSAMPLE_NETS = config['SUBSAMPLE_NETS']
NUM_SAMPLING_EVAL = config['NUM_SAMPLING_EVAL']
# Min tree depth
MIN_TREE_DEPTH = config['MIN_TREE_DEPTH']
# Max tree depth
MAX_TREE_DEPTH = config['MAX_TREE_DEPTH']
# Data folder (data2, data3 etc)
data_folder = config['data_folder']
# data loading
dataset_order = ['wt', 'wtgraddata', 'inpactdata', 'inpactgraddata', 'preactdata', 
'preactgraddata', 'actdata', 'actgraddata', 'wtgradnoise', 'inpactnoise', 'inpactgradnoise', 'preactnoise',
'preactgradnoise', 'actnoise', 'actgradnoise', 'wtgradperturb', 'inpactperturb', 'inpactgradperturb', 
'preactperturb', 'preactgradperturb', 'actperturb', 'actgradperturb']

fixed_ordering = [2, 3, 10, 17, 0, 8, 15, 1, 9, 16, 4, 11, 18, 5, 12, 19, 6, 13, 20, 7, 14, 21]
# METRIC, EVOLUTION_DATASET, TEST_DATASET, EVOL_SEARCH_SPACE, TEST_SEARCH_SPACE, nds_evol_space, nds_test_space
dataset_files_available = os.listdir('/' + data_folder + '/yakhauri/datasets/')
# For each item in EVOLUTION_DATASET, if dataset_files_available has the string contained, add to list
evolution_filename_list = []
for item in EVOLUTION_DATASET:
    for file_name in dataset_files_available:
        if file_name.__contains__(item):
            evolution_filename_list.append('/' + data_folder + '/yakhauri/datasets/' + file_name)

evol_choice_dict = {}
for fpath in evolution_filename_list:
    if fpath.__contains__('_in_20.pt'):
        evol_choice_dict[fpath+str(0)] = torch.load(fpath)
    else:
        for ix in range(4):
            evol_choice_dict[fpath+str(ix)] = torch.load(fpath[:-3] + "_subs_20_id_" + str(ix) + ".pt")
# evol_choice_list = [torch.load(fpath) for fpath in evolution_filename_list]        
# Prog validity check 

prog_val_list = []
for item in PROG_VAL_DSET:
    for file_name in dataset_files_available:
        if file_name.__contains__(item):
            prog_val_list.append('/' + data_folder + '/yakhauri/datasets/' + file_name)

prog_val_choice_dict = {}
for fpath in prog_val_list:
    if fpath.__contains__('_in_20.pt'):
        prog_val_choice_dict[fpath+str(0)] = torch.load(fpath)
    else:
        for ix in range(4):
            prog_val_choice_dict[fpath+str(ix)] = torch.load(fpath[:-3] + "_subs_20_id_" + str(ix) + ".pt")
# evol_choice_list = [torch.load(fpath) for fpath in prog_val_list]        



test_filename_list = []
for item in TEST_DATASET:
    for file_name in dataset_files_available:
        if file_name.__contains__(item):
            test_filename_list.append('/' + data_folder + '/yakhauri/datasets/' + file_name)
# test_choice_dict = {}
# for fpath in test_filename_list:
#     test_choice_dict[fpath] = torch.load(fpath)
# test_choice_list = [torch.load(fpath) for fpath in test_filename_list]        

#### Genetric Tree Configuration ####
# Discuss representation here
toolbox = base.Toolbox()
# 22 static address inputs, 1 float output. 
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(torch.Tensor, 22), torch.Tensor)
pset.addPrimitive(OP0, [torch.Tensor,torch.Tensor], torch.Tensor)
pset.addPrimitive(OP1, [torch.Tensor,torch.Tensor], torch.Tensor)
pset.addPrimitive(OP2, [torch.Tensor,torch.Tensor], torch.Tensor)
pset.addPrimitive(OP3, [torch.Tensor,torch.Tensor], torch.Tensor)
pset.addPrimitive(OP4, [torch.Tensor,torch.Tensor], torch.Tensor)
pset.addPrimitive(OP5, [torch.Tensor,torch.Tensor], torch.Tensor)
# pset.addPrimitive(OP6, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP7, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP8, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP9, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP10, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP11, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP12, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP13, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP14, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP15, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP16, [torch.Tensor], torch.Tensor)
# pset.addPrimitive(OP17, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP18, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP19, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP20, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP21, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP22, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP23, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP24, [torch.Tensor], torch.Tensor)
pset.addPrimitive(OP25, [torch.Tensor,torch.Tensor], torch.Tensor)
pset.addPrimitive(OP26, [torch.Tensor,torch.Tensor], torch.Tensor)
pset.addPrimitive(OP27, [torch.Tensor,torch.Tensor], torch.Tensor)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=MIN_TREE_DEPTH, max_=MAX_TREE_DEPTH)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
# toolbox.register("compile_parallel", gp.compile, pset=pset)

def labels_to_true_labels(label):
    true_label = {}
    for k,v in label.items():
        if v.__contains__("OP"):
            true_label[k] = OPLIST[int(v[2:])]
        else:
            true_label[k] = dataset_order[int(v[3:])]
    return true_label

def graph_computation(expr, suffix):
    nodes, edges, labels = gp.graph(expr)
    labels = labels_to_true_labels(labels)
    g = pgv.AGraph(nodesep=1,strict=False, overlap=False, splines='true')
    g.node_attr["shape"] = "box"
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")
    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw(logfolder + "/trees_explored/tree_" + suffix + ".pdf")

def graph_pearson_evol(inps):
    individual, current_evol_full_dict, current_dkey = inps
    tau_list = []
    pearson_list = []
    # iterate_over = list(evol_choice_dict.keys())
    # for dkey in iterate_over:
    # print("Dataset in progress: ", dkey)
    # master_dict = evol_choice_dict[dkey]
    master_dict = current_evol_full_dict
    dkey = current_dkey
    # master_dict = dict_to_iterate[dkey]
    # It may be better to just load the data-set here.
    nets_score_set = []
    nets_test_acc_set = []
    keys_to_eval = master_dict.keys()
    func = toolbox.compile(expr=individual)
    for net_key in tqdm(keys_to_eval):
        per_layer_score = []
        for layer_key in master_dict[net_key].keys():
            if layer_key == 'net_info':
                try:
                    nets_test_acc_set.append(master_dict[net_key][layer_key]['test-accuracy'])
                except:
                    nets_test_acc_set.append(master_dict[net_key][layer_key]['test_accuracy'])
            else:
                eval_out = eval_layer(func, master_dict[net_key][layer_key])
                eval_out = torch.sum(eval_out)/torch.numel(eval_out)
                if math.isnan(eval_out.item()) or math.isinf(eval_out.item()):
                    return -1,
                if eval_out.item() == -100:
                    return -1,
                per_layer_score.append(eval_out.item())
        nets_score_set.append(per_layer_score)
    if any(-100 in x for x in nets_score_set):
        return -1,
    else:
        nets_processed_score = [sum(x)/len(x) for x in nets_score_set]
        if np.isfinite(np.asarray(nets_processed_score)).all()==False:
            return -1,
        tau, p_score = stats.kendalltau(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
        pearson, _ = stats.pearsonr(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
        suffix = dkey + "_" + str(individual.fitness.values[0]) + "_eval_tau_" + str(tau)[:6] + "_pearson_" + str(pearson)[:6]
        suffix = suffix.split("/")[-1]
        tau_list.append(tau)
        pearson_list.append(pearson)
        plt.title(' Pearson: ' + str(pearson)[:6] + "; KendallTau: " + str(tau)[:6])
        plt.scatter(nets_test_acc_set, nets_processed_score)
        plt.xlabel('Accuracy')
        plt.ylabel('Score')
        if not os.path.exists(logfolder + "/graphs/" + str(individual.fitness.values[0])):
            os.mkdir(logfolder + "/graphs/" + str(individual.fitness.values[0]))
        path = logfolder + "/graphs/" + str(individual.fitness.values[0]) + "/graph_" + suffix + ".png"
        plt.savefig(path)
        plt.clf()
        plt.title(' Pearson: ' + str(pearson)[:6] + "; KendallTau: " + str(tau)[:6])
        plt.scatter(nets_processed_score, nets_test_acc_set)
        plt.ylabel('Accuracy')
        plt.xlabel('Score')
        if not os.path.exists(logfolder + "/graphs/T" + str(individual.fitness.values[0])):
            os.mkdir(logfolder + "/graphs/T" + str(individual.fitness.values[0]))
        path = logfolder + "/graphs/T" + str(individual.fitness.values[0]) + "/graph_" + suffix + ".png"
        plt.savefig(path)
        plt.clf()
    return (tau, pearson)

def graph_pearson_test(inps):
    individual, current_test_full_dict, current_dkey = inps
    # dataset can be evol or test
    # if dataset=='evol':
    #     dict_to_iterate = evol_choice_dict
    # else:
    #     dict_to_iterate = test_choice_dict
    # Go through each 
    tau_list = []
    pearson_list = []
    # iterate_over = test_filename_list
    # for dkey in iterate_over:
    # print("Dataset in progress: ", dkey)
    # print("Loading data-set for testing...")
    # master_dict = torch.load(dkey)
    # print("Dataset _finally_ loaded")
    master_dict = current_test_full_dict
    dkey = current_dkey
    # master_dict = dict_to_iterate[dkey]
    # It may be better to just load the data-set here.
    nets_score_set = []
    nets_test_acc_set = []
    keys_to_eval = master_dict.keys()
    func = toolbox.compile(expr=individual)
    for net_key in tqdm(keys_to_eval):
        per_layer_score = []
        for layer_key in master_dict[net_key].keys():
            if layer_key == 'net_info':
                try:
                    nets_test_acc_set.append(master_dict[net_key][layer_key]['test-accuracy'])
                except:
                    nets_test_acc_set.append(master_dict[net_key][layer_key]['test_accuracy'])
            else:
                eval_out = eval_layer(func, master_dict[net_key][layer_key])
                eval_out = torch.sum(eval_out)/torch.numel(eval_out)
                if math.isnan(eval_out.item()) or math.isinf(eval_out.item()):
                    return -1,
                if eval_out.item() == -100:
                    return -1,
                per_layer_score.append(eval_out.item())
        nets_score_set.append(per_layer_score)
    if any(-100 in x for x in nets_score_set):
        return -1,
    else:
        nets_processed_score = [sum(x)/len(x) for x in nets_score_set]
        if np.isfinite(np.asarray(nets_processed_score)).all()==False:
            return -1,
        tau, p_score = stats.kendalltau(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
        pearson, _ = stats.pearsonr(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
        suffix = dkey + "_" + str(individual.fitness.values[0]) + "_test_tau_" + str(tau)[:6] + "_pearson_" + str(pearson)[:6]
        suffix = suffix.split("/")[-1]
        tau_list.append(tau)
        pearson_list.append(pearson)
        plt.title(' Pearson: ' + str(pearson)[:6] + "; KendallTau: " + str(tau)[:6])
        plt.scatter(nets_test_acc_set, nets_processed_score)
        plt.xlabel('Accuracy')
        plt.ylabel('Score')
        if not os.path.exists(logfolder + "/graphs/" + str(individual.fitness.values[0])):
            os.mkdir(logfolder + "/graphs/" + str(individual.fitness.values[0]))
        path = logfolder + "/graphs/" + str(individual.fitness.values[0]) + "/graph_" + suffix + ".png"
        # path = logfolder + "/graphs/graph_" + suffix + ".png"
        plt.savefig(path)
        plt.clf()
        plt.title(' Pearson: ' + str(pearson)[:6] + "; KendallTau: " + str(tau)[:6])
        plt.scatter(nets_processed_score, nets_test_acc_set)
        plt.ylabel('Accuracy')
        plt.xlabel('Score')
        if not os.path.exists(logfolder + "/graphs/T" + str(individual.fitness.values[0])):
            os.mkdir(logfolder + "/graphs/T" + str(individual.fitness.values[0]))
        path = logfolder + "/graphs/T" + str(individual.fitness.values[0]) + "/graph_" + suffix + ".png"
        # path = logfolder + "/graphs/graph_" + suffix + ".png"
        plt.savefig(path)
        plt.clf()
    return (tau, pearson)

def populate(runtime_addrs_space, net_stats):
    # Here, we populate the static run-time space
    net_addr_key = list(net_stats.keys())
    map_list = []
    for i in range(STATIC_ADDRS):
        try:
            runtime_addrs_space[i] = net_stats[net_addr_key[fixed_ordering[i-1]]]
            map_list.append(net_stats[net_addr_key[fixed_ordering[i-1]]])
        except Exception as  e:
            print(e)
            print(net_stats)
            print(runtime_addrs_space)
            print(net_stats.keys())
            print(runtime_addrs_space.keys())
    return runtime_addrs_space, map_list

def eval_layer(func, net_stats): 
    runtime_addrs_space = {}
    runtime_addrs_space, map_list = populate(runtime_addrs_space, net_stats)
    try:
        output = func(*map_list)
    except Exception as e:
        # print(e)
        # print("Function failed")
        return torch.Tensor([-100])
    return output

def evalOneMaxMultiSet(individual):
    # [TODO] Test on all EVOLUTION_DATASET options, choose the minimal metric???
    # OR
    # [IMPLEMENTED] On each 'NUM_SAMPLING_EVAL', evaluate on a randomly sampled EVOLUTION_DATASET  
    net_score_superset = []
    net_val_acc_superset = []
    net_test_acc_superset = []
    if len(EVOLUTION_DATASET) < NUM_SAMPLING_EVAL:
        rsample = random.sample(range(len(evol_choice_dict.keys())), NUM_SAMPLING_EVAL)
    for i in range(NUM_SAMPLING_EVAL):
        if len(EVOLUTION_DATASET) >= NUM_SAMPLING_EVAL:
            # It is same as ==, anything > NUM_SAMPLING_EVAL wont get sampled
            master_dict = evol_choice_dict[list(evol_choice_dict.keys())[4*i + random.randint(0, 3)]]
        else:
            # Here, the evol_choice_dict is assumed to be greater than NUM_SAMPLING_EVAL
            # Some may get sampled more than once randomly
            master_dict = evol_choice_dict[list(evol_choice_dict.keys())[rsample[i]]]
                        #     # If number of sampling evaluations is equal to the number of datasets in consideration,
                        #     # Make sure all of them are evaluated
                        #     if len(EVOLUTION_DATASET)==NUM_SAMPLING_EVAL:
                        #         # if fpath.__contains__('_in_20.pt'):
                        #         #     master_dict = evol_choice_dict[list(evol_choice_dict.keys())[0]]
                        #         # else:
                        #         master_dict = evol_choice_dict[list(evol_choice_dict.keys())[4*i + random.randint(0, 3)]]
                        #     elif len(evol_choice_dict.keys()) >= NUM_SAMPLING_EVAL:
                        #         master_dict = evol_choice_dict[list(evol_choice_dict.keys())[rsample[i]]]
                        #     else:
                        #         master_dict = evol_choice_dict[list(evol_choice_dict.keys())[0]]
                        #     # rand_id_dict = random.randint(0, NUM_SAMPLING_EVAL)
                        #     # master_dict = evol_choice_dict[]
                        # # for ix in range(NUM_SAMPLING_EVAL):
                        # #     evol_choice_dict[fpath+str(ix)] = torch.load(fpath[:-3] + "_subs_20_id_" + str(ix)) + ".pt"
        nets_score_set = []
        nets_test_acc_set = []
        if SUBSAMPLE_NETS == len(list(master_dict.keys())):
            keys_to_eval = [x for x in range(SUBSAMPLE_NETS)]
        elif SUBSAMPLE_NETS > len(list(master_dict.keys())):
            keys_to_eval = [random.randint(0, len(list(master_dict.keys()))-1)  for _ in range(SUBSAMPLE_NETS)]
        else: 
            # keys_to_eval = [random.randint(0, len(list(master_dict.keys()))-1)  for _ in range(SUBSAMPLE_NETS)]
            keys_to_eval = random.sample(range(len(list(master_dict.keys()))), SUBSAMPLE_NETS)
        func = toolbox.compile(expr=individual)
        # In each subsampled space, evaluate every network
        for net_key in keys_to_eval:
            per_layer_score = []
            # Go through each attribute
            for layer_key in master_dict[net_key].keys():
                # If attribute is not a layer, but the network info, append that information
                if layer_key == 'net_info':
                    try:
                        nets_test_acc_set.append(master_dict[net_key][layer_key]['test-accuracy'])
                    except:
                        nets_test_acc_set.append(master_dict[net_key][layer_key]['test_accuracy'])
                # Attribute is a layer, evaluate this later and append it to per_layer_score
                else:
                    eval_out = eval_layer(func, master_dict[net_key][layer_key])
                    eval_out = torch.sum(eval_out)/torch.numel(eval_out)
                    if math.isnan(eval_out.item()) or math.isinf(eval_out.item()):
                        return -1,
                    if eval_out.item() == -100:
                        return -1,
                    per_layer_score.append(eval_out.item())
            # Append a list of per layer scores to the net super score
            nets_score_set.append(per_layer_score)
        # For this entire set, append the list of list of per layer scores
        net_score_superset.append(nets_score_set)
        net_test_acc_superset.append(nets_test_acc_set)
    # net_score_superset structure is : first level index contains eval sample, each index therein contains 
    print(".", end='', flush=True)
    for item in net_score_superset:
        if any(-100 in x for x in item):
            return -1
    else:
        nets_processed_score = []
        for item in net_score_superset:
            nets_processed_score.append([sum(x)/len(x) for x in item])
        if np.isfinite(np.asarray(nets_processed_score)).all()==False:
            return -1,
        metric_scores = []
        for idx in range(len(nets_processed_score)):
            if METRIC=='kendalltau':
                metric, p_score = stats.kendalltau(np.asarray(nets_processed_score[idx]), np.asarray(net_test_acc_superset[idx]))
            else:
                metric, _ = stats.pearsonr(np.asarray(nets_processed_score[idx]), np.asarray(net_test_acc_superset[idx]))
            if math.isnan(metric) or math.isinf(metric):
                return -1,
            metric_scores.append(metric)
        metric_scores_processed = [x  if (x==-1 or x>0) else -1*x for x in metric_scores]
        return min(metric_scores_processed),
     
def evalOneMaxMultiSetMini(individual):
    net_score_superset = []
    net_val_acc_superset = []
    net_test_acc_superset = []
    for i in range(2):
        master_dict = prog_val_choice_dict[list(prog_val_choice_dict.keys())[i]]
        nets_score_set = []
        nets_test_acc_set = []
        keys_to_eval = [0, 1]
        func = toolbox.compile(expr=individual)
        for net_key in keys_to_eval:
            per_layer_score = []
            for layer_key in master_dict[net_key].keys():
                if layer_key == 'net_info':
                    try:
                        nets_test_acc_set.append(master_dict[net_key][layer_key]['test-accuracy'])
                    except:
                        nets_test_acc_set.append(master_dict[net_key][layer_key]['test_accuracy'])
                else:
                    eval_out = eval_layer(func, master_dict[net_key][layer_key])
                    eval_out = torch.sum(eval_out)/torch.numel(eval_out)
                    if math.isnan(eval_out.item()) or math.isinf(eval_out.item()):
                        return -1,
                    if eval_out.item() == -100:
                        return -1,
                    per_layer_score.append(eval_out.item())
            nets_score_set.append(per_layer_score)
        net_score_superset.append(nets_score_set)
        net_test_acc_superset.append(nets_test_acc_set)
    print(".", end='', flush=True)
    for item in net_score_superset:
        if any(-100 in x for x in item):
            return -1
    else:
        nets_processed_score = []
        for item in net_score_superset:
            nets_processed_score.append([sum(x)/len(x) for x in item])
        if np.isfinite(np.asarray(nets_processed_score)).all()==False:
            return -1,
        metric_scores = []
        for idx in range(len(nets_processed_score)):
            if METRIC=='kendalltau':
                metric, p_score = stats.kendalltau(np.asarray(nets_processed_score[idx]), np.asarray(net_test_acc_superset[idx]))
            else:
                metric, _ = stats.pearsonr(np.asarray(nets_processed_score[idx]), np.asarray(net_test_acc_superset[idx]))
            if math.isnan(metric) or math.isinf(metric):
                return -1,
            metric_scores.append(metric)
        metric_scores_processed = [x  if (x==-1 or x>0) else -1*x for x in metric_scores]
        return min(metric_scores_processed),


toolbox.register("evaluate", evalOneMaxMultiSet)
toolbox.register("evaluate_mini", evalOneMaxMultiSetMini)
toolbox.register("graph_pearson_evol", graph_pearson_evol)
toolbox.register("graph_pearson_test", graph_pearson_test)
toolbox.register("select", tools.selTournament, tournsize=TOURSIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

checkpoint = ''
if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG, filename=logfolder + "/logfile.log", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
    if nproc > 1:
        pool = multiprocessing.Pool(processes=nproc)
        toolbox.register("map", pool.map)

    if len(checkpoint)>0:
        # A file name has been given, then load the data from the file
        with open(checkpoint, "r") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        random.setstate(cp["rndstate"])
    else:
        # Here, instead of doing this, ensure that all population individuals have fitness > -1
        population = toolbox.population(n=POPSIZE)
        # true_pop = []
        # while len(true_pop)<POPSIZE:
        #     print("Current have ", len(true_pop), " valid individuals")
        #     print("Trying ", POPSIZE, " more individuals to fill agents")
        #     population = toolbox.population(n=POPSIZE)
        #     fits = toolbox.map(toolbox.evaluate_mini, population)
        #     for fit, ind in zip(fits, population):
        #         if fit[0]!=-1:
        #             true_pop.append(ind)
        # print("Total population size: ", len(true_pop))
        # population = true_pop[:POPSIZE]
    print("Starting evolution") 
    logging.info("Starting Evolution")
    gen_imp = []
    top10_list = []
    for gen in range(NGEN): 
        a = time.time()
        valid_offsp = []
        st_offspr_time = time.time()
        while len(valid_offsp) < POPSIZE:
            print("Currently have ", len(valid_offsp), " valid individuals")
            print("Trying ", POPSIZE, " more individuals to fill agents")
            offspring = algorithms.varOr(population=population, toolbox=toolbox, lambda_=lambda_, cxpb=CXPR, mutpb=MUTPR)
            temp_fits = toolbox.map(toolbox.evaluate_mini, offspring)
            for fit, ind in zip(temp_fits, offspring):
                if fit[0]!=-1:
                    valid_offsp.append(ind)
        offspring = valid_offsp[:POPSIZE]
        print("Time taken to generate ", POPSIZE, " offsprings: ", time.time() - st_offspr_time)
        offspring_heights = [x.height for x in offspring]
        print("Program Lengths: ", offspring_heights, " Mean: ", sum(offspring_heights)/len(offspring_heights), " Max: ", max(offspring_heights))
        # exit(0)
        logging.info("Program Lengths: ")
        logging.info([x.height for x in offspring])
        print("."*len(offspring))
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        gen_imp.extend([x.fitness.values[0] for x in offspring])
        rewards = [x.fitness.values[0] for x in offspring]
        rewards.sort(reverse=True)
        print("\ngen: ", str(gen), "\nreward: ", [str(x)[:5] for x in rewards])
        logging.info("gen: " +  str(gen) + "\nreward: ")
        logging.info([str(x)[:5] for x in rewards])
        gen_imp.sort(reverse=True)
        print("best agents: ", gen_imp[:10])
        logging.info("best agents: ")
        logging.info(gen_imp[:10])
        population = toolbox.select(offspring, k=MU)
        # Remove duplicate (same fitness) graphs here
        pre_prune_population = len(population)
        population = list(toolz.unique(population, key=lambda x: x.fitness.values))
        print("Population pre duplicate pruning: ", pre_prune_population, "\t Population post duplicate pruning: ", len(population))
        top10_iter = tools.selBest(population, k=10)
        top10_list.append(top10_iter)
        for graph_comp in top10_iter:
            graph_computation(graph_comp, suffix="perf_" + str(graph_comp.fitness.values[0]) + "_gen_" + str(gen))
        print("Time elasped: ", time.time() - a)
        logging.info("Time elasped: " +  str(time.time() - a))
    
    
    if nproc > 1:
        pool.close()
        pool.terminate()
        pool.join()
    
    top_from_all = [item for sublist in top10_list for item in sublist]
    top_all_30 = tools.selBest(top_from_all, k=50)
    top_all_30_fitness = [x.fitness.values[0] for x in top_all_30]
    top_all_indexes = [top_all_30_fitness.index(x) for x in set(top_all_30_fitness)]
    topx_all_unique_list = [x for idx, x in enumerate(top_all_30) if idx in top_all_indexes]
    topx_all_unique_list = [x for x in topx_all_unique_list if x.height < 9]
    if len(topx_all_unique_list)>3:
        topx_all_unique_list = topx_all_unique_list[:3]
    # if nproc > 1:
    #     pool = multiprocessing.Pool(processes=nproc//4)
    #     toolbox.register("map", pool.map)

    # Select some best
    top30 = tools.selBest(population, k=50)
    # Get their fitnesses
    top30_fitness = [x.fitness.values[0] for x in top30]
    # Get the indexes of unique fitnesses (Identical fitnesses removed)
    indexes = [top30_fitness.index(x) for x in set(top30_fitness)]
    # Filter out duplicate individuals
    topx_unique_list = [x for idx, x in enumerate(top30) if idx in indexes]
    # Filter out individuals which have a height of greater than 8
    topx_unique_list = [x for x in topx_unique_list if x.height <= 9]
    # If there are more than 3 in this list, only evaluate 3
    if len(topx_unique_list)>3:
        topx_unique_list = topx_unique_list[:3]
    print("Top 3 selected")
    # iterate over EVOLUTION DATASET
    overall_perf_dict = {}
    for dkey in EVOLUTION_DATASET:
        current_dkey = dkey
        st_l = time.time()
        print("Graphing for evolutionary sets...")
        print("Datset: ", dkey)
        current_evol_full_dict = torch.load('/' + data_folder + '/yakhauri/datasets/'+dkey)
        print("Dataset loading time: ", time.time() - st_l)
        # toolbox.map(toolbox.graph_pearson_evol, topx_unique_list, repeat(current_evol_full_dict), repeat(current_dkey))
        # aggr_inputs = [(x, current_evol_full_dict, current_dkey) for x in topx_unique_list]
        # toolbox.map(toolbox.graph_pearson_evol, aggr_inputs)
        tau_, pearson_ = graph_pearson_evol((topx_unique_list[0], current_evol_full_dict, current_dkey))
        print("Tau: ", tau_, "\t Pearson: ", pearson_)
        overall_perf_dict[dkey+"topx_unique_0"] = (tau_, pearson_)
        # if len(topx_unique_list)>1:
        #     graph_pearson_evol((topx_unique_list[1], current_evol_full_dict, current_dkey))
        # if len(topx_unique_list)>2:
        #     graph_pearson_evol((topx_unique_list[2], current_evol_full_dict, current_dkey))
        tau_, pearson_ = graph_pearson_evol((topx_all_unique_list[0], current_evol_full_dict, current_dkey))
        print("Tau: ", tau_, "\t Pearson: ", pearson_)
        overall_perf_dict[dkey+"topx_all_unique_0"] = (tau_, pearson_)
        if len(topx_all_unique_list)>1:
            tau_, pearson_ = graph_pearson_evol((topx_all_unique_list[1], current_evol_full_dict, current_dkey))
            print("Tau: ", tau_, "\t Pearson: ", pearson_)
            overall_perf_dict[dkey+"topx_all_unique_1"] = (tau_, pearson_)
        # if len(topx_all_unique_list)>2:
        #     graph_pearson_evol((topx_all_unique_list[2], current_evol_full_dict, current_dkey))
        # toolbox.map(toolbox.graph_pearson_evol, topx_unique_list, [current_evol_full_dict for _ in range(len(topx_unique_list))], [current_dkey for _ in range(len(topx_unique_list))])
    for dkey in TEST_DATASET:
        current_dkey = dkey
        st_l = time.time()
        print("Graphing for test sets...")
        print("Datset: ", dkey)
        if dkey.__contains__('ImageNet_NDS'):
            current_test_full_dict = torch.load('/' + data_folder + '/yakhauri/datasets/'+dkey)
        else:
            current_test_full_dict = torch.load('/' + data_folder + '/yakhauri/datasets/'+dkey)
        print("Dataset loading time: ", time.time() - st_l)
        tau_, pearson_ = graph_pearson_test((topx_unique_list[0], current_test_full_dict, current_dkey))
        print("Tau: ", tau_, "\t Pearson: ", pearson_)
        overall_perf_dict[dkey+"topx_unique_0"] = (tau_, pearson_)
        # if len(topx_unique_list)>1:
        #     graph_pearson_test((topx_unique_list[1], current_test_full_dict, current_dkey))
        # if len(topx_unique_list)>2:
        #     graph_pearson_test((topx_unique_list[2], current_test_full_dict, current_dkey))
        tau_, pearson_ = graph_pearson_test((topx_all_unique_list[0], current_test_full_dict, current_dkey))
        print("Tau: ", tau_, "\t Pearson: ", pearson_)
        overall_perf_dict[dkey+"topx_all_unique_0"] = (tau_, pearson_)
        if len(topx_all_unique_list)>1:
            tau_, pearson_ = graph_pearson_test((topx_all_unique_list[1], current_test_full_dict, current_dkey))
            print("Tau: ", tau_, "\t Pearson: ", pearson_)
            overall_perf_dict[dkey+"topx_all_unique_1"] = (tau_, pearson_)
        # if len(topx_all_unique_list)>2:
        #     graph_pearson_test((topx_all_unique_list[2], current_test_full_dict, current_dkey))
            # aggr_inputs = [(x, current_test_full_dict, current_dkey) for x in topx_unique_list]
            # toolbox.map(toolbox.graph_pearson_test, aggr_inputs)
            # toolbox.map(toolbox.graph_pearson_test, topx_unique_list, repeat(current_test_full_dict), repeat(current_dkey))
    pprint(overall_perf_dict)
    for i in topx_unique_list:
        indv_fitness = i.fitness.values[0]
        try:
            print("Train fitness: ", indv_fitness, " Tree Height: ", str(i.height))
            suffix = "_topx_search_" + METRIC + "_" + str(indv_fitness) + "_gen_" + str(gen)
            graph_computation(i, suffix=suffix)
            # graph_pearson(i, suffix=suffix, dataset='evol')
            logging.info(str(i.height)  +  suffix)
            print(str(i.height)  + suffix)
            print("*********************")
            logging.info("*********************")
        except Exception as e:
            print("Exception: ", e)
            print(indv_fitness)
            print(i, " Failed")
            logging.info(i)
            logging.info("Failed")
    logging.info(overall_perf_dict)

    # for i in topx_unique_list:
    #     indv_fitness = i.fitness.values[0]
    #     try:
    #         print(indv_fitness)
    #         suffix = "_topx_search_test_" + METRIC + "_" + str(indv_fitness) + "_gen_" + str(gen)
    #         graph_computation(i, suffix=suffix)
    #         # graph_pearson(i, suffix=suffix, dataset='test')
    #         logging.info(str(i.height) + suffix)
    #         print(str(i.height)  + suffix)
    #         print("*********************")
    #         logging.info("*********************")
    #     except Exception as e:
    #         print("Exception: ", e)
    #         print(indv_fitness)
    #         print(i, " Failed")
    #         logging.info(i)
    #         logging.info("Failed")

    with open(logfolder + "/models/models.pkl", "wb") as cp_file:
        cp = dict(population=topx_unique_list, rndstate=random.getstate())
        pickle.dump(cp, cp_file)

    with open(logfolder + "/models/ending_population.pkl", "wb") as cp_file:
        cp = dict(population=population, rndstate=random.getstate())
        pickle.dump(cp, cp_file)
    # if nproc > 1:
    #     pool.close()
    
        

##################### BACKUP CODE #####################


# required_order = ['wt', 'wtgraddata', 'wtgradnoise', 'wtgradperturb',
# 'inpactdata', 'inpactnoise', 'inpactperturb', 'inpactgraddata', 'inpactgradnoise', 'inpactgradperturb',
# 'preactdata', 'preactnoise', 'preactperturb', 'preactgraddata', 'preactgradnoise', 'preactgradperturb',
# 'actdata', 'actnoise', 'actperturb', 'actgraddata', 'actgradnoise', 'actgradperturb']

# fixed_ordering = [0, 1, 8, 15, 2, 9, 16, 3, 10, 17, 4, 11, 18, 5, 12, 19, 6, 13, 20, 7, 14, 21]
# dataset_idx = [(0, 'wt'), (1, 'wtgraddata'), (2, 'inpactdata'), (3, 'inpactgraddata'), (4, 'preactdata'), 
# (5, 'preactgraddata'), (6, 'actdata'), (7, 'actgraddata'), (8, 'wtgradnoise'), (9, 'inpactnoise'), 
# (10, 'inpactgradnoise'), (11, 'preactnoise'), (12, 'preactgradnoise'), (13, 'actnoise'), 
# (14, 'actgradnoise'), (15, 'wtgradperturb'), (16, 'inpactperturb'), (17, 'inpactgradperturb'), 
# (18, 'preactperturb'), (19, 'preactgradperturb'), (20, 'actperturb'), (21, 'actgradperturb')]


# dset_key_binding = {0: 'wt', 1: 'wtgraddata', 2: 'inpactdata', 3: 'inpactgraddata', 4: 'preactdata', 
#                     5: 'preactgraddata', 6: 'actdata', 7: 'actgraddata', 8: 'wtgradnoise', 9: 'inpactnoise', 
#                     10: 'inpactgradnoise', 11: 'preactnoise', 12: 'preactgradnoise', 13: 'actnoise', 
#                     14: 'actgradnoise', 15: 'wtgradperturb', 16: 'inpactperturb', 17: 'inpactgradperturb', 
#                     18: 'preactperturb', 19: 'preactgradperturb', 20: 'actperturb', 21: 'actgradperturb'}

################################################### New ordering 


# required_order = ['wt', 'wtgraddata', 'wtgradnoise', 'wtgradperturb',
# 'inpactdata', 'inpactnoise', 'inpactperturb', 'inpactgraddata', 'inpactgradnoise', 'inpactgradperturb',
# 'preactdata', 'preactnoise', 'preactperturb', 'preactgraddata', 'preactgradnoise', 'preactgradperturb',
# 'actdata', 'actnoise', 'actperturb', 'actgraddata', 'actgradnoise', 'actgradperturb']


# static_addr =  ['inpactdata', 'inpactgraddata', 'wt', 'wtgraddata', 'preactdata', 'preactgraddata', 'actdata', 'actgraddata', 
#                 'inpactnoise', 'inpactgradnoise', 'wtgradnoise', 'preactnoise', 'preactgradnoise', 'actnoise', 'actgradnoise', 
#                 'inpactperturb', 'inpactgradperturb', 'wtgradperturb', 'preactperturb', 'preactgradperturb', 'actperturb', 'actgradperturb']

# fixed_ordering = [2, 3, 10, 17, 0, 8, 15, 1, 9, 16, 4, 11, 18, 5, 12, 19, 6, 13, 20, 7, 14, 21]

# {0: 'inpactdata', 1: 'inpactgraddata', 2: 'wt', 3: 'wtgraddata', 4: 'preactdata', 5: 'preactgraddata', 6: 'actdata', 7: 'actgraddata', 
# 8: 'inpactnoise', 9: 'inpactgradnoise', 10: 'wtgradnoise', 11: 'preactnoise', 12: 'preactgradnoise', 13: 'actnoise', 14: 'actgradnoise',
# 15: 'inpactperturb', 16: 'inpactgradperturb', 17: 'wtgradperturb', 18: 'preactperturb', 19: 'preactgradperturb', 20: 'actperturb', 21: 'actgradperturb'}

# def evalOneMax_test_test(individual):
#     # Test on all TEST_DATASET options, choose the minimal metric
#     master_metric_list = []
#     for master_dict_key in test_choice_dict.keys():
#         master_dict = test_choice_dict[master_dict_key]
#         nets_score_set = []
#         nets_test_acc_set = []
#         keys_to_eval = master_dict.keys()
#         # keys_to_eval = [idx for idx in range(NUM_NETS)]
#         func = toolbox.compile(expr=individual)
#         for net_key in tqdm(keys_to_eval):
#             per_layer_score = []
#             flag_empty = 0
#             for layer_key in master_dict[net_key].keys():
#                 if layer_key == 'net_info':
#                     try:
#                         nets_test_acc_set.append(master_dict[net_key][layer_key]['test-accuracy'])
#                     except:
#                         nets_test_acc_set.append(master_dict[net_key][layer_key]['test_accuracy'])
#                 else:
#                     # assert len(list(master_dict[net_key][layer_key].keys()))==22, str(net_key) + "_" + str(layer_key)
#                     if len(list(master_dict[net_key][layer_key].keys()))!=22:
#                         flag_empty=1
#                         continue
#                     eval_out = eval_layer(func, master_dict[net_key][layer_key])
#                     eval_out = torch.sum(eval_out)/torch.numel(eval_out)
#                     if math.isnan(eval_out.item()) or math.isinf(eval_out.item()):
#                         return -1,
#                     if eval_out.item() == -100:
#                         return -1,
#                     per_layer_score.append(eval_out.item())
#             if flag_empty!=1:
#                 nets_score_set.append(per_layer_score)
#         if any(-100 in x for x in nets_score_set):
#             return -1,
#         else:
#             nets_processed_score = [sum(x)/len(x) for x in nets_score_set]
#             if(len(set(nets_processed_score))<=1):
#                 return -1,
#             if METRIC=='kendalltau':
#                 metric, p_score = stats.kendalltau(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
#             else:
#                 metric, _ = stats.pearsonr(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
#             if np.isfinite(np.asarray(nets_processed_score)).all()==False:
#                 return -1,
#             if metric != -1 and metric < 0:
#                 metric = -1*metric
#             master_metric_list.append(metric)
#     return min(master_metric_list),

# Test on all TEST_DATASET options, choose the minimal metric
# metric = evalOneMax_test_test(i)[0]
# suffix = "_".join(EVOLUTION_DATASET) + "_topx_search_" + METRIC + "_" + str(indv_fitness) + "_gen_" + str(gen) + "_".join(TEST_DATASET)
