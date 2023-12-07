import random
import operator
from deap import creator, base, tools, algorithms, gp
import numpy as np
import torch
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
from tqdm import tqdm
import pickle
from get_inference_data import NetStatisticsFetcher

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

#### This file runs evaluation by generating data on the fly
#### In final evaluation and test stages, it loads data.

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

net_stat_fetcher = NetStatisticsFetcher(batch_size=1)

# Evolution dataset
# EVOLUTION_DATASET = ['cifar10_NDS_nds_nasnet', 'cifar10_NASBench201', 'ImageNet16-120_NASBench201']
EVOLUTION_DATASET = ['net_stat_NN_22_stats_cifar10_NASBench201__1000.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_pnas_200.pt']

# Test dataset
# TEST_DATASET = ['cifar10_NDS_nds_nasnet', 'cifar10_NASBench201', 'ImageNet16-120_NASBench201']
# TEST_DATASET = ['net_stat_NN_22_stats_cifar100_NASBench201__1000.pt']
TEST_DATASET = ['net_stat_NN_22_stats_cifar100_NASBench201__1000.pt', 'net_stat_NN_22_stats_cifar10_NASBench201__1000.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_amoeba_200.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_darts_200.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_enas_200.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_nasnet_200.pt', 'net_stat_NN_22_stats_cifar10_NDS_nds_pnas_200.pt', 'net_stat_NN_22_stats_ImageNet16-120_NASBench201__1000.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_amoeba_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_darts_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_enas_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_nasnet_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_pnas_in_20.pt']
# TEST_DATASET = ['net_stat_NN_22_stats_ImageNet_NDS_nds_darts_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_enas_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_nasnet_in_20.pt', 'net_stat_NN_22_stats_ImageNet_NDS_nds_pnas_in_20.pt']
dataset_files_available = os.listdir('/data3/yakhauri/datasets/')
# To read the test and evolution dataset file names,
# split file name by '/', take the last one. Then split file name by '_'
# Take the [5] index for data-set, [6] for search space. If it is NDS, [8] for exact network space
evolution_filename_list = []
for item in EVOLUTION_DATASET:
    for file_name in dataset_files_available:
        if file_name.__contains__(item):
            evolution_filename_list.append('/data3/yakhauri/datasets/' + file_name)

global global_evol_dict
# Assign current_evol_full_dict to each dict in EVOLUTION_DATASET, and use that in graphing pearson
global current_evol_full_dict
# Assign current_test_full_dict to each dict in TEST_DATASET, and use that in graphing pearson
global current_test_full_dict


test_filename_list = []
for item in TEST_DATASET:
    for file_name in dataset_files_available:
        if file_name.__contains__(item):
            test_filename_list.append('/data3/yakhauri/datasets/' + file_name)

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
# data loading
dataset_order = ['wt', 'wtgraddata', 'inpactdata', 'inpactgraddata', 'preactdata', 
'preactgraddata', 'actdata', 'actgraddata', 'wtgradnoise', 'inpactnoise', 'inpactgradnoise', 'preactnoise',
'preactgradnoise', 'actnoise', 'actgradnoise', 'wtgradperturb', 'inpactperturb', 'inpactgradperturb', 
'preactperturb', 'preactgradperturb', 'actperturb', 'actgradperturb']

fixed_ordering = [2, 3, 10, 17, 0, 8, 15, 1, 9, 16, 4, 11, 18, 5, 12, 19, 6, 13, 20, 7, 14, 21]

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
pset.addPrimitive(OP6, [torch.Tensor], torch.Tensor)
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

# Instead of loading data for each individual
# run all individuals on one dataset 
def graph_pearson_evol(individual):
    tau_list = []
    pearson_list = []
    # iterate_over = list(evol_choice_dict.keys())
    iterate_over = evolution_filename_list
    for dkey in iterate_over:
        print("Loading data-set for evaluating...")
        # master_dict = evol_choice_dict[dkey]
        master_dict = torch.load(dkey)
        print("Dataset _finally_ loaded")
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
            suffix = dkey + "_eval_tau_" + str(tau)[:6] + "_pearson_" + str(pearson)[:6]
            suffix = suffix.split("/")[-1]
            tau_list.append(tau)
            pearson_list.append(pearson)
            plt.title(' Pearson: ' + str(pearson)[:6] + "; KendallTau: " + str(tau)[:6])
            plt.scatter(nets_test_acc_set, nets_processed_score)
            plt.xlabel('Accuracy')
            plt.ylabel('Score')
            path = logfolder + "/graphs/graph_" + suffix + ".png"
            plt.savefig(path)
            plt.clf()

def graph_pearson_test(individual):
    # dataset can be evol or test
    # if dataset=='evol':
    #     dict_to_iterate = evol_choice_dict
    # else:
    #     dict_to_iterate = test_choice_dict
    # Go through each 
    tau_list = []
    pearson_list = []
    iterate_over = test_filename_list
    for dkey in iterate_over:
        print("Loading data-set for testing...")
        master_dict = torch.load(dkey)
        print("Dataset _finally_ loaded")
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
            suffix = dkey + "_test_tau_" + str(tau)[:6] + "_pearson_" + str(pearson)[:6]
            suffix = suffix.split("/")[-1]
            tau_list.append(tau)
            pearson_list.append(pearson)
            plt.title(' Pearson: ' + str(pearson)[:6] + "; KendallTau: " + str(tau)[:6])
            plt.scatter(nets_test_acc_set, nets_processed_score)
            plt.xlabel('Accuracy')
            plt.ylabel('Score')
            path = logfolder + "/graphs/graph_" + suffix + ".png"
            plt.savefig(path)
            plt.clf()

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
    # New Implementation:
    # Create a dictionary of NUM_SAMPLING_EVAL random inferences of SUBSAMPLE_NETS, keep this is a global dict
    # of name global global_evol_dict = {}
    net_score_superset = []
    net_val_acc_superset = []
    net_test_acc_superset = []
    for i in range(NUM_SAMPLING_EVAL): 
        master_dict = global_evol_dict[i]
        # master_dict = evol_choice_dict[list(evol_choice_dict.keys())[random.randint(0, len(evol_choice_dict.keys())-1)]]
        nets_score_set = []
        nets_test_acc_set = []
        keys_to_eval = list(master_dict.keys())
        # keys_to_eval = [random.randint(0, len(list(master_dict.keys()))-1)  for _ in range(SUBSAMPLE_NETS)]
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
     
def evalOneMax_test_evol(individual):
    # Test on all EVOLUTION_DATASET options, choose the minimal metric
    master_metric_list = []
    for master_dict_key in evol_choice_dict.keys():
        master_dict = evol_choice_dict[master_dict_key]
        nets_score_set = []
        nets_test_acc_set = []
        keys_to_eval = master_dict.keys()
        func = toolbox.compile(expr=individual)
        for net_key in tqdm(keys_to_eval):
            per_layer_score = []
            flag_empty = 0
            for layer_key in master_dict[net_key].keys():
                if layer_key == 'net_info':
                    try:
                        nets_test_acc_set.append(master_dict[net_key][layer_key]['test-accuracy'])
                    except:
                        nets_test_acc_set.append(master_dict[net_key][layer_key]['test_accuracy'])
                else:
                    # assert len(list(master_dict[net_key][layer_key].keys()))==22, str(net_key) + "_" + str(layer_key)
                    if len(list(master_dict[net_key][layer_key].keys()))!=22:
                        flag_empty=1
                        continue
                    eval_out = eval_layer(func, master_dict[net_key][layer_key])
                    eval_out = torch.sum(eval_out)/torch.numel(eval_out)
                    if math.isnan(eval_out.item()) or math.isinf(eval_out.item()):
                        return -1,
                    if eval_out.item() == -100:
                        return -1,
                    per_layer_score.append(eval_out.item())
            nets_score_set.append(per_layer_score)
            if flag_empty!=1:
                nets_score_set.append(per_layer_score)
        if any(-100 in x for x in nets_score_set):
            return -1,
        else:
            nets_processed_score = [sum(x)/len(x) for x in nets_score_set]
            if(len(set(nets_processed_score))<=1):
                return -1,
            if METRIC=='kendalltau':
                metric, p_score = stats.kendalltau(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
            else:
                metric, _ = stats.pearsonr(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
            if np.isfinite(np.asarray(nets_processed_score)).all()==False:
                return -1,
            if metric != -1 and metric < 0:
                metric = -1*metric
            master_metric_list.append(metric)
    return min(master_metric_list),
        
    

# toolbox.register("evaluate", evalOneMaxMultiSet)
toolbox.register("graph_pearson_evol", graph_pearson_evol)
toolbox.register("graph_pearson_test", graph_pearson_test)
toolbox.register("evaluate", evalOneMaxMultiSet)
toolbox.register("select", tools.selTournament, tournsize=TOURSIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=MIN_TREE_DEPTH, max_=MAX_TREE_DEPTH)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

checkpoint = ''
if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG, filename=logfolder + "/logfile.log", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


    if len(checkpoint)>0:
        # A file name has been given, then load the data from the file
        with open(checkpoint, "r") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        random.setstate(cp["rndstate"])
    else:
        population = toolbox.population(n=POPSIZE)

    print("Starting evolution") 
    logging.info("Starting Evolution")
    gen_imp = []
    top10_list = []
    for gen in range(NGEN): 
        a = time.time()
        offspring = algorithms.varOr(population=population, toolbox=toolbox, lambda_=lambda_, cxpb=CXPR, mutpb=MUTPR)
        print("Program Lengths: ", [x.height for x in offspring])
        logging.info("Program Lengths: ")
        logging.info([x.height for x in offspring])
        global_evol_dict = {}
        # Here, populate a evol dict with NUM_SAMPLING_EVAL with SUBSAMPLE_NETS networks each
        print("Generating network statistics at run-time...")
        for i_netstats in range(NUM_SAMPLING_EVAL):
            dkey = evolution_filename_list[random.randint(0, len(evolution_filename_list)-1)]
            dd = dkey.split('/')[-1]
            dd = dd.split('_')
            dset_name, dset_search_space = dd[5], dd[6]
            if dset_search_space=='NDS':
                dset_netspace = dd[8]
                if dd[9]=='in':
                    dset_netspace += '_in'
            else:
                dset_netspace = ''
            # Ensure SUBSAMPLE_NETS is smaller than 20 I guess?
            global_evol_dict[i_netstats] = net_stat_fetcher.get_many_net_inference_data(dataset=dset_name, search_space=dset_search_space, 
                                                                    network_space=dset_netspace, num_nets=SUBSAMPLE_NETS)
        if nproc > 1:
            pool = multiprocessing.Pool(processes=nproc)
            toolbox.register("map", pool.map)
        fits = toolbox.map(toolbox.evaluate, offspring)
        if nproc > 1:
            pool.close()
            pool.terminate()
            pool.join()
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        gen_imp.extend([x.fitness.values[0] for x in offspring])
        rewards = [x.fitness.values[0] for x in offspring]
        rewards.sort(reverse=True)
        print("gen: ", str(gen), "\nreward: ", [str(x)[:5] for x in rewards])
        logging.info("gen: " +  str(gen) + "\nreward: ")
        logging.info([str(x)[:5] for x in rewards])
        gen_imp.sort(reverse=True)
        print("best agents: ", gen_imp[:10])
        logging.info("best agents: ")
        logging.info(gen_imp[:10])
        population = toolbox.select(offspring, k=MU)
        top10_iter = tools.selBest(population, k=10)
        top10_list.append(top10_iter)
        for graph_comp in top10_iter:
            graph_computation(graph_comp, suffix="perf_" + str(graph_comp.fitness.values[0]) + "_gen_" + str(gen))
        print("Time elasped: ", time.time() - a)
        logging.info("Time elasped: " +  str(time.time() - a))
    
    # Select some best
    top30 = tools.selBest(population, k=50)
    # Get their fitnesses
    top30_fitness = [x.fitness.values[0] for x in top30]
    # Get the indexes of unique fitnesses (Identical fitnesses removed)
    indexes = [top30_fitness.index(x) for x in set(top30_fitness)]
    # Filter out duplicate individuals
    topx_unique_list = [x for idx, x in enumerate(top30) if idx in indexes]
    # Filter out individuals which have a height of greater than 8
    topx_unique_list = [x for x in topx_unique_list if x.height < 9]
    # If there are more than 3 in this list, only evaluate 3
    if len(topx_unique_list)>3:
        topx_unique_list = topx_unique_list[:3]

    # print("Graphing for evolutionary sets...")
    # if nproc > 1:
    #     pool = multiprocessing.Pool(processes=nproc)
    #     toolbox.register("map", pool.map)
    # toolbox.map(toolbox.graph_pearson_evol, [topx_unique_list[0]])
    # toolbox.map(toolbox.graph_pearson_evol, [topx_unique_list[1]])
    # toolbox.map(toolbox.graph_pearson_evol, [topx_unique_list[2]])
    # print("Graphing for test sets...")
    # toolbox.map(toolbox.graph_pearson_test, [topx_unique_list[0]])
    # toolbox.map(toolbox.graph_pearson_test, [topx_unique_list[1]])
    # toolbox.map(toolbox.graph_pearson_test, [topx_unique_list[2]])
    # if nproc > 1:
    #     pool.close()
    #     pool.terminate()
    #     pool.join()

    
    # # Assign current_evol_full_dict to each dict in EVOLUTION_DATASET, and use that in graphing pearson
    # global current_evol_full_dict
    # # Assign current_test_full_dict to each dict in TEST_DATASET, and use that in graphing pearson
    # global current_test_full_dict


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
    if nproc > 1:
        pool.close()
    
        

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

# def graph_pearson_evol(individual):
#     tau_list = []
#     pearson_list = []
#     iterate_over = evolution_filename_list
#     for dkey in iterate_over:
#         print("Dataset in progress: ", dkey)
#         dd = dkey.split('/')[-1]
#         dd = dd.split('_')
#         dset_name, dset_search_space = dd[5], dd[6]
#         if dset_search_space=='NDS':
#             dset_netspace = dd[8]
#             if dd[9]=='in':
#                 dset_netspace += '_in'
#         if dset_name == 'ImageNet':
#             num_nets_togen = 20
#         elif dataset == 'cifar10' and search_space=='NDS':
#             num_nets_togen = 200
#         else:
#             num_nets_togen = 1000
#         master_dict = net_stat_fetcher.get_many_net_inference_data(num_nets_togen)
#         # master_dict = dict_to_iterate[dkey]
#         # It may be better to just load the data-set here.
#         nets_score_set = []
#         nets_test_acc_set = []
#         keys_to_eval = master_dict.keys()
#         func = toolbox.compile(expr=individual)
#         for net_key in tqdm(keys_to_eval):
#             per_layer_score = []
#             for layer_key in master_dict[net_key].keys():
#                 if layer_key == 'net_info':
#                     try:
#                         nets_test_acc_set.append(master_dict[net_key][layer_key]['test-accuracy'])
#                     except:
#                         nets_test_acc_set.append(master_dict[net_key][layer_key]['test_accuracy'])
#                 else:
#                     eval_out = eval_layer(func, master_dict[net_key][layer_key])
#                     eval_out = torch.sum(eval_out)/torch.numel(eval_out)
#                     if math.isnan(eval_out.item()) or math.isinf(eval_out.item()):
#                         return -1,
#                     if eval_out.item() == -100:
#                         return -1,
#                     per_layer_score.append(eval_out.item())
#             nets_score_set.append(per_layer_score)
#         if any(-100 in x for x in nets_score_set):
#             return -1,
#         else:
#             nets_processed_score = [sum(x)/len(x) for x in nets_score_set]
#             if np.isfinite(np.asarray(nets_processed_score)).all()==False:
#                 return -1,
#             tau, p_score = stats.kendalltau(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
#             pearson, _ = stats.pearsonr(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
#             suffix = dkey + "_" + str(individual.fitness.values[0]) + "_eval_tau_" + str(tau)[:6] + "_pearson_" + str(pearson)[:6]
#             suffix = suffix.split("/")[-1]
#             tau_list.append(tau)
#             pearson_list.append(pearson)
#             plt.title(' Pearson: ' + str(pearson)[:6] + "; KendallTau: " + str(tau)[:6])
#             plt.scatter(nets_test_acc_set, nets_processed_score)
#             plt.xlabel('Accuracy')
#             plt.ylabel('Score')
#             path = logfolder + "/graphs/graph_" + suffix + ".png"
#             plt.savefig(path)
#             plt.clf()

# def graph_pearson_test(individual):
#     # dataset can be evol or test
#     # if dataset=='evol':
#     #     dict_to_iterate = evol_choice_dict
#     # else:
#     #     dict_to_iterate = test_choice_dict
#     # Go through each 
#     tau_list = []
#     pearson_list = []
#     iterate_over = test_filename_list
#     for dkey in iterate_over:
#         print("Dataset in progress: ", dkey)
#         print("Loading data-set for testing...")
#         master_dict = torch.load(dkey)
#         print("Dataset _finally_ loaded")
#         # master_dict = dict_to_iterate[dkey]
#         # It may be better to just load the data-set here.
#         nets_score_set = []
#         nets_test_acc_set = []
#         keys_to_eval = master_dict.keys()
#         func = toolbox.compile(expr=individual)
#         for net_key in tqdm(keys_to_eval):
#             per_layer_score = []
#             for layer_key in master_dict[net_key].keys():
#                 if layer_key == 'net_info':
#                     try:
#                         nets_test_acc_set.append(master_dict[net_key][layer_key]['test-accuracy'])
#                     except:
#                         nets_test_acc_set.append(master_dict[net_key][layer_key]['test_accuracy'])
#                 else:
#                     eval_out = eval_layer(func, master_dict[net_key][layer_key])
#                     eval_out = torch.sum(eval_out)/torch.numel(eval_out)
#                     if math.isnan(eval_out.item()) or math.isinf(eval_out.item()):
#                         return -1,
#                     if eval_out.item() == -100:
#                         return -1,
#                     per_layer_score.append(eval_out.item())
#             nets_score_set.append(per_layer_score)
#         if any(-100 in x for x in nets_score_set):
#             return -1,
#         else:
#             nets_processed_score = [sum(x)/len(x) for x in nets_score_set]
#             if np.isfinite(np.asarray(nets_processed_score)).all()==False:
#                 return -1,
#             tau, p_score = stats.kendalltau(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
#             pearson, _ = stats.pearsonr(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
#             suffix = dkey + "_" + str(individual.fitness.values[0]) + "_test_tau_" + str(tau)[:6] + "_pearson_" + str(pearson)[:6]
#             suffix = suffix.split("/")[-1]
#             tau_list.append(tau)
#             pearson_list.append(pearson)
#             plt.title(' Pearson: ' + str(pearson)[:6] + "; KendallTau: " + str(tau)[:6])
#             plt.scatter(nets_test_acc_set, nets_processed_score)
#             plt.xlabel('Accuracy')
#             plt.ylabel('Score')
#             path = logfolder + "/graphs/graph_" + suffix + ".png"
#             plt.savefig(path)
#             plt.clf()

# def evalOneMaxMultiSet(individual):
#     # [TODO] Test on all EVOLUTION_DATASET options, choose the minimal metric???
#     # OR
#     # [IMPLEMENTED] On each 'NUM_SAMPLING_EVAL', evaluate on a randomly sampled EVOLUTION_DATASET  
#     net_score_superset = []
#     net_val_acc_superset = []
#     net_test_acc_superset = []
#     for i in range(NUM_SAMPLING_EVAL):
#         # master_dict = evol_choice_dict[list(evol_choice_dict.keys())[random.randint(0, len(evol_choice_dict.keys())-1)]]
#         dkey = evolution_filename_list[random.randint(0, len(evolution_filename_list)-1)]
#         dd = dkey.split('/')[-1]
#         dd = dd.split('_')
#         dset_name, dset_search_space = dd[5], dd[6]
#         if dset_search_space=='NDS':
#             dset_netspace = dd[8]
#             if dd[9]=='in':
#                 dset_netspace += '_in'
#         else:
#             dset_netspace = ''
#         # Ensure SUBSAMPLE_NETS is smaller than 20 I guess?
#         master_dict = net_stat_fetcher.get_many_net_inference_data(dataset=dset_name, search_space=dset_search_space, 
#                                                                 network_space=dset_netspace, num_nets SUBSAMPLE_NETS)
#         nets_score_set = []
#         nets_test_acc_set = []
#         keys_to_eval = list(master_dict.keys())
#         # keys_to_eval = [random.randint(0, len(list(master_dict.keys()))-1)  for _ in range(SUBSAMPLE_NETS)]
#         func = toolbox.compile(expr=individual)
#         # In each subsampled space, evaluate every network
#         for net_key in keys_to_eval:
#             per_layer_score = []
#             # Go through each attribute
#             for layer_key in master_dict[net_key].keys():
#                 # If attribute is not a layer, but the network info, append that information
#                 if layer_key == 'net_info':
#                     try:
#                         nets_test_acc_set.append(master_dict[net_key][layer_key]['test-accuracy'])
#                     except:
#                         nets_test_acc_set.append(master_dict[net_key][layer_key]['test_accuracy'])
#                 # Attribute is a layer, evaluate this later and append it to per_layer_score
#                 else:
#                     eval_out = eval_layer(func, master_dict[net_key][layer_key])
#                     eval_out = torch.sum(eval_out)/torch.numel(eval_out)
#                     if math.isnan(eval_out.item()) or math.isinf(eval_out.item()):
#                         return -1,
#                     if eval_out.item() == -100:
#                         return -1,
#                     per_layer_score.append(eval_out.item())
#             # Append a list of per layer scores to the net super score
#             nets_score_set.append(per_layer_score)
#         # For this entire set, append the list of list of per layer scores
#         net_score_superset.append(nets_score_set)
#         net_test_acc_superset.append(nets_test_acc_set)
#     # net_score_superset structure is : first level index contains eval sample, each index therein contains 
#     for item in net_score_superset:
#         if any(-100 in x for x in item):
#             return -1
#     else:
#         nets_processed_score = []
#         for item in net_score_superset:
#             nets_processed_score.append([sum(x)/len(x) for x in item])
#         if np.isfinite(np.asarray(nets_processed_score)).all()==False:
#             return -1,
#         metric_scores = []
#         for idx in range(len(nets_processed_score)):
#             if METRIC=='kendalltau':
#                 metric, p_score = stats.kendalltau(np.asarray(nets_processed_score[idx]), np.asarray(net_test_acc_superset[idx]))
#             else:
#                 metric, _ = stats.pearsonr(np.asarray(nets_processed_score[idx]), np.asarray(net_test_acc_superset[idx]))
#             if math.isnan(metric) or math.isinf(metric):
#                 return -1,
#             metric_scores.append(metric)
#         metric_scores_processed = [x  if (x==-1 or x>0) else -1*x for x in metric_scores]
#         return min(metric_scores_processed),
     