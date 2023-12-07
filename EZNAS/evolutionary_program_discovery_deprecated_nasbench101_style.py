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
from tqdm import tqdm

now = datetime.now()
today = date.today()

# Metric to optimize over
METRIC = 'pearson'
# METRIC = 'kendalltau'
# Dataset
# EVOLUTION_DATASET = 'CIFAR_10'
# EVOLUTION_DATASET = 'cifar100'
# EVOLUTION_DATASET = 'ImageNet16-120'
EVOLUTION_DATASET = 'mixed'
# Test dataset
# TEST_DATASET = 'CIFAR_10'
TEST_DATASET = 'cifar100'
# TEST_DATASET = 'ImageNet16-120'
# TEST_DATASET = 'mixed'

current_time = now.strftime("%Hh_%Mm_%Ss")
if os.path.exists("logs")==False:
    os.mkdir("logs")

date = today.strftime("%b_%d_%Y")
logfolder = "logs/" + date + "_" + current_time + "_" + METRIC + "_evol_" + EVOLUTION_DATASET + "_test_" + TEST_DATASET
os.mkdir(logfolder)
if os.path.exists(logfolder + '/trees_explored/') == False:
    os.mkdir(logfolder + "/trees_explored/")
if os.path.exists(logfolder + '/graphs/') == False:
    os.mkdir(logfolder + "/graphs/")

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

fixed_ordering = [0, 1, 8, 15, 2, 9, 16, 3, 10, 17, 4, 11, 18, 5, 12, 19, 6, 13, 20, 7, 14, 21]

if EVOLUTION_DATASET != 'mixed':
    optional_evol_string = 'NASBench201_' if EVOLUTION_DATASET!='CIFAR_10' else ''
    eval_dataset_path = '/data1/yakhauri/net_stats_dataset_' + str(NUM_NETS) + '_NN_22_stats_' + optional_evol_string + str(EVOLUTION_DATASET) + '.pt'
    master_evol_dict = torch.load(eval_dataset_path)
else:
    addrs_cifar10 = '/data1/yakhauri/net_stats_dataset_500_NN_22_stats_CIFAR_10.pt'
    addrs_cifar100 = '/data1/yakhauri/net_stats_dataset_500_NN_22_stats_NASBench201_cifar100.pt'
    addrs_imagenet = '/data1/yakhauri/net_stats_dataset_500_NN_22_stats_NASBench201_ImageNet16-120.pt'
    master_evol_dict_cifar10 = torch.load(addrs_cifar10)
    master_evol_dict_cifar100 = torch.load(addrs_cifar100)
    master_evol_dict_imagenet = torch.load(addrs_imagenet)
    choice_evol_dicts = [master_evol_dict_cifar10, master_evol_dict_cifar100, master_evol_dict_imagenet]

if TEST_DATASET != 'mixed':
    optional_test_string = 'NASBench201_' if TEST_DATASET!='CIFAR_10' else ''
    test_dataset_path = '/data1/yakhauri/net_stats_dataset_' + str(NUM_NETS) + '_NN_22_stats_' + optional_test_string + str(TEST_DATASET) + '_test.pt'
    master_test_dict = torch.load(test_dataset_path)
else:
    addrs_cifar10 = '/data1/yakhauri/net_stats_dataset_500_NN_22_stats_CIFAR_10_test.pt'
    addrs_cifar100 = '/data1/yakhauri/net_stats_dataset_500_NN_22_stats_NASBench201_cifar100_test.pt'
    addrs_imagenet = '/data1/yakhauri/net_stats_dataset_500_NN_22_stats_NASBench201_ImageNet16-120_test.pt'
    master_test_dict_cifar10 = torch.load(addrs_cifar10)
    master_test_dict_cifar100 = torch.load(addrs_cifar100)
    master_test_dict_imagenet = torch.load(addrs_imagenet)
    choice_test_dicts = [master_test_dict_cifar10, master_test_dict_cifar100, master_test_dict_imagenet]

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

def graph_pearson(individual, suffix, dataset='evol'):
    # dataset can be evol or test
    if dataset=='evol':
        if EVOLUTION_DATASET!='mixed':
            master_dict = master_evol_dict
        else:
            master_dict = choice_evol_dicts[random.randint(0,2)]
    else:
        if TEST_DATASET!='mixed':
            master_dict = master_test_dict
        else:
            master_dict = choice_test_dicts[random.randint(0,2)]
    nets_score_set = []
    nets_test_acc_set = []
    keys_to_eval = [idx for idx in range(NUM_NETS)]
    keys_to_eval = keys_to_eval[:38] + keys_to_eval[39:]
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
    net_score_superset = []
    net_val_acc_superset = []
    net_test_acc_superset = []
    for i in range(NUM_SAMPLING_EVAL):
        if EVOLUTION_DATASET!='mixed':
            master_dict = master_evol_dict
        else:
            master_dict = choice_evol_dicts[random.randint(0,2)]
        nets_score_set = []
        nets_test_acc_set = []
        keys_to_eval = [random.randint(0, NUM_NETS-1)  for _ in range(SUBSAMPLE_NETS)]
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
        # 
        metric_scores_processed = [x  if (x==-1 or x>0) else -1*x for x in metric_scores]
        return min(metric_scores_processed),
     
def evalOneMax_test_evol(individual):
    # Take the test data-set of the same task it was evolved on
    if EVOLUTION_DATASET!='mixed':
        master_dict = master_evol_dict
    else:
        master_dict = choice_evol_dicts[random.randint(0,2)]
    nets_score_set = []
    nets_test_acc_set = []
    keys_to_eval = [idx for idx in range(NUM_NETS)]
    keys_to_eval = keys_to_eval[:38] + keys_to_eval[39:]
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
        if(len(set(nets_processed_score))<=1):
            return -1,
        if METRIC=='kendalltau':
            metric, p_score = stats.kendalltau(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
        else:
            metric, _ = stats.pearsonr(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
        if np.isfinite(np.asarray(nets_processed_score)).all()==False:
            return -1,
        if metric != -1 and metric < 0:
            return -1*metric,
        return metric,
    
def evalOneMax_test_test(individual):
    # Take the test data-set of the same task it was evolved on
    if EVOLUTION_DATASET!='mixed':
        master_dict = master_test_dict
    else:
        master_dict = choice_test_dicts[random.randint(0,2)]
    nets_score_set = []
    nets_test_acc_set = []
    keys_to_eval = [idx for idx in range(NUM_NETS)]
    keys_to_eval = keys_to_eval[:38] + keys_to_eval[39:]
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
                assert len(list(master_dict[net_key][layer_key].keys()))==22, str(net_key) + "_" + str(layer_key)
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
        if(len(set(nets_processed_score))<=1):
            return -1,
        if METRIC=='kendalltau':
            metric, p_score = stats.kendalltau(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
        else:
            metric, _ = stats.pearsonr(np.asarray(nets_processed_score), np.asarray(nets_test_acc_set))
        if np.isfinite(np.asarray(nets_processed_score)).all()==False:
            return -1,
        if metric != -1 and metric < 0:
            return -1*metric,
        return metric,


toolbox.register("evaluate", evalOneMaxMultiSet)
toolbox.register("select", tools.selTournament, tournsize=TOURSIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=MIN_TREE_DEPTH, max_=MAX_TREE_DEPTH)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG, filename=logfolder + "/logfile.log", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
    if nproc > 1:
        pool = multiprocessing.Pool(processes=nproc)
        toolbox.register("map", pool.map)

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
        fits = toolbox.map(toolbox.evaluate, offspring)
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
    
    top30 = tools.selBest(population, k=30)
    top30_fitness = [x.fitness.values[0] for x in top30]
    indexes = [top30_fitness.index(x) for x in set(top30_fitness)]
    topx_unique_list = [x for idx, x in enumerate(top30) if idx in indexes]
    if len(topx_unique_list)>10:
        topx_unique_list = topx_unique_list[:10]
    for i in topx_unique_list:
        indv_fitness = i.fitness.values[0]
        try:
            print("Train fitness: ", indv_fitness, " Tree Height: ", str(i.height))
            metric = evalOneMax_test_evol(i)[0]
            graph_computation(i, suffix="_" + str(EVOLUTION_DATASET) + "_topx_search_" + METRIC + "_" + str(indv_fitness) + "_gen_" + str(gen) + "_true_" + METRIC + "_" + str(metric))
            graph_pearson(i, suffix="_" + str(EVOLUTION_DATASET) + "_topx_search_" + METRIC + "_" + str(indv_fitness) + "_gen_" + str(gen) + "_true_" + METRIC + "_" + str(metric), dataset='evol')
            logging.info(str(i.height)  +  ", _" + str(EVOLUTION_DATASET) + ", _topx_search_" + METRIC + "_" + str(indv_fitness) + "_gen_" + str(gen) + "_true_" + METRIC + "_" + str(metric))
            print(str(i.height)  +  ", _" + str(EVOLUTION_DATASET) + ", _topx_search_" + METRIC + "_" + str(indv_fitness) + "_gen_" + str(gen) + "_true_" + METRIC + "_" + str(metric))
            print("*********************")
            logging.info("*********************")
        except Exception as e:
            print("Exception: ", e)
            print(indv_fitness)
            print(i, " Failed")
            logging.info(i)
            logging.info("Failed")

    for i in topx_unique_list:
        indv_fitness = i.fitness.values[0]
        try:
            print(indv_fitness)
            metric = evalOneMax_test_test(i)[0]
            graph_computation(i, suffix="_" + str(EVOLUTION_DATASET) + "_topx_search_" + METRIC + "_" + str(indv_fitness) + "_gen_" + str(gen) + "_" + str(TEST_DATASET)  + "_true_" + METRIC + "_" + str(metric))
            graph_pearson(i, suffix="_" + str(EVOLUTION_DATASET) + "_topx_search_" + METRIC + "_" + str(indv_fitness) + "_gen_" + str(gen) + "_" + str(TEST_DATASET)  + "_true_" + METRIC + "_" + str(metric), dataset='test')
            logging.info(str(i.height)  +  ", _" + str(EVOLUTION_DATASET) + "_topx_search_" + METRIC + "_" + str(indv_fitness) + "_gen_" + str(gen) + "_" + str(TEST_DATASET)  + "_true_" + METRIC + "_" + str(metric))
            print(str(i.height)  +  ", _" + str(EVOLUTION_DATASET) + "_topx_search_" + METRIC + "_" + str(indv_fitness) + "_gen_" + str(gen) + "_" + str(TEST_DATASET)  + "_true_" + METRIC + "_" + str(metric))
            print("*********************")
            logging.info("*********************")
        except Exception as e:
            print("Exception: ", e)
            print(indv_fitness)
            print(i, " Failed")
            logging.info(i)
            logging.info("Failed")

    if nproc > 1:
        pool.close()
        

##################### BACKUP CODE #####################


# required_order = ['wt', 'wtgraddata', 'wtgradnoise', 'wtgradperturb',
# 'inpactdata', 'inpactnoise', 'inpactperturb', 'inpactgraddata', 'inpactgradnoise', 'inpactgradperturb',
# 'preactdata', 'preactnoise', 'preactperturb', 'preactgraddata', 'preactgradnoise', 'preactgradperturb',
# 'actdata', 'actnoise', 'actperturb', 'actgraddata', 'actgradnoise', 'actgradperturb']

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

