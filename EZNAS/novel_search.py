from deap import creator, base, tools, algorithms, gp
from deap.tools import History
import numpy as np
import torch.multiprocessing
from absl import app
import scipy.stats as stats
from datetime import datetime, date
import matplotlib.pyplot as plt
from itertools import repeat
from tqdm import tqdm
from pprint import pprint
from pprint import pformat
import torch, pickle, toolz, warnings, networkx, matplotlib, shutil, os, logging, random, operator, time, multiprocessing, argparse, yaml, itertools, math
from torchsummary import summary
import torch.nn as nn

##### CLEANED IMPORTS #####
from nats_bench import create
import xautodl
from xautodl.models import get_cell_based_tiny_net
from models import get_cell_based_tiny_net
from nasspace import *
from dataset_utils import *
from nas_201_api import NASBench201API as API
from nasbench import api
from nas_101_api.model import Network
from gp_func_defs import *
from evol_utilities import *

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
now = datetime.now()
today = date.today()

device = "cuda:0"

args = get_args()
seed = args.seed

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


toolbox = base.Toolbox()
pset = generate_pset(toolbox)

NB201_FPATH = "/home/ya255/projects/Hardware-Aware-Automated-Machine-Learning/EZNAS/dataset_generator/NAS-Bench-201-v1_1-096897.pth"
NATSB_FPATH = "/home/ya255/projects/Hardware-Aware-Automated-Machine-Learning/EZNAS/dataset_generator/NATS-sss-v1_0-50262.pickle.pbz2"

api_dict = {"NATS-sss": create(
            NATSB_FPATH,
            "sss",
            fast_mode=False,
            verbose=True),
            "NASBench-201": API(NASBENCH_TFRECORD, verbose=False),
            "NDS": None,
            }


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=1)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluateIndividual)

operations = list(pset.primitives.items())[0][1]
terminals = list(pset.terminals.items())[0][1]


def get_firstgen(num_individuals=100):
    search_spaces = ['NASBench201', 'nds_amoeba', 'NATSBench']
    num_networks = 10
    args.api_loc = NASBENCH_TFRECORD
    nats_sss_api = api_dict['NATS-sss']
    nb201_api = api_dict['NASBench-201']
    operations = list(pset.primitives.items())[0][1]
    terminals = list(pset.terminals.items())[0][1]
    # Here, we generate the first generation in a while loop
    # criteria is that each individual should return a valid fitness (> -1)
    newgen = []
    while len(newgen) < num_individuals:
        ind = toolbox.individual()
        descriptor = (
            ind, search_spaces, num_networks,
            args, nats_sss_api, nb201_api, operations, terminals, device
            )
        # evaluate individual using toolbox
        fitness = toolbox.evaluate(descriptor)
        if fitness > -1:
            ind.fitness.values = fitness,
            newgen.append(ind)
    return newgen


if __name__ == "__main__":
    population = get_firstgen(200)
    best_individuals = []
    NUM_GENERATIONS = 100
    for gen in range(NUM_GENERATIONS):  # Replace NUM_GENERATIONS with your number of generations
        valid_offspring = []
        while len(valid_offspring) < 200:
            # Generate offspring
            offspring = algorithms.varOr(population, toolbox, lambda_=200, cxpb=0.5, mutpb=0.5)

            # Create descriptors and evaluate each individual in the offspring
            for ind in offspring:
                descriptor = (
                    ind, search_spaces, num_networks, args, nats_sss_api, nb201_api, operations, terminals, device
                )
                fitness = toolbox.evaluate(descriptor)
                if fitness > -1:
                    ind.fitness.values = (fitness,)
                    valid_offspring.append(ind)
                    if len(valid_offspring) == 200:
                        break

        # Update population with valid offspring
        population[:] = valid_offspring
        
        # Track best 3 individuals
        best_individuals.extend(sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)[:3])

        print("Best individuals: ", best_individuals)
        print("Best 10 fitness: ", sorted([ind.fitness.values[0] for ind in population], reverse=True)[:10])
        print("Average and standard deviation of fitness: {}, {}".format(np.mean([ind.fitness.values[0] for ind in population]), np.std([ind.fitness.values[0] for ind in population])))

# # def evaluate_final(individuals):
    
# def get_nextgen(prevgen):
#     # Here, we take the previous generation with their assigned fitnesses
#     # and generate the next generation. Each individual should return a valid fitness (> -1)
#     individuals = None
#     return individuals

# def evaluate_individual(individual):
#     # Here, we evaluate the individual
#     # We evaluate it on 'NUMNETS' on each of the 'SOURCESPACE' search spaces
#     # We can then return the minimum fitness.
#     # If evaluation fails, return -1
#     fitness = None
#     return fitness

    