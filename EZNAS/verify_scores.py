import random
import numpy as np
import torch.multiprocessing
from absl import app
import scipy.stats as stats
import yaml
import itertools
from datetime import datetime, date
import os
import matplotlib.pyplot as plt
import matplotlib
from itertools import repeat
from tqdm import tqdm
import toolz
import numpy as np
import random
import torch.nn as nn
import torch
from deap import creator, base, tools, algorithms, gp
from deap.tools import History
# Secondary Imports
from nasbench import api
from nats_bench import create
import xautodl
from xautodl.models import get_cell_based_tiny_net
from nas_201_api import NASBench201API as API
# Our imports
from models import get_cell_based_tiny_net
from evol_utilities import *
from gp_func_defs import *
from nasspace import *
from dataset_utils import *

# Base setup
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
last_layer_alg = False
last_layer_set = [0, -1]
eval_on_sets = ["darts"]

nets_score_set = {x: [] for x in eval_on_sets}
nets_test_acc_set = {x: [] for x in eval_on_sets}

if args.nds_space == "":
    NUM_NETS = 15625
elif args.search_space == "NATSBench":
    NUM_NETS = 32768
else:
    NUM_NETS = 4000

nnetz_sam = list(range(NUM_NETS))[::-1]
if args.n_samples != -1:
    nnetz_sam = list(random.sample(nnetz_sam, args.n_samples))

## Toolbox
toolbox = base.Toolbox()
pset = generate_pset(toolbox)
args.nasspace = args.nds_space
dataset = args.dataset
search_space = args.search_space
batch_size = args.batch_size
nds_space = args.nds_space if search_space == "NDS" else ""

## Assertion checks
if dataset == "ImageNet":
    nds_space += "_in"
if search_space == "NDS":
    assert dataset != "cifar100" or dataset != "ImageNet16-120"

# Load dataset
train_loader = get_eznas_trainloader(batch_size, dataset)

# Load network sampler (NDS) and API
network_sampler = get_search_space(args)
if search_space == "NATSBench":
    api = create(
        os.environ['PROJ_HOME'] + "dataset_generator/NATS-sss-v1_0-50262.pickle.pbz2",
        "sss",
        fast_mode=False,
        verbose=True,
    )
elif search_space == "NASBench201":
    NASBENCH_TFRECORD = os.environ['PROJ_HOME'] + "dataset_generator/NAS-Bench-201-v1_1-096897.pth"
    api = API(NASBENCH_TFRECORD, verbose=False)
else:
    api = None

# Register functions
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=1)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
operations = list(pset.primitives.items())[0][1]
terminals = list(pset.terminals.items())[0][1]

score_set = []

for evset in eval_on_sets:
    evaluate_networks(
        nnetz_sam,
        search_space,
        network_sampler,
        api,
        dataset,
        train_loader,
        toolbox,
        [evset],
        last_layer_alg,
        last_layer_set,
        operations,
        terminals,
        device,
        nets_score_set,
        nets_test_acc_set,
    )
    nets_processed_score = [sum(x) / len(x) for x in nets_score_set[evset]]
    tau, p_score = stats.kendalltau(
        np.asarray(nets_processed_score), np.asarray(nets_test_acc_set[evset])
    )
    pearson, _ = stats.pearsonr(
        np.asarray(nets_processed_score), np.asarray(nets_test_acc_set[evset])
    )
    spearman = stats.spearmanr(
        np.asarray(nets_processed_score),
        np.asarray(nets_test_acc_set[evset]),
        nan_policy="omit",
    ).correlation
    print(f'Base Program [{evset}] Perf:\t Tau: {tau:.3f}, Pearson: {pearson:.3f}, Spearman: {spearman:.3f}')

if not os.path.exists("./reproduce/"):
    os.makedirs("./reproduce/")

with open(f'./reproduce/all_tests.csv', "a") as f:
    f.write(
        f"{args.seed},{args.n_samples},{args.batch_size},{args.dataset},{args.search_space},{args.nds_space},{tau},{pearson},{spearman}\n"
    )