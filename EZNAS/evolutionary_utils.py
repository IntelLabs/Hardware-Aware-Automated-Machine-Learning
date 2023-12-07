from func_defs import *
import random
from operator import attrgetter
import yaml


with open("evol_config.yaml", "r") as f:
    config = yaml.load(f)

LEN_IND_ADDR = config['LEN_IND_ADDR']
NUM_MATH_OPS = config['NUM_MATH_OPS']
NEW_INST_MIN_LEN = config['NEW_INST_MIN_LEN']
NEW_INST_MAX_LEN = config['NEW_INST_MAX_LEN']

def reduce_tree_repr(individual):
    if len(individual)<=4:
        return individual
    tp = type(individual)
    # for idx in range(len(individual)//4 - 1, -1, -1):
    #     instructions_inverted.append(individual[idx*len(individual)//4:idx*len(individual)//4 + 4])
    instructions = [individual[i:i + 4] for i in range(0, len(individual), 4)]
    # always_preserve = instructions[-1]
    # instructions_inverted = instructions[::-1]
    # Instructions are now in inverted order. 
    # Mask will identify which instructions are relevant
    # Minimal amortization: 
    # If the write address of any instruction is used ANYWHERE in the rest of the algorithm, it is relevant.
    # Do 4 passes of this clean up
    for _ in range(4):
        mask = [0]*len(instructions)        
        for idx_inst, inst in enumerate(instructions):
            for idx_checker, checker in enumerate(instructions):
                write_of_inst = inst[0]
                tva = write_of_inst==checker[2]
                tvb = write_of_inst==checker[3]
                tvargs = tva or tvb
                if idx_checker > idx_inst and tvargs:
                    mask[idx_inst] = 1
                    mask[idx_checker] = 1
        mask[-1] = 1
        # print(mask)
        new_individual = []
        for idx, inst in enumerate(instructions):
            if mask[idx]==1:
                new_individual.extend(inst)
        instructions = [new_individual[i:i + 4] for i in range(0, len(new_individual), 4)]
    # print(len(new_individual))
    return tp(new_individual)

'''
def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.
    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    chosen = []
    for i in xrange(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen
'''

def selRandom(individuals, k):
    return [random.choice(individuals) for i in range(k)]

def selTournamentSansPerfDuplicates(individuals, k, tournsize, fit_attr="fitness"):
    chosen = []
    perfs = []
    rand_init_count = 0
    while len(chosen) < k:
        aspirants = selRandom(individuals, tournsize)
        sampled_aspirant = max(aspirants, key=attrgetter(fit_attr))
        tp = type(sampled_aspirant)
        if sampled_aspirant.fitness in perfs:
            sampled = tp(load_individuals_test(1)[0])
            print(len(sampled))
            if len(sampled)>0:
                chosen.append(sampled)
                rand_init_count+=1
            else:
                pass
        else:
            chosen.append(sampled_aspirant)
            perfs.append(sampled_aspirant.fitness)
    print("Initialized " + str(rand_init_count) + " new individuals")
    # print(chosen)
    return chosen

def selTournamentSansDuplicates(individuals, k, tournsize, fit_attr="fitness"):
    chosen = []
    # for i in range(k):
    #     aspirants = selRandom(individuals, tournsize)
    #     chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    while len(chosen) < k:
        aspirants = selRandom(individuals, tournsize)
        sampled_aspirant = max(aspirants, key=attrgetter(fit_attr))
        tp = type(sampled_aspirant)
        if sampled_aspirant in chosen:
            # print(load_individuals_test(1)[0])
            chosen.append(tp(load_individuals_test(1)[0]))
        else:
            chosen.append(sampled_aspirant)
    # print(chosen)
    # exit(0)
    return chosen

def load_individuals(creator, n):
    print("Requested " + str(n) + " randomly initialized Individuals")
    individuals = []
    # Creation rule:
    # Choose operator being used
    # Choose a read type (weight or activation)
    # WRITE RULES
    # if op preserve dim, 
    #   set write location to any dynamic address in weight or activation space
    # if op not preserve dim,
    #   set write location range to the scalar addresses
    # READ RULES
    # if op argmatch dim
    #   choose any location, ensure second is from same location
    # if op not argmatch dim
    #   doesn't matter, do not deal with this condition
    while(len(individuals)<n):
        # if(len(individuals)%100):
        # print(len(individuals), end=', ', flush=True)
        wt_addr_written_to = []
        act_addr_written_to = []
        scalar_addr_written_to = []
        individual = []
        # for inst_len in range(random.randint(4, 8)):
        while len(reduce_tree_repr(individual)) < random.randint(NEW_INST_MIN_LEN, NEW_INST_MAX_LEN):
            math_op = random.randint(0, NUM_MATH_OPS-1)
            # [TODO] This requires that rangemix is True
            read_type = random.randint(0, 1)
            if read_type == 0:
                read_range = wt_addr_range
            else:
                read_range = activation_superrange
            # It is a scalar output
            if op_preserve_dim[math_op]==0:
                write_addr = scalar_addr_range[random.randint(0, len(scalar_addr_range)-1)]
                pre_written_set_choice = scalar_addr_written_to
                scalar_addr_written_to.append(write_addr)
            else:
                if read_type == 0:
                    wt_write_addrs = wt_addr_range[-1*LEN_IND_ADDR:]
                    if(len(wt_write_addrs)==0):
                        continue
                    write_addr = wt_write_addrs[random.randint(0, len(wt_write_addrs)-1)]
                    pre_written_set_choice = wt_addr_written_to
                    wt_addr_written_to.append(wt_addr_written_to)
                else:
                    act_write_addrs = activation_superrange[-3*LEN_IND_ADDR:]
                    if(len(act_write_addrs)==0):
                        continue
                    write_addr = act_write_addrs[random.randint(0, len(act_write_addrs)-1)]
                    pre_written_set_choice = act_addr_written_to
                    act_addr_written_to.append(write_addr)
            # Choose anything from static addresses or written addresses
            # print(set(read_range) & set(pre_written_set_choice))
            # read_addr_1 = list(set(read_range) & set(pre_written_set_choice))
            if len(read_range) > 4 + LEN_IND_ADDR:
                read_addr_1_list = read_range[:-3*LEN_IND_ADDR] + pre_written_set_choice
            else:
                read_addr_1_list = read_range[:-1*LEN_IND_ADDR] + pre_written_set_choice
            if(len(read_addr_1_list)==0):
                continue
            read_addr_1 = read_addr_1_list[random.randint(0, len(read_addr_1_list)-1)]
            # read_addr_2 = list(set(read_range) & set(pre_written_set_choice))
            if len(read_range) > 4 + LEN_IND_ADDR:
                read_addr_2_list = read_range[:-3*LEN_IND_ADDR] + pre_written_set_choice
            else:
                read_addr_2_list = read_range[:-1*LEN_IND_ADDR] + pre_written_set_choice
            if(len(read_addr_2_list)==0):
                continue
            read_addr_2 = read_addr_2_list[random.randint(0, len(read_addr_2_list)-1)]
            # print(write_addr, math_op, read_addr_1, read_addr_2)
            if(isinstance(write_addr, int)==False):
                continue
            if(isinstance(math_op, int)==False):
                continue
            if(isinstance(read_addr_1, int)==False):
                continue
            if(isinstance(read_addr_2, int)==False):
                continue
            if(write_addr==read_addr_1):
                continue
            if(write_addr==read_addr_2):
                continue
            individual.append(write_addr)
            individual.append(math_op)
            individual.append(read_addr_1)
            individual.append(read_addr_2)
        # if len(reduce_tree_repr(individual))>0:
        individuals.append(creator(reduce_tree_repr(individual)))
    return individuals

def load_individuals_test(n):
    # print("Requested " + str(n) + " randomly initialized Individuals")
    individuals = []
    # Creation rule:
    # Choose operator being used
    # Choose a read type (weight or activation)
    # WRITE RULES
    # if op preserve dim, 
    #   set write location to any dynamic address in weight or activation space
    # if op not preserve dim,
    #   set write location range to the scalar addresses
    # READ RULES
    # if op argmatch dim
    #   choose any location, ensure second is from same location
    # if op not argmatch dim
    #   doesn't matter, do not deal with this condition
    while(len(individuals)<n):
        wt_addr_written_to = []
        act_addr_written_to = []
        scalar_addr_written_to = []
        individual = []
        # for inst_len in range(random.randint(3, 8)):
        while len(reduce_tree_repr(individual)) < random.randint(NEW_INST_MIN_LEN, NEW_INST_MAX_LEN):
            math_op = random.randint(0, NUM_MATH_OPS-1)
            # [TODO] This requires that rangemix is True
            read_type = random.randint(0, 1)
            if read_type == 0:
                read_range = wt_addr_range
            else:
                read_range = activation_superrange
            # It is a scalar output
            if op_preserve_dim[math_op]==0:
                write_addr = scalar_addr_range[random.randint(0, len(scalar_addr_range)-1)]
                pre_written_set_choice = scalar_addr_written_to
                scalar_addr_written_to.append(write_addr)
            else:
                if read_type == 0:
                    wt_write_addrs = wt_addr_range[-1*LEN_IND_ADDR:]
                    if(len(wt_write_addrs)==0):
                        continue
                    write_addr = wt_write_addrs[random.randint(0, len(wt_write_addrs)-1)]
                    pre_written_set_choice = wt_addr_written_to
                    wt_addr_written_to.append(wt_addr_written_to)
                else:
                    act_write_addrs = activation_superrange[-3*LEN_IND_ADDR:]
                    if(len(act_write_addrs)==0):
                        continue
                    write_addr = act_write_addrs[random.randint(0, len(act_write_addrs)-1)]
                    pre_written_set_choice = act_addr_written_to
                    act_addr_written_to.append(write_addr)
            # Choose anything from static addresses or written addresses
            # print(set(read_range) & set(pre_written_set_choice))
            # read_addr_1 = list(set(read_range) & set(pre_written_set_choice))
            if len(read_range) > 4 + LEN_IND_ADDR:
                read_addr_1_list = read_range[:-3*LEN_IND_ADDR] + pre_written_set_choice
            else:
                read_addr_1_list = read_range[:-1*LEN_IND_ADDR] + pre_written_set_choice
            if(len(read_addr_1_list)==0):
                continue
            read_addr_1 = read_addr_1_list[random.randint(0, len(read_addr_1_list)-1)]
            # read_addr_2 = list(set(read_range) & set(pre_written_set_choice))
            if len(read_range) > 4 + LEN_IND_ADDR:
                read_addr_2_list = read_range[:-3*LEN_IND_ADDR] + pre_written_set_choice
            else:
                read_addr_2_list = read_range[:-1*LEN_IND_ADDR] + pre_written_set_choice
            if(len(read_addr_2_list)==0):
                continue
            read_addr_2 = read_addr_2_list[random.randint(0, len(read_addr_2_list)-1)]
            # print(write_addr, math_op, read_addr_1, read_addr_2)
            if(isinstance(write_addr, int)==False):
                continue
            if(isinstance(math_op, int)==False):
                continue
            if(isinstance(read_addr_1, int)==False):
                continue
            if(isinstance(read_addr_2, int)==False):
                continue
            if(write_addr==read_addr_1):
                continue
            if(write_addr==read_addr_2):
                continue
            individual.append(write_addr)
            individual.append(math_op)
            individual.append(read_addr_1)
            individual.append(read_addr_2)
        # if len(reduce_tree_repr(individual))>0:
        individuals.append(reduce_tree_repr(individual))
    return individuals
