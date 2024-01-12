from deap import creator, base, tools, algorithms, gp
import itertools
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import torch.nn as nn
import torch
# Our imports
from models import get_cell_based_tiny_net
from gp_func_defs import *

dataset_order = [
    "wt", "wtgraddata", "inpactdata", "inpactgraddata", "preactdata", "preactgraddata", "actdata", "actgraddata", 
    "wtgradnoise", "inpactnoise", "inpactgradnoise", "preactnoise", "preactgradnoise", "actnoise", "actgradnoise", 
    "wtgradperturb", "inpactperturb", "inpactgradperturb", "preactperturb", "preactgradperturb", "actperturb", "actgradperturb", 
    ]

fixed_ordering = [
    2, 3, 10, 17, 0, 8, 15, 1, 9, 16, 4, 11, 18, 5, 12, 19, 6, 13, 20, 7, 14, 21, 
    ]

STATIC_ADDRS = 22

class Hook:
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output


def initialize_module(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)



def populate(runtime_addrs_space, net_stats):
    net_addr_key = list(net_stats.keys())
    map_list = []
    for i in range(STATIC_ADDRS):
        try:
            runtime_addrs_space[i] = net_stats[net_addr_key[fixed_ordering[i - 1]]]
            map_list.append(net_stats[net_addr_key[fixed_ordering[i - 1]]])
        except Exception as e:
            # import pdb; pdb.set_trace()
            print("Populate function error: ", e)
            raise ValueError("Populate function error: ", e)
    return runtime_addrs_space, map_list


def eval_layer(func, net_stats, device):
    runtime_addrs_space = {}
    runtime_addrs_space, map_list = populate(runtime_addrs_space, net_stats)
    try:
        output = func(*map_list)
    except Exception:
        return torch.Tensor([-100]).to(device)
    return output


def generate_pset(toolbox):
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(torch.Tensor, 22), torch.Tensor)
    pset.addPrimitive(OP0, [torch.Tensor, torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP1, [torch.Tensor, torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP2, [torch.Tensor, torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP3, [torch.Tensor, torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP4, [torch.Tensor, torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP5, [torch.Tensor, torch.Tensor], torch.Tensor)
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
    pset.addPrimitive(OP18, [torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP19, [torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP20, [torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP21, [torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP22, [torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP23, [torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP24, [torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP25, [torch.Tensor, torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP26, [torch.Tensor, torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP27, [torch.Tensor, torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP28, [torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP29, [torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP30, [torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP31, [torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP32, [torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP33, [torch.Tensor], torch.Tensor)
    pset.addPrimitive(OP34, [torch.Tensor], torch.Tensor)
    return pset


def get_individual(dataset, toolbox, operations, terminals):
    if dataset == "topprog":
        individual = [0, 0, 0]
        while len(individual) > 2:
            individual = toolbox.individual()
        individual[-1].value = "ARG12"
        individual[0] = operations[9]
        individual.insert(1, operations[8])
        individual.insert(2, operations[16])
    if dataset == "darts":
        individual = [0, 0, 0]
        while len(individual) > 2:
            individual = toolbox.individual()
        individual[-1].value = "ARG8"
        individual[0] = operations[16]
        individual.insert(1, operations[27])
        individual.insert(2, operations[18])
        individual.insert(3, operations[13])
    if dataset == "enas":
        individual = [0, 0, 0]
        while len(individual) > 2:
            individual = toolbox.individual()
        individual[-1].value = "ARG19"
        individual[0] = operations[15]
        individual.insert(1, operations[0])
        individual.insert(2, operations[16])
        individual.insert(3, terminals[19])
        individual.insert(4, operations[10])
        individual.insert(5, operations[16])
        individual.insert(6, operations[6])
        individual.insert(7, operations[27])
        individual.insert(8, operations[6])
        individual.insert(9, operations[27])
    if dataset == "pnas":
        individual = [0, 0]
        while len(individual) != 3:
            individual = toolbox.individual()
        individual[-1].value = "ARG20"
        individual[-2].value = "ARG0"
        individual[0] = operations[24]
        individual.insert(1, operations[11])
        individual.insert(2, operations[11])
        individual.insert(3, operations[11])
        individual.insert(5, operations[30])
    if dataset == "nasnet":
        individual = [0, 0, 0]
        while len(individual) > 2:
            individual = toolbox.individual()
        individual[-1].value = "ARG6"
        individual[0] = operations[16]
        individual.insert(1, operations[31])
        individual.insert(2, operations[6])
        individual.insert(3, operations[0])
        individual.insert(4, operations[6])
        individual.insert(5, terminals[19])
    return individual


def process_layer_data(i, state_dict, hookF, hookB, flat_layer_module_list, process_noise=False, perturb=0):
    "layer" + str(i)
    if not process_noise and perturb == 0:
        try:
            state_dict["layer" + str(i)]["inpactdata"] = (
                hookF[3 * i].input[0].clone().detach()
            )
            state_dict["layer" + str(i)]["inpactgraddata"] = (
                hookB[3 * i].input[0].clone().detach()
            )

            state_dict["layer" + str(i)]["wt"] = (
                flat_layer_module_list[3 * i + 1].weight.clone().detach()
            )
            state_dict["layer" + str(i)]["wtgraddata"] = (
                flat_layer_module_list[3 * i + 1].weight.grad.clone().detach()
            )
            state_dict["layer" + str(i)]["preactdata"] = (
                hookF[3 * i + 1].input[0].clone().detach()
            )
            state_dict["layer" + str(i)]["preactgraddata"] = (
                hookB[3 * i + 1].input[0].clone().detach()
            )

            state_dict["layer" + str(i)]["actdata"] = (
                hookF[3 * i + 2].output.clone().detach()
            )
            state_dict["layer" + str(i)]["actgraddata"] = (
                hookB[3 * i + 2].output[0].clone().detach()
            )
        except Exception as e:
            # import pdb; pdb.set_trace()
            print("Failing!: ", e)
            raise ValueError("Missing input vec")
    elif process_noise and perturb == 0:
        try:
            state_dict["layer" + str(i)]["inpactnoise"] = (
                hookF[3 * i].input[0].clone().detach()
            )
            state_dict["layer" + str(i)]["inpactgradnoise"] = (
                hookB[3 * i].input[0].clone().detach()
            )
            state_dict["layer" + str(i)]["wtgradnoise"] = (
                flat_layer_module_list[3 * i + 1].weight.grad.clone().detach()
            )
            state_dict["layer" + str(i)]["preactnoise"] = (
                hookF[3 * i + 1].input[0].clone().detach()
            )
            state_dict["layer" + str(i)]["preactgradnoise"] = (
                hookB[3 * i + 1].input[0].clone().detach()
            )
            state_dict["layer" + str(i)]["actnoise"] = (
                hookF[3 * i + 2].output.clone().detach()
            )
            state_dict["layer" + str(i)]["actgradnoise"] = (
                hookB[3 * i + 2].output[0].clone().detach()
            )
        except Exception as e:
            # import pdb; pdb.set_trace()
            print("Failing!: ", e)
            raise ValueError("Missing input vec")
    elif not process_noise and perturb:
        try:
            state_dict["layer" + str(i)]["inpactperturb"] = (
                hookF[3 * i].input[0].clone().detach()
                - state_dict["layer" + str(i)]["inpactdata"]
            )
            state_dict["layer" + str(i)]["inpactgradperturb"] = (
                hookB[3 * i].input[0].clone().detach()
                - state_dict["layer" + str(i)]["inpactgraddata"]
            )
            state_dict["layer" + str(i)]["wtgradperturb"] = (
                flat_layer_module_list[3 * i + 1].weight.grad.clone().detach()
                - state_dict["layer" + str(i)]["wtgraddata"]
            )
            state_dict["layer" + str(i)]["preactperturb"] = (
                hookF[3 * i + 1].input[0].clone().detach()
                - state_dict["layer" + str(i)]["preactdata"]
            )
            state_dict["layer" + str(i)]["preactgradperturb"] = (
                hookB[3 * i + 1].input[0].clone().detach()
                - state_dict["layer" + str(i)]["preactgraddata"]
            )
            state_dict["layer" + str(i)]["actperturb"] = (
                hookF[3 * i + 2].output.clone().detach()
                - state_dict["layer" + str(i)]["actdata"]
            )
            state_dict["layer" + str(i)]["actgradperturb"] = (
                hookB[3 * i + 2].output[0].clone().detach()
                - state_dict["layer" + str(i)]["actgraddata"]
            )
        except Exception as e:
            # import pdb; pdb.set_trace()
            print("Failing!: ", e)
            raise ValueError("Missing input vec")
    else:
        pass



def process_network_output(net, data_sample, state_dict, hookF, hookB, flat_layer_module_list, device, process_noise=False, perturb=0):
    # data_sample requires grad True
    data_sample.requires_grad = True
    if process_noise:
        data_sample = torch.randn(data_sample.shape)
        data_sample = torch.autograd.Variable(data_sample, requires_grad=True)
    if perturb:
        data_sample = data_sample + perturb ** 0.5 * torch.randn(data_sample.shape)
    data_sample = data_sample.to(device)
    out, _ = net(data_sample)
    out.backward(torch.ones_like(out))
    for i in range(len(flat_layer_module_list) // 3):
        try:
            process_layer_data(i, state_dict, hookF, hookB, flat_layer_module_list, process_noise, perturb)
        except Exception as e:
            state_dict["layer" + str(i)] = {}
            # raise ValueError("Error in process_network_output: ", e)




def evaluateIndividual(descriptor):
    individual, search_spaces, num_networks,  \
    args, nats_sss_api, nb201_api, \
    operations, terminals, device = descriptor
    space_to_corr = {}
    try:
        for nas_space in search_spaces:
            args.nasspace = nas_space
            network_sampler = get_search_space(args)
            if nas_space != "NATSBench":
                tot_networks = network_sampler.__len__()
            else:
                tot_networks = 32800
            net_ids = random.sample(list(range(tot_networks)), num_networks)
            score_list = []
            acc_list = []
            for i_reg in net_ids:
                state_dict = {}
                results, net = initialize_network(nas_space, i_reg, network_sampler, nats_sss_api, nb201_api)
                net = net.to(device)
                state_dict["net_info"] = results
                for m in net.modules():
                    initialize_module(m)
                data_sample = torch.randn(16, 3, 32, 32)
                flat_layer_module_list = create_layer_sequence(net, nas_space)
                num_tot_layers = len(flat_layer_module_list)
                num_conv_layers = num_tot_layers//3
                for i in range(num_conv_layers):
                    state_dict['layer' + str(i)] = {}
                hookF = [Hook(layer) for layer in flat_layer_module_list]
                hookB = [Hook(layer, backward=True) for layer in flat_layer_module_list]
                process_network_output(net, data_sample, state_dict, hookF, hookB, flat_layer_module_list, device)
                net.zero_grad()
                hookF = [Hook(layer) for layer in flat_layer_module_list]
                hookB = [Hook(layer, backward=True) for layer in flat_layer_module_list]
                process_network_output(net, data_sample, state_dict, hookF, hookB, flat_layer_module_list, device, process_noise=True)
                net.zero_grad()
                hookF = [Hook(layer) for layer in flat_layer_module_list]
                hookB = [Hook(layer, backward=True) for layer in flat_layer_module_list]
                process_network_output(net, data_sample, state_dict, hookF, hookB, flat_layer_module_list, device, perturb=0.01)
                jitem = []
                perlayer_score = []
                for layer_key in list(state_dict.keys()):
                    if layer_key == "net_info":
                        acc_list.append(state_dict[layer_key]["test-accuracy"])
                    else:
                        if state_dict[layer_key] != {}:
                            eval_out = eval_layer(individual, state_dict[layer_key], device)
                            if eval_out == torch.Tensor([-100]).to(device):
                                pass
                            else:
                                eval_out = torch.sum(eval_out) / torch.numel(eval_out)
                                perlayer_score.append(eval_out.item())
                        else:
                            perlayer_score.append(0)
                            print("There was an error at layer: ", layer_key, " for network: ", i_reg, " thus it is 0")
                score_list.append(sum(perlayer_score)/len(perlayer_score))
            # calculate spearman rank correlation between score_list and acc_list
            tau, _ = stats.kendalltau(np.asarray(score_list), np.asarray(acc_list))
            pearson, _ = stats.pearsonr(np.asarray(score_list), np.asarray(acc_list))
            spearman, _ = stats.spearmanr(np.asarray(score_list), np.asarray(acc_list), nan_policy='omit').correlation
            space_to_corr[nas_space] = (tau, pearson, spearman)
        # Now, return the minimum of the tau across all search spaces
        return min([space_to_corr[x][0] for x in space_to_corr.keys()])
    except Exception as e:
        print("Error in evaluateIndividual: ", e)
        return -1000


def evaluate_scores(state_dict, toolbox, eval_on_sets, last_layer_alg, last_layer_set, nets_score_set, operations, terminals, device, nets_test_acc_set):
    for evset in eval_on_sets:
        func = toolbox.compile(expr=get_individual(evset, toolbox, operations, terminals))
        per_layer_score = []
        if last_layer_alg == True:
            layers_to_eval = list([list(state_dict.keys())[z] for z in last_layer_set])
        else:
            layers_to_eval = list(state_dict.keys())
        for layer_key in layers_to_eval:
            if layer_key == "net_info":
                # import pdb; pdb.set_trace()
                try:
                    nets_test_acc_set[evset].append(
                        state_dict[layer_key]["test-accuracy"]
                    )
                except:
                    nets_test_acc_set[evset].append(
                        state_dict[layer_key]["test_accuracy"]
                    )
            else:
                if state_dict[layer_key] != {}:
                    eval_out = eval_layer(func, state_dict[layer_key], device)
                    if eval_out == torch.Tensor([-100]).to(device):
                        pass
                    else:
                        eval_out = torch.sum(eval_out) / torch.numel(eval_out)
                        per_layer_score.append(eval_out.item())
                else:
                    per_layer_score.append(0)
                    print("There was an error at layer: ", layer_key, " thus it is 0")
        nets_score_set[evset].append(per_layer_score)



def evaluate_networks(nnetz_sam, search_space, network_sampler, api, dataset, train_loader, toolbox, eval_on_sets, last_layer_alg, last_layer_set, operations, terminals, device, nets_score_set, nets_test_acc_set):
    for evset in eval_on_sets:
        fail_counter = 0
        for i_reg in tqdm(nnetz_sam):
            # try:
            # print("Network number: ", i_reg)
            state_dict = {}
            results, net = initialize_network(search_space, i_reg, network_sampler, api, dataset)
            # print("Network initialized, converting to cuda")
            net = net.to(device)
            # print("Network converted to cuda")
            state_dict["net_info"] = results

            for m in net.modules():
                initialize_module(m)

            data_sample = next(iter(train_loader))[0]
            if len(data_sample.shape) == 3:
                assert data_sample.shape[0] == 3
                data_sample = data_sample.unsqueeze(0)

            flat_layer_module_list = create_layer_sequence(net, search_space)

            if len(flat_layer_module_list) % 3 != 0:
                print("Taking secondary approach for network: ", i_reg, " due to incorrect layer sequence of length ", len(flat_layer_module_list))
                flat_layer_module_list = create_layer_sequence_disordered(net, data_sample, device)

            num_tot_layers = len(flat_layer_module_list)
            num_conv_layers = num_tot_layers//3
            for i in range(num_conv_layers):
                state_dict['layer' + str(i)] = {}

            hookF = [Hook(layer) for layer in flat_layer_module_list]
            hookB = [Hook(layer, backward=True) for layer in flat_layer_module_list]
            # print(i_reg)
            try:
                # Process network outputs
                process_network_output(net, data_sample, state_dict, hookF, hookB, flat_layer_module_list, device)
                net.zero_grad()
                hookF = [Hook(layer) for layer in flat_layer_module_list]
                hookB = [Hook(layer, backward=True) for layer in flat_layer_module_list]
                process_network_output(net, data_sample, state_dict, hookF, hookB, flat_layer_module_list, device, process_noise=True)
                net.zero_grad()
                hookF = [Hook(layer) for layer in flat_layer_module_list]
                hookB = [Hook(layer, backward=True) for layer in flat_layer_module_list]
                process_network_output(net, data_sample, state_dict, hookF, hookB, flat_layer_module_list, device, perturb=0.01)
                # Evaluation logic
                evaluate_scores(state_dict, toolbox, eval_on_sets, last_layer_alg, last_layer_set, nets_score_set, operations, terminals, device, nets_test_acc_set)
            except Exception as e:
                print("Error in evaluate_networks for network: ", i_reg, " with error: ", e)
                fail_counter += 1
            if len(nets_score_set[evset])%100==0:
                # print out kendall tau till this point
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
                print("For ", len(nets_processed_score), " networks on ", evset)
                print("Tau: ", tau, " Pearson: ", pearson, " Spearman: ", spearman)
                print("\n\n")
        print("Number of networks failed: ", fail_counter)



def desired_pattern_enforcer(filtered_modules):
    desired_pattern = ["Conv2d", "BatchNorm2d", "ReLU"]
    pattern_index = 0
    corrected_list = []
    for layer in filtered_modules:
        layer_type = layer.__class__.__name__
        if layer_type == desired_pattern[pattern_index]:
            corrected_list.append(layer)
            pattern_index = (pattern_index + 1) % len(desired_pattern)
        elif layer_type == "Conv2d":
            pattern_index = 1
    return corrected_list


def create_layer_sequence_disordered(net, dummy_input, device):    
    execution_order = []
    def forward_hook(module, input, output):
        execution_order.append(module)
    for name, module in net.named_modules():
        module.register_forward_hook(forward_hook)
    net(dummy_input.to(device))
    filtered_modules = [mod for mod in execution_order if 
                        isinstance(mod, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU))]
    filtered_modules = desired_pattern_enforcer(filtered_modules) # repeat to ensure pattern enforced
    filtered_modules = desired_pattern_enforcer(filtered_modules) # hacky.
    filtered_modules = desired_pattern_enforcer(filtered_modules)
    return filtered_modules

def create_layer_sequence(net, search_space):
    
    flat_layer_module_list = [y for y in net.modules() if y.__class__.__name__ == "Conv2d" or y.__class__.__name__ == "BatchNorm2d" or y.__class__.__name__  =="ReLU"]
    flat_layer_module_list_cleaned = []
    idx = 0
    while idx < len(flat_layer_module_list):
        if flat_layer_module_list[idx].__class__.__name__=='Conv2d' and flat_layer_module_list[idx+1].__class__.__name__=='Conv2d':
            flat_layer_module_list_cleaned.append(flat_layer_module_list[idx+1])
            idx = idx+2
        # Skip the entire layer if it does not have batchnorm2d as the next layer (avoid avgpool->conv2d type layers etc)
        elif flat_layer_module_list[idx].__class__.__name__=='Conv2d' and flat_layer_module_list[idx+1].__class__.__name__!='BatchNorm2d':
            idx = idx+1
        else:
            flat_layer_module_list_cleaned.append(flat_layer_module_list[idx])
            idx = idx + 1
    flat_layer_module_list = flat_layer_module_list_cleaned
    flat_layer_module_list = flat_layer_module_list[2:]
    if search_space=='NASBench201' or search_space=='NATSBench':
        flat_layer_module_list = flat_layer_module_list[:-1]
        if flat_layer_module_list[-1].__class__.__name__=='BatchNorm2d' and flat_layer_module_list[-2].__class__.__name__=='BatchNorm2d':
            flat_layer_module_list = flat_layer_module_list[:-1]
    
    return flat_layer_module_list


def initialize_network(search_space, i_reg, network_sampler, api, dataset):
    if search_space == "NDS":
        results = {
            "test-accuracy": network_sampler.get_final_accuracy(i_reg, "s", "s")
        }
        net = network_sampler.get_network(i_reg)
    elif search_space == "NATSBench":
        results = api.get_more_info(i_reg, dataset, hp="90" if search_space == "NATSBench" else 200)
        config = api.get_net_config(i_reg, dataset)
        net = get_cell_based_tiny_net(config)
    elif search_space == "NASBench201":
        try:
            if dataset == 'cifar10':
                acc_results = sum([api.get_more_info(i_reg, 'cifar10-valid', None,
                                                        use_12epochs_result=False,
                                                        is_random=seed)['valid-accuracy'] for seed in [777, 888, 999]])/3.
                val_acc = acc_results
            else:
                acc_results = sum([api.get_more_info(i_reg, dataset, None,
                                                        use_12epochs_result=False,
                                                        is_random=seed)['valid-accuracy'] for seed in [777, 888, 999]])/3.
                val_acc = acc_results
        except:
            if dataset == 'cifar10':
                acc_results = api.get_more_info(i_reg, 'cifar10-valid', None,
                                                        use_12epochs_result=False,
                                                        is_random=False)['valid-accuracy'] 
                val_acc = acc_results
            else:
                acc_results = api.get_more_info(i_reg, dataset, None,
                                                        use_12epochs_result=False,
                                                        is_random=False)['valid-accuracy'] 
                val_acc = acc_results
        results = {"test-accuracy": val_acc}
        results["test_accuracy"] = val_acc
        config = api.get_net_config(i_reg, dataset)
        net = get_cell_based_tiny_net(config)
    else:
        raise ValueError("Invalid search space")

    return results, net


def process_layers(net):
    flat_layer_module_list = [
        layer for layer in net.modules()
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU))
    ]
    cleaned_list = []
    idx = 0
    while idx < len(flat_layer_module_list):
        if isinstance(flat_layer_module_list[idx], torch.nn.Conv2d):
            cleaned_list.append(flat_layer_module_list[idx])
            idx += 2 if idx + 1 < len(flat_layer_module_list) and isinstance(flat_layer_module_list[idx + 1], torch.nn.Conv2d) else 1
        else:
            cleaned_list.append(flat_layer_module_list[idx])
            idx += 1

    return cleaned_list