import argparse
import csv
import os
from copy import deepcopy
from pathlib import Path

import jstyleson as json
import torch
from transformers import AutoModelForSequenceClassification

from nncf import NNCFConfig
from nncf.common.utils.os import safe_open
from nncf.experimental.torch.sparsity.movement.algo import MovementSparsityController
from nncf.experimental.torch.nas.bootstrapNAS.training.progressive_shrinking_controller import ProgressiveShrinkingController
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import (
    create_compressed_model_from_algo_names,
)
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import SubnetConfig
from nncf.torch import create_compressed_model
from nncf.torch.model_creation import create_nncf_network


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--importance_weight_dir", type=str, help="importance weight dir")
    parser.add_argument("--config_save_name", type=str, help="nncf config after search space generation")
    args = parser.parse_args()
    return args


def load_nncf_config(nncf_config_path):
    nncf_config_path = Path(nncf_config_path).resolve()
    with safe_open(nncf_config_path) as f:
        loaded_json = json.load(f)
    nncf_config = NNCFConfig.from_dict(loaded_json)
    return nncf_config


class AutoSearchSpaceGenerator:

    def __init__(self, nncf_config, model_name, sparsity_weight):
        self.nncf_config = nncf_config #self.load_nncf_config(nncf_config_path)
        self.bnas_ctrl = self.create_bnas_ctrl(model_name)
        self.sparsity_ctrl = self.create_sparsity_ctrl(model_name, sparsity_weight)

    def create_bnas_ctrl(self, model_name) -> ProgressiveShrinkingController:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        nncf_config = deepcopy(self.nncf_config) # use deepcopy in case there are modifications in ctrl
        nncf_network = create_nncf_network(model, nncf_config)

        algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
        training_ctrl, _ = create_compressed_model_from_algo_names(
            nncf_network, nncf_config, algo_names=[algo_name]
        )
        return training_ctrl

    def create_sparsity_ctrl(self, model_name, sparsity_weight) -> MovementSparsityController:

        def get_dict_name(node_name):
            split_name = node_name.split('/')
            dict_name = []
            for name in split_name:
                if name.endswith(']'):
                    dict_name.append(name[:-1].split('[')[-1])
            return '.'.join(dict_name)

        def load_ctrl_state(sparsity_weight, sparsity_ctrl):
            resume_state_dict = torch.load(sparsity_weight, map_location=sparsity_ctrl.model.device)
            # forget to save ctrl state --> use this function to update the thre
            for sparsity_module in sparsity_ctrl._structured_mask_handler.sparsified_module_info_list:
                # load from state dict
                op = sparsity_module.module.pre_ops['0'].op
                op.training = True
                op.frozen = False
                op.weight_importance.data = resume_state_dict[get_dict_name(sparsity_module.module_node_name) + '.pre_ops.0.op.weight_importance']
                op.weight_ctx.binary_mask = op._calc_training_binary_mask()
                if hasattr(op, "bias_importance"):
                    op.bias_importance.data = resume_state_dict[get_dict_name(sparsity_module.module_node_name) + '.pre_ops.0.op.bias_importance']
                    op.bias_ctx.binary_mask = op._calc_training_binary_mask(is_bias=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        nncf_config = deepcopy(self.nncf_config)

        compression_ctrl, _ = create_compressed_model(model, nncf_config)
        load_ctrl_state(sparsity_weight, compression_ctrl)
        return compression_ctrl

    def get_macs_for_thre(self, thre, output_dir = '/home/yzheng/AutoML/projs/reorg_mask/tmp_resu'):
        def create_subnet_config_based_on_csv(csv_file):
            # an ugly func to generate config based on csv file -- currently only work for bert/vit model
            subnet_width_group = {}
            supernet_width_group = {}
            group_idx = 0
            with open(csv_file) as f:
                reader = csv.reader(f)
                for row in reader:
                    if 'query' in row[2] or 'intermediate' in row[2]:
                        original_shape, prune_shape = int(row[5][1:-2]), int(row[6][1:-2])
                        subnet_width_group[group_idx] = prune_shape
                        supernet_width_group[group_idx] = original_shape
                        group_idx += 1
            return subnet_width_group, supernet_width_group

        def resolve_config_based_on_csv(subnet_width_group, supernet_width_group, width_block_info,
                                        depth_skipped_blocks, propagation_graph, discard_percent = 0.02):
            # discard percent: when subnet width < discard_percent * supernet width --> convert to elastic depth
            resolve_config = SubnetConfig()
            resolve_config[ElasticityDim.WIDTH] = {}
            resolve_config[ElasticityDim.DEPTH] = []
            for width_block_id, subnet_width in subnet_width_group.items():
                supernet_width = supernet_width_group[width_block_id]
                if subnet_width < discard_percent * supernet_width:
                    # convert to elastic depth id
                    cluster = width_block_info.get_cluster_by_id(width_block_id)
                    min_ele_id = min([ele.nncf_node_id for ele in cluster.elements])
                    depth_skipped_blocks_start_node_name = [block.start_node_name for block in depth_skipped_blocks]
                    depth_start_node_id = min_ele_id
                    depth_start_node_name = None
                    while(depth_start_node_id >=0):
                        depth_start_node_name = propagation_graph.get_node_by_id(depth_start_node_id).node_name
                        if depth_start_node_name in depth_skipped_blocks_start_node_name:
                            break
                        depth_start_node_id -= 1
                    if depth_start_node_id < 0:
                        raise ValueError("skipped blocks in depth handler are not correct")
                    resolve_config[ElasticityDim.DEPTH].append(depth_skipped_blocks_start_node_name.index(depth_start_node_name))
                    resolve_config[ElasticityDim.WIDTH][width_block_id] = supernet_width
                else:
                    resolve_config[ElasticityDim.WIDTH][width_block_id] = subnet_width
            return resolve_config

        # generate binary masks based on a specific importance threshold
        for sparsity_module in self.sparsity_ctrl._structured_mask_handler.sparsified_module_info_list:
            sparsity_module.operand._importance_threshold = thre
            op = sparsity_module.module.pre_ops['0'].op
            op.weight_ctx.binary_mask = op._calc_training_binary_mask()
            if hasattr(op, "bias_importance"):
                op.bias_ctx.binary_mask = op._calc_training_binary_mask(is_bias=True)

        self.sparsity_ctrl._scheduler._params.enable_structured_masking = True
        self.sparsity_ctrl._scheduler._controller.reset_independent_structured_mask()
        self.sparsity_ctrl._scheduler._controller.resolve_structured_mask()
        self.sparsity_ctrl._structured_mask_handler.populate_dependent_structured_mask_to_operand()
        self.sparsity_ctrl._structured_mask_handler.report_structured_sparsity(output_dir, f'structured_sparsity_thre{thre}')

        # generate subnet
        subnet_width_group, supernet_width_group = create_subnet_config_based_on_csv(os.path.join(output_dir, f'structured_sparsity_thre{thre}.csv'))
        resolve_config = resolve_config_based_on_csv(subnet_width_group, supernet_width_group,
                                                      self.bnas_ctrl.multi_elasticity_handler.width_handler._pruned_module_groups_info,
                                                      self.bnas_ctrl.multi_elasticity_handler.depth_handler._skipped_blocks,
                                                      self.bnas_ctrl.multi_elasticity_handler.width_handler.propagation_graph)
        self.bnas_ctrl.multi_elasticity_handler.activate_subnet_for_config(resolve_config)
        resolve_subnet_flops, _ = self.bnas_ctrl.multi_elasticity_handler.count_flops_and_weights_for_active_subnet()
        return resolve_subnet_flops / 2e6, resolve_config

    def generate_active_config_for_target_macs(self, target_macs, tolerance = 100):
        thre_min, thre_max = -1, 1 # TODO: replace with importance min & max
        # binary search
        print(f'start searching config for target macs = {target_macs}. This step may take around 1~2 minutes')
        while(thre_min <= thre_max):
            mid = (thre_min + thre_max) / 2
            macs, config = self.get_macs_for_thre(mid)
            if abs(macs - target_macs) <= tolerance:
                print(f'hit! real macs: {macs}, target macs: {target_macs}')
                return config
            elif macs < target_macs:
                thre_max = mid
            else:
                thre_min = mid
        raise ValueError("cannot find a config that satisfy the requirements")

    def combine_search_space(self, target_config_list):
        assert len(target_config_list) > 0
        width_group_num = len(target_config_list[0][ElasticityDim.WIDTH].keys())
        depth_block_num = len(self.bnas_ctrl.multi_elasticity_handler.depth_handler._skipped_blocks)
        overwrite_group_width = [[] for _ in range(width_group_num)]
        depth_fixed_block_id = set(range(depth_block_num)) # if all configs share the same skip block --> fixed block
        depth_skipped_block_id = set() # the skip blocks are only included in some configs

        for target_config in target_config_list:
            # combine elastic width
            for i in range(width_group_num):
                if target_config[ElasticityDim.WIDTH][i] not in overwrite_group_width[i]:
                    overwrite_group_width[i].append(target_config[ElasticityDim.WIDTH][i])

            # combine elastic depth
            depth_fixed_block_id = depth_fixed_block_id & set(target_config[ElasticityDim.DEPTH])
            depth_skipped_block_id = depth_skipped_block_id | set(target_config[ElasticityDim.DEPTH])

        depth_skipped_block_id = depth_skipped_block_id.difference(depth_fixed_block_id)
        # prepare the result based on the nncf config format
        for i in range(width_group_num):
            overwrite_group_width[i] = sorted(overwrite_group_width[i], reverse=True)
        depth_fixed_block = [[self.bnas_ctrl.multi_elasticity_handler.depth_handler._skipped_blocks[i].start_node_name,
                              self.bnas_ctrl.multi_elasticity_handler.depth_handler._skipped_blocks[i].end_node_name]
                               for i in depth_fixed_block_id]
        depth_skipped_block = [[self.bnas_ctrl.multi_elasticity_handler.depth_handler._skipped_blocks[i].start_node_name,
                              self.bnas_ctrl.multi_elasticity_handler.depth_handler._skipped_blocks[i].end_node_name]
                               for i in depth_skipped_block_id]
        return overwrite_group_width, depth_fixed_block, depth_skipped_block

    def convert_to_nncf_config(self, save_config_name, overwrite_group_width, depth_fixed_block, depth_skipped_block):
        new_config = deepcopy(self.nncf_config)
        assert len(self.bnas_ctrl.multi_elasticity_handler.width_handler._pruned_module_groups_info.clusters) == len(overwrite_group_width)
        new_config['bootstrapNAS']['training']['elasticity']['width']['overwrite_groups_widths'] = overwrite_group_width
        assert len(depth_fixed_block) == 0, "depth_fixed_block will be supported in the future"
        new_config['bootstrapNAS']['training']['elasticity']['depth']['skipped_blocks'] = depth_skipped_block

        os.makedirs(os.path.dirname(os.path.abspath(save_config_name)), exist_ok=True)
        with open(save_config_name, 'w') as f:
            json.dump(new_config, f)
        print(f'save new config to {save_config_name}')

    def generate_search_space(self, cmax, cmin, N, save_config_name):
        # step 1 : get maximum subnet
        self.bnas_ctrl.multi_elasticity_handler.activate_maximum_subnet()
        maximum_flops, _ = self.bnas_ctrl.multi_elasticity_handler.count_flops_and_weights_for_active_subnet()
        cmax = min(cmax, int(maximum_flops / 2e6))
        print(f'cmax: {cmax}')

        # step2: target macs
        target_config_list = []
        step = (cmax - cmin) // (N - 1)
        target_macs_list = list(range(cmin, cmax, step))
        target_macs_list[-1] = cmax
        for target_macs in target_macs_list:
            target_config = self.generate_active_config_for_target_macs(target_macs)
            target_config_list.append(target_config)
        overwrite_group_width, depth_fixed_block, depth_skipped_block = self.combine_search_space(target_config_list)
        self.convert_to_nncf_config(save_config_name, overwrite_group_width, depth_fixed_block, depth_skipped_block)


if __name__ == "__main__":
    cmax = 11500
    cmin = 4400
    N = 5
    args = get_argument_parser()
    # nncf config should include all possible blocks (both elastic depth and elastic width)--> search space will be generated based on those groups
    template_nncf_config = load_nncf_config('./eftnas_configs/eftnas_search_space_demo.json')
    sparsity_weight = os.path.join(args.importance_weight_dir, 'movement_sparsity_0/pytorch_model.bin')
    assert os.path.exists(sparsity_weight), "importance weight dir is incorrect"

    ## overwrite some hyperparam in nncf config
    search_space_gen = AutoSearchSpaceGenerator(template_nncf_config, "bert-base-uncased", sparsity_weight)
    search_space_gen.generate_search_space(cmax, cmin, N, args.config_save_name)

