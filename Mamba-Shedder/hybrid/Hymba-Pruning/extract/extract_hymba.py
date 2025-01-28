import argparse
import json
import logging
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from prune import DEPENDENCY_GROUPS, reorder_in_mlp_block


SELF_ATTN_MODULES = [
    "model.layers.*.mamba.pre_avg_layernorm1.weight",
]


SSM_MODULES = [
    "model.layers.*.mamba.A_log.0",
    "model.layers.*.mamba.B_layernorm.weight",
    "model.layers.*.mamba.C_layernorm.weight",
    "model.layers.*.mamba.D.0",
    "model.layers.*.mamba.dt_layernorm.weight",
    "model.layers.*.mamba.dt_proj.0.bias",
    "model.layers.*.mamba.dt_proj.0.weight",
    "model.layers.*.mamba.pre_avg_layernorm2.weight",
    "model.layers.*.mamba.x_proj.0.weight"
]


HYMBA_BLOCK_MODULES = [
    "model.layers.*.mamba.A_log.0",
    "model.layers.*.mamba.B_layernorm.weight",
    "model.layers.*.mamba.C_layernorm.weight",
    "model.layers.*.mamba.conv1d.bias",
    "model.layers.*.mamba.conv1d.weight",
    "model.layers.*.mamba.D.0",
    "model.layers.*.mamba.dt_layernorm.weight",
    "model.layers.*.mamba.dt_proj.0.bias",
    "model.layers.*.mamba.dt_proj.0.weight",
    "model.layers.*.mamba.in_proj.weight",
    "model.layers.*.mamba.out_proj.weight",
    "model.layers.*.mamba.pre_avg_layernorm1.weight",
    "model.layers.*.mamba.pre_avg_layernorm2.weight",
    "model.layers.*.mamba.x_proj.0.weight"
]


MOE_MODULES = [
    "model.layers.*.moe.experts.0.down_proj.weight",
    "model.layers.*.moe.experts.0.gate_proj.weight",
    "model.layers.*.moe.experts.0.up_proj.weight",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the Hymba model."
    )
    parser.add_argument(
        "--weight_reorder",
        action="store_true",
        help="Flag to indicate whether to perform weight reorder in MOE module (FFN, MLP)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Directory to save the compressed model."
    )
    parser.add_argument(
        "--pruned_model_config_file",
        type=str,
        help="Path to the pruned model configuration file."
    )

    args = parser.parse_args()
    model_path = args.model_path
    weight_reorder = args.weight_reorder
    output_path = args.output_path
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    pruned_model_config_file = args.pruned_model_config_file

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.float16, trust_remote_code=True)

    if weight_reorder:
        for layer in model.model.layers:
            reorder_in_mlp_block(layer.moe.experts[0])

    # Load pruning results
    with open(pruned_model_config_file, "r") as f:
        pruned_config = json.load(f)
    logging.info(f"Detect a pruned model config: {pruned_config}")
    state_dict = model.state_dict()

    if pruned_config.get("pruned_mlp_channels"):
        mlp_groups = []
        module_to_weight = {}
        for layer in model.model.layers:
            mlp_modules = layer.moe.experts[0]
            mlp_groups.append({name: module for name, module in mlp_modules.named_children() if isinstance(module, torch.nn.Linear)})

        for group_idx, value in pruned_config["pruned_mlp_channels"].items():
            group = mlp_groups[int(group_idx)]
            for name, module in group.items():
                if name in DEPENDENCY_GROUPS:
                    module_to_weight[module] = module.weight[:, :value]
                else:
                    module_to_weight[module] = module.weight[:value]

        linear_modules = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
        for name, module in linear_modules.items():
            if module in module_to_weight:
                sd_weight_key = name + ".weight"
                assert sd_weight_key in state_dict
                pruned_weight = module_to_weight[module]
                state_dict[sd_weight_key] = pruned_weight.clone()
                # bias
                sd_bias_key = name + ".bias"
                if sd_bias_key in state_dict:
                    state_dict[sd_bias_key] = state_dict[sd_bias_key][:pruned_weight.size(0)].clone()

    def prune_modules(state_dict, idx, module_names):
        for module_name in module_names:
            module_name = module_name.replace("*", str(idx))
            if module_name in state_dict:
                del state_dict[module_name]

    if pruned_config.get("pruned_self_attn_idx"):
        pruned_self_attn_idx = pruned_config["pruned_self_attn_idx"]
        for idx in pruned_self_attn_idx:
            prune_modules(state_dict, idx, SELF_ATTN_MODULES)
    if pruned_config.get("pruned_ssm_idx"):
        pruned_ssm_idx = pruned_config["pruned_ssm_idx"]
        for idx in pruned_ssm_idx:
            prune_modules(state_dict, idx, SSM_MODULES)
    if pruned_config.get("pruned_hymba_block_idx"):
        pruned_hymba_block_idx = pruned_config["pruned_hymba_block_idx"]
        for idx in pruned_hymba_block_idx:
            prune_modules(state_dict, idx, HYMBA_BLOCK_MODULES)
    if pruned_config.get("pruned_moe_idx"):
        pruned_moe_idx = pruned_config["pruned_moe_idx"]
        for idx in pruned_moe_idx:
            prune_modules(state_dict, idx, MOE_MODULES)

    model.save_pretrained(output_path, state_dict=state_dict)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    main()
