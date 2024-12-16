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

import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Path to the pre-trained model."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="prune_result",
        help="Directory to save the compressed model."
    )
    parser.add_argument(
        "--weight_reorder",
        action="store_true",
        help="Flag to indicate whether to perform weight reorder."
    )
    parser.add_argument(
        "--pruned_model_config_file",
        type=str,
        default=None,
        help="Path to the pruned model configuration file."
    )

    args = parser.parse_args()
    model_path = args.model_path
    output_path = args.output_path
    weight_reorder = args.weight_reorder
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    pruned_model_config_file = args.pruned_model_config_file

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype="float16",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if weight_reorder:
        for layer in utils.get_layers(model):
            utils.reorder_in_attn_block(getattr(layer, utils.get_attn_key(model)), model=model)
            utils.reorder_in_mlp_block(getattr(layer, utils.get_mlp_key(model)))

    # Load pruning results
    with open(pruned_model_config_file, "r") as f:
        pruned_config = json.load(f)
    logging.info(f"Detect a pruned model config: {pruned_config}")
    state_dict = model.state_dict()

    def get_groups(model, key):
        groups = []
        for layer in utils.get_layers(model):
            modules = getattr(layer, key)
            groups.append({name: module for name, module in modules.named_children() if isinstance(module, torch.nn.Linear)})
        return groups

    def get_pruned_weights(groups, pruned_channels):
        num_key_value_heads = utils.get_num_kv_heads(model)
        num_attention_heads = utils.get_num_attention_heads(model)
        num_key_value_groups = num_attention_heads // num_key_value_heads
        k_proj_key, v_proj_key = utils.get_k_key(model), utils.get_v_key(model)

        module_to_weight = {}
        for group_idx, value in pruned_channels.items():
            group = groups[int(group_idx)]
            for name, module in group.items():
                if name in utils.DEPENDENCY_GROUPS:
                    module_to_weight[module] = module.weight[:, :value]
                else:
                    if name in [k_proj_key, v_proj_key]:
                        module_to_weight[module] = module.weight[:value // num_key_value_groups]
                    else:
                        module_to_weight[module] = module.weight[:value]
        return module_to_weight

    mlp_groups = get_groups(model, utils.get_mlp_key(model))
    attn_groups = get_groups(model, utils.get_attn_key(model))
    module_to_weight = {}
    if pruned_config.get("pruned_attn_width"):
        module_to_weight.update(get_pruned_weights(attn_groups, pruned_config["pruned_attn_width"]))
    if pruned_config.get("pruned_mlp_width"):
        module_to_weight.update(get_pruned_weights(mlp_groups, pruned_config["pruned_mlp_width"]))

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

    def prune_modules(state_dict, idx, key):
        target = f".{str(idx)}.{key}"
        remove_key = []
        for name, module in state_dict.items():
            if target in name:
                remove_key.append(name)
        for key in remove_key:
            del state_dict[key]

    if pruned_config.get("pruned_attn_idx"):
        pruned_attn_idx = pruned_config["pruned_attn_idx"]
        for idx in pruned_attn_idx:
            prune_modules(state_dict, idx, utils.get_attn_key(model))
    if pruned_config.get("pruned_mlp_idx"):
        pruned_mlp_idx = pruned_config["pruned_mlp_idx"]
        for idx in pruned_mlp_idx:
            prune_modules(state_dict, idx, utils.get_mlp_key(model))

    model.save_pretrained(output_path, state_dict=state_dict)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    main()
