import argparse
import json
import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# mamba-2
MAMBA_MODULES = [
    "model.mamba_layers.*.mamba.dt_bias",
    "model.mamba_layers.*.mamba.A_log",
    "model.mamba_layers.*.mamba.D",
    "model.mamba_layers.*.mamba.in_proj.0.weight",
    "model.mamba_layers.*.mamba.conv1d.weight",
    "model.mamba_layers.*.mamba.conv1d.bias",
    "model.mamba_layers.*.mamba.norm.weight",
    "model.mamba_layers.*.mamba.out_proj.weight",
    "model.mamba_layers.*.input_layernorm.weight"
]

MHA_MODULES = [
    "model.blocks.*.self_attn.q_proj.weight",
    "model.blocks.*.self_attn.k_proj.weight",
    "model.blocks.*.self_attn.v_proj.weight",
    "model.blocks.*.self_attn.o_proj.weight",
    "model.blocks.*.input_layernorm.weight",
    "model.blocks.*.self_attn.q_proj.bias",
    "model.blocks.*.self_attn.k_proj.bias",
    "model.blocks.*.self_attn.v_proj.bias",
    "model.blocks.*.self_attn.o_proj.bias",
]

MLP_MODULES = [
    "model.blocks.*.feed_forward.linear_fc1_up.weight",
    "model.blocks.*.feed_forward.linear_fc1_gate.weight",
    "model.blocks.*.feed_forward.linear_fc2.weight",
    "model.blocks.*.pre_ff_layernorm.weight",
    "model.blocks.*.feed_forward.linear_fc1_up.bias",
    "model.blocks.*.feed_forward.linear_fc1_gate.bias",
    "model.blocks.*.feed_forward.linear_fc2.bias",
]

SSM_MODULES = [
    "model.mamba_layers.*.mamba.D",
    "model.mamba_layers.*.mamba.dt_bias",
]

DEPENDENCY_GROUPS = {
    "linear_fc2": "linear_fc1_up"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the Zamba model."
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
    output_path = args.output_path
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    pruned_model_config_file = args.pruned_model_config_file

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.float16)

    # Load pruning results
    with open(pruned_model_config_file, "r") as f:
        pruned_config = json.load(f)
    logging.info(f"Detect a pruned model config: {pruned_config}")
    state_dict = model.state_dict()

    if pruned_config.get("pruned_mlp_channels"):
        mlp_groups = []
        module_to_weight = {}
        for transformers_block in model.model.blocks:
            mlp_modules = transformers_block.feed_forward
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

    if pruned_config.get("pruned_mamba_block_idx"):
        pruned_mamba_block_idx = pruned_config["pruned_mamba_block_idx"]
        for idx in pruned_mamba_block_idx:
            prune_modules(state_dict, idx, MAMBA_MODULES)
    if pruned_config.get("pruned_mha_idx"):
        pruned_mha_idx = pruned_config["pruned_mha_idx"]
        for idx in pruned_mha_idx:
            prune_modules(state_dict, idx, MHA_MODULES)
    if pruned_config.get("pruned_mlp_idx"):
        pruned_mlp_idx = pruned_config["pruned_mlp_idx"]
        for idx in pruned_mlp_idx:
            prune_modules(state_dict, idx, MLP_MODULES)
    if pruned_config.get("pruned_ssm_idx"):
        pruned_ssm_idx = pruned_config["pruned_ssm_idx"]
        for idx in pruned_ssm_idx:
            prune_modules(state_dict, idx, SSM_MODULES)

    model.save_pretrained(output_path, state_dict=state_dict)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    main()
