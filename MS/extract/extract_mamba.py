import argparse
import json
import logging
import os
import torch

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer


MAMBA_MODULES = [
    "backbone.layers.*.mixer.dt_bias",
    "backbone.layers.*.mixer.A_log",
    "backbone.layers.*.mixer.D",
    "backbone.layers.*.mixer.in_proj.weight",
    "backbone.layers.*.mixer.conv1d.weight",
    "backbone.layers.*.mixer.conv1d.bias",
    "backbone.layers.*.mixer.norm.weight",
    "backbone.layers.*.mixer.out_proj.weight",
    "backbone.layers.*.mixer.dt_proj.weight",   # Mamba-1
    "backbone.layers.*.mixer.dt_proj.bias",   # Mamba-1
    "backbone.layers.*.mixer.x_proj.weight",   # Mamba-1
    "backbone.layers.*.norm.weight",
]

# only for Mamba-2
SSM_MODULES = [
    "backbone.layers.*.mixer.D",
    "backbone.layers.*.mixer.dt_bias",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the Mamba model."
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
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(model_path, device="cuda", dtype=torch.float16)

    # Load pruning results
    with open(pruned_model_config_file, "r") as f:
        pruned_config = json.load(f)
    logging.info(f"Detect a pruned model config: {pruned_config}")
    state_dict = model.state_dict()

    def prune_modules(state_dict, idx, module_names):
        for module_name in module_names:
            module_name = module_name.replace("*", str(idx))
            if module_name in state_dict:
                del state_dict[module_name]

    if pruned_config.get("pruned_mamba_block_idx"):
        pruned_mamba_block_idx = pruned_config["pruned_mamba_block_idx"]
        for idx in pruned_mamba_block_idx:
            prune_modules(state_dict, idx, MAMBA_MODULES)
    if pruned_config.get("pruned_ssm_idx"):
        pruned_ssm_idx = pruned_config["pruned_ssm_idx"]
        for idx in pruned_ssm_idx:
            prune_modules(state_dict, idx, SSM_MODULES)

    model.save_pretrained(output_path, state_dict=state_dict)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    main()
