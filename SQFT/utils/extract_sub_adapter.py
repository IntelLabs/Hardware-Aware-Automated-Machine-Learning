import argparse
import json
import os
import shutil

import torch
from peft.utils import CONFIG_NAME, WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME


def is_valid_elastic_adapter_config(config):
    if not isinstance(config, list):
        return False
    for item in config:
        if not isinstance(item, dict):
            return False
        if "target" not in item or "search_space" not in item:
            return False
        if not isinstance(item["target"], list) or not isinstance(item["search_space"], list):
            return False
    return True


def is_valid_custom_adapter_config(config):
    if not isinstance(config, list):
        return False
    for item in config:
        if not isinstance(item, dict):
            return False
        if "target" not in item or "active_rank_value" not in item:
            return False
        if not isinstance(item["target"], list) or not isinstance(item["active_rank_value"], int):
            return False
    return True


def load_custom_adapter_config(custom_adapter_config_file):
    if custom_adapter_config_file is None:
        raise ValueError("Missing custom adapter config file.")

    with open(custom_adapter_config_file, "r") as f:
        custom_adapter_config = json.load(f)
    
    if not is_valid_custom_adapter_config(custom_adapter_config):
        raise ValueError("Invalid configuration: `custom_adapter_config`.")
    
    return custom_adapter_config


def get_rank_value(space, strategy):
    if strategy == "maximal":
        return space[0]
    elif strategy == "heuristic":
        return space[(len(space) - 1) // 2]
    elif strategy == "minimal":
        return space[-1]
    else:
        raise ValueError("Invalid strategy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_model", required=True, type=str, help="Path to trained adapter")
    parser.add_argument(
        "--elastic_adapter_config_file",
        required=True,
        type=str,
        help="The JSON file that stores the search space of the elastic adapter"
    )
    parser.add_argument(
        "--adapter_version",
        choices=["maximal", "heuristic", "minimal", "custom"],
        default="heuristic", type=str,
        help="Version of the subnetwork to extract"
    )
    parser.add_argument(
        "--custom_adapter_config_file",
        type=str,
        help="Dictionary with the configuration of the sub-adapter"
    )
    parser.add_argument(
        "--output_dir",
        default="extracted_sub_adapter",
        type=str,
        help="The output directory where the sub-adapter will be saved",
    )

    args = parser.parse_args()
    adapter_model_path = args.adapter_model
    elastic_adapter_config_file = args.elastic_adapter_config_file
    adapter_version = args.adapter_version
    custom_adapter_config_file = args.custom_adapter_config_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(elastic_adapter_config_file, "r") as f:
       elastic_adapter_config = json.load(f)
       if not is_valid_elastic_adapter_config(elastic_adapter_config):
           raise ValueError("Invalid configuration: `elastic_adapter_config`")

    sub_adapter_config = {}
    if adapter_version in ["maximal", "heuristic", "minimal"]:
        for group in elastic_adapter_config:
            modules = group["target"]
            for module in modules:
                sub_adapter_config[module] = get_rank_value(group["search_space"], adapter_version)
    else:
        custom_adapter_config = load_custom_adapter_config(custom_adapter_config_file)  
        for group in custom_adapter_config:
            modules = group["target"]
            for module in modules:
                sub_adapter_config[module] = group["active_rank_value"]

    # Load adapter weights
    try:
        from safetensors.torch import load_file
        super_adapter_weights = load_file(os.path.join(adapter_model_path, SAFETENSORS_WEIGHTS_NAME))
    except:
        super_adapter_weights = torch.load(os.path.join(adapter_model_path, WEIGHTS_NAME))

    sub_adapter_weights = {}
    for weight_key, weight_tensor in super_adapter_weights.items():
        target_key = weight_key.rsplit('.', 1)[0]
        if "lora_A" in weight_key or "lora_B" in weight_key:
            if "lora_B" in weight_key:
                target_key = target_key.replace("lora_B", "lora_A")
            if target_key in sub_adapter_config:
                active_r = sub_adapter_config[target_key]
                if "lora_A" in weight_key:
                    new_weight_tensor = weight_tensor[:active_r].clone()
                else:  # "lora_B" in weight_key
                    new_weight_tensor = weight_tensor[:, :active_r].clone()
            else:
                new_weight_tensor = weight_tensor.clone()
        else:
            new_weight_tensor = weight_tensor.clone()
        sub_adapter_weights[weight_key] = new_weight_tensor
    os.makedirs(output_dir, exist_ok=True)
    torch.save(sub_adapter_weights, os.path.join(output_dir, WEIGHTS_NAME))
    config_path = os.path.join(adapter_model_path, CONFIG_NAME)
    shutil.copy(config_path, output_dir)


if __name__ == "__main__":
    main()
