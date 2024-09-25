import argparse
import os
import re
import sys
sys.path.append('.')

import torch
from peft.utils import CONFIG_NAME, WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME
from transformers import AutoConfig

from utils.load_nncf_config import load_nncf_config

PATTERN = re.compile(r"[[](.*?)[]]", re.S)


def get_width_for_query_prefix(torch_module_to_width, query_module, length=5):
    """
    Get the width for a given query module prefix.

    Args:
        torch_module_to_width (dict): Mapping from torch module to width.
        query_module (str): The query module name.
        length (int, optional): The length of the prefix to match. Default is 5.

    Returns:
        int: The width for the query module prefix.
    """
    query_module_list = query_module.split(".")
    width = next(
        (
            value
            for torch_module, value in torch_module_to_width.items()
            if torch_module.split(".")[:length] == query_module_list[:length]
        ),
        None,
    )
    return width


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        required=True,
        type=str,
        help="Path to trained model or model identifier from huggingface.co/models",
    )
    parser.add_argument("--adapter_model", required=True, type=str, help="Path to trained adapter")
    parser.add_argument("--nncf_config", required=True, type=str, help="Path to the NNCF configuration file")
    parser.add_argument("--search_space", required=True, default=None, type=str, help="Search space for the subnetwork")
    parser.add_argument(
        "--subnet_version", choices=["maximal", "heuristic", "minimal", "custom"], default="heuristic", type=str,
        help="Version of the subnetwork to extract"
    )
    parser.add_argument(
        "--custom_config", default=None, type=str, help="Dictionary with the configuration of the subnetwork"
    )
    parser.add_argument(
        "--output_dir",
        default="extracted_sub_adapter",
        type=str,
        help="The output directory where the sub-adapter will be saved",
    )

    args = parser.parse_args()
    base_model_path = args.base_model
    adapter_model_path = args.adapter_model
    nncf_config = args.nncf_config
    search_space = args.search_space
    subnet_version = args.subnet_version
    custom_config = args.custom_config
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load search space (currently consider `width` only)
    model_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    nncf_config = load_nncf_config(
        nncf_config,
        num_hidden_layers=model_config.num_hidden_layers,
        search_space=search_space,
    )
    try:
        overwrite_groups = nncf_config["bootstrapNAS"]["training"]["elasticity"]["width"]["overwrite_groups"]
        overwrite_groups_widths = nncf_config["bootstrapNAS"]["training"]["elasticity"]["width"][
            "overwrite_groups_widths"
        ]
        assert len(overwrite_groups) == len(overwrite_groups_widths)
    except Exception:
        raise ValueError("Cannot get the search space in NNCF config.")

    if subnet_version == "maximal":
        subnetwork_config = {idx: space[0] for idx, space in enumerate(overwrite_groups_widths)}
    elif subnet_version == "heuristic":
        subnetwork_config = {idx: space[(len(space) - 1) // 2] for idx, space in enumerate(overwrite_groups_widths)}
    elif subnet_version == "minimal":
        subnetwork_config = {idx: space[-1] for idx, space in enumerate(overwrite_groups_widths)}
    else:
        assert custom_config is not None, "Missing custom subnetwork config."
        from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim

        subnetwork_config = torch.load(custom_config)["subnet_config"][ElasticityDim.WIDTH]

    # Mapping: nncf node -> width
    nncf_node_to_width = {}
    for idx, value in subnetwork_config.items():
        space = overwrite_groups_widths[idx]
        assert min(space) <= value <= max(space)
        cur_dict = {node: value for node in overwrite_groups[idx]}
        nncf_node_to_width.update(cur_dict)

    # Prune adapter model (LoRA low-rank)
    lora_torch_module_to_width = {
        ".".join(re.findall(PATTERN, k)): v for k, v in nncf_node_to_width.items() if "lora_A" in k
    }
    num_module_name_item = list(lora_torch_module_to_width.keys())[0].split(".").index("lora_A")
    # Load adapter weights
    try:
        super_adapter_weights = torch.load(os.path.join(adapter_model_path, WEIGHTS_NAME))
    except:
        from safetensors.torch import load_file
        super_adapter_weights = load_file(os.path.join(adapter_model_path, SAFETENSORS_WEIGHTS_NAME))
    sub_adapter_weights = {}
    for weight_key, weight_tensor in super_adapter_weights.items():
        width = get_width_for_query_prefix(lora_torch_module_to_width, weight_key, length=num_module_name_item)
        if width is not None:
            is_loraA = "lora_A" in weight_key
            new_weight_tensor = weight_tensor[:width].clone() if is_loraA else weight_tensor[:, :width].clone()
        else:
            new_weight_tensor = weight_tensor.clone()
        sub_adapter_weights[weight_key] = new_weight_tensor
    os.makedirs(output_dir, exist_ok=True)
    torch.save(sub_adapter_weights, os.path.join(output_dir, WEIGHTS_NAME))
    config_path = os.path.join(adapter_model_path, CONFIG_NAME)
    os.system(f"cp {config_path} {output_dir}")


if __name__ == "__main__":
    main()
