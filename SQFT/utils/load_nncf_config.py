"""
Some NNCF config preprocessing code for Shears.

This module provides preprocessing functionality for NNCF (Neural Network Compression Framework) configuration files
used in Shears. It includes utility functions for handling JSON files and preprocessing NNCF configurations.
"""
import json
from pathlib import Path
from nncf import NNCFConfig
from nncf.common.utils.os import safe_open

def parse_nncf_config(nncf_config_path, num_hidden_layers=1, search_space=None):
    """Parse and preprocess the NNCF configuration file.

    Args:
        nncf_config_path (str): Path to the NNCF configuration file.
        num_hidden_layers (int): Number of hidden layers to consider for the search space.
        search_space (list, optional): List of search space widths. Defaults to None.

    Returns:
        dict: The preprocessed NNCF configuration.
    """
    with safe_open(Path(nncf_config_path)) as file:
        loaded_json = json.load(file)

    base_overwrite_groups = loaded_json["bootstrapNAS"]["training"]["elasticity"]["width"]["overwrite_groups"]
    base_overwrite_groups_widths = loaded_json["bootstrapNAS"]["training"]["elasticity"]["width"][
        "overwrite_groups_widths"]
    overwrite_groups, overwrite_groups_widths = [], []
    for group, width in zip(base_overwrite_groups, base_overwrite_groups_widths):
        current_search_space = width if search_space is None else search_space
        if group[0].startswith("{re}"):
            new_group = [[item.replace("{re}", "").replace("{*}", str(i)) for item in group] for i in range(num_hidden_layers)]
            new_width = [current_search_space for _ in range(num_hidden_layers)]
        else:
            new_group = [group]
            new_width = [current_search_space]
        overwrite_groups.extend(new_group)
        overwrite_groups_widths.extend(new_width)

    loaded_json["bootstrapNAS"]["training"]["elasticity"]["width"]["overwrite_groups"] = overwrite_groups
    loaded_json["bootstrapNAS"]["training"]["elasticity"]["width"][
        "overwrite_groups_widths"] = overwrite_groups_widths
    return loaded_json

def add_lr_epochs(nncf_config, learning_rate=3e-4, num_epochs=3):
    """Add learning rate and epochs to the NNCF configuration.

    Args:
        nncf_config (dict): The NNCF configuration dictionary.
        learning_rate (float): The initial learning rate to set.
        num_epochs (int): The number of epochs to set.

    Returns:
        dict: The updated NNCF configuration.
    """
    stage_description = nncf_config["bootstrapNAS"]["training"]["schedule"]["list_stage_descriptions"][0]
    if stage_description["init_lr"] == -1:
        stage_description["init_lr"] = learning_rate
    if stage_description["epochs"] == -1:
        stage_description["epochs"] = num_epochs
        stage_description["epochs_lr"] = num_epochs

    return nncf_config

def load_nncf_config(nncf_config_path, learning_rate=3e-4, num_epochs=3, num_hidden_layers=32, search_space=None):
    """Load and preprocess the NNCF configuration file.

    Args:
        nncf_config_path (str): Path to the NNCF configuration file.
        learning_rate (float): The initial learning rate to set.
        num_epochs (int): The number of epochs to set.
        num_hidden_layers (int): Number of hidden layers to consider for the search space.
        search_space (str, optional): Comma-separated string of search space widths. Defaults to None.

    Returns:
        NNCFConfig: The preprocessed NNCF configuration object.
    """
    if search_space is not None:
        search_space = [int(width) for width in search_space.split(",")]
    loaded_json = parse_nncf_config(nncf_config_path, num_hidden_layers=num_hidden_layers, search_space=search_space)
    loaded_json = add_lr_epochs(loaded_json, learning_rate=learning_rate, num_epochs=num_epochs)
    nncf_config = NNCFConfig.from_dict(loaded_json)
    return nncf_config
