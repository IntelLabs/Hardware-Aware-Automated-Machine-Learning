"""
Some NNCF config preprocessing code for Shears.

This module provides preprocessing functionality for NNCF (Neural Network Compression Framework) configuration files
used in Shears. It includes utility functions for handling JSON files and preprocessing NNCF configurations.
"""
import json
from pathlib import Path
from nncf import NNCFConfig
from nncf.common.utils.os import safe_open

def parse_nncf_config(nncf_config_path, num=1):

    with safe_open(Path(nncf_config_path)) as f:
        loaded_json = json.load(f)

    base_overwrite_groups = loaded_json["bootstrapNAS"]["training"]["elasticity"]["width"]["overwrite_groups"]
    base_overwrite_groups_widths = loaded_json["bootstrapNAS"]["training"]["elasticity"]["width"][
        "overwrite_groups_widths"]
    overwrite_groups, overwrite_groups_widths = [], []
    for group, width in zip(base_overwrite_groups, base_overwrite_groups_widths):
        if group[0].startswith("{re}"):
            new_group = [[item.replace("{re}", "").replace("{*}", str(i)) for item in group] for i in range(num)]
            new_width = [width for _ in range(num)]
        else:
            new_group = [group]
            new_width = [width]
        overwrite_groups.extend(new_group)
        overwrite_groups_widths.extend(new_width)

    loaded_json["bootstrapNAS"]["training"]["elasticity"]["width"]["overwrite_groups"] = overwrite_groups
    loaded_json["bootstrapNAS"]["training"]["elasticity"]["width"][
        "overwrite_groups_widths"] = overwrite_groups_widths
    return loaded_json

def add_lr_epochs(nncf_config, lr=3e-4, epochs=3):
    stage_desc = nncf_config["bootstrapNAS"]["training"]["schedule"]["list_stage_descriptions"][0]
    if stage_desc["init_lr"] == -1:
        stage_desc["init_lr"] = lr
    if stage_desc["epochs"] == -1:
        stage_desc["epochs"] = epochs
        stage_desc["epochs_lr"] = epochs

    return nncf_config

def load_nncf_config(nncf_config, lr=3e-4, epochs=3, num_hidden_layers=32):
    loaded_json = parse_nncf_config(nncf_config, num=num_hidden_layers)
    loaded_json = add_lr_epochs(loaded_json, lr=lr, epochs=epochs)
    nncf_config = NNCFConfig.from_dict(loaded_json)
    return nncf_config
