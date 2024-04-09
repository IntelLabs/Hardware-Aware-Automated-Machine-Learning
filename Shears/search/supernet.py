import os
import re
from typing import TypeVar

import torch
from peft.utils import CONFIG_NAME
from peft.utils import WEIGHTS_NAME

from nncf import NNCFConfig
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.search.supernet import TrainedSuperNet
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import (
    create_compressed_model_from_algo_names,
)
from nncf.torch.model_creation import create_nncf_network

TModel = TypeVar("TModel")
P = re.compile(r"[[](.*?)[]]", re.S)


class ShearsSuperNet(TrainedSuperNet):
    """
    An interface for handling trained Shears super-networks. This class can be used to quickly implement
    third party solutions for subnetwork search on existing super-networks.
    """

    @classmethod
    def from_checkpoint(
        cls,
        model: TModel,
        nncf_config: NNCFConfig,
        supernet_elasticity_path: str,
        supernet_weights_path: str,
    ) -> "ShearsSuperNet":
        """
        Loads existing super-network weights and elasticity information, and creates the SuperNetwork interface.

        :param model: base model that was used to create the super-network.
        :param nncf_config: configuration used to create the super-network.
        :param supernet_elasticity_path: (not used in this subclass) path to file containing state information about
            the super-network.
        :param supernet_weights_path: (not used in this subclass) trained weights to resume the super-network.
        :return: Shears super-network with wrapped functionality.
        """
        nncf_network = create_nncf_network(model, nncf_config)
        algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
        elasticity_ctrl, model = create_compressed_model_from_algo_names(
            nncf_network, nncf_config, algo_names=[algo_name]
        )
        elasticity_ctrl.multi_elasticity_handler.enable_all()
        elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
        return cls(elasticity_ctrl, model)

    def activate_heuristic_subnet(self) -> None:
        """
        Activates the heuristic subnetwork in the super-network (only for width).
        """
        heuristic_config = {k: v[(len(v) - 1) // 2] for k, v in self._m_handler.width_search_space.items()}
        heuristic_config = {ElasticityDim.WIDTH: heuristic_config}
        self.activate_config(heuristic_config)

    def extract_and_save_active_sub_adapter(self, super_adapter_dir, sub_adapter_dir) -> None:
        """
        Extracts and saves the active sub-adapter in the super-network.
        """
        # TODO: better way to obtain the pruned module groups
        pruned_module_groups = self._m_handler.width_handler._pruned_module_groups_info.get_all_clusters()
        width_search_space = self._m_handler.width_search_space
        subnet_width_config = self.get_active_config()[ElasticityDim.WIDTH]
        assert len(pruned_module_groups) == len(width_search_space) == len(subnet_width_config)
        num_group = len(pruned_module_groups)

        node_value_dict = {}
        for i in range(num_group):
            group = pruned_module_groups[i]
            space = width_search_space[i]
            value = subnet_width_config[i]
            assert min(space) <= value <= max(space)
            cur_dict = {node.node_name: value for node in group.elements}
            node_value_dict.update(cur_dict)

        # map: module name -> width
        module_value_dict = {".".join(re.findall(P, k)[:-2]): v for k, v in node_value_dict.items()}

        trained_adapter_weights = torch.load(os.path.join(super_adapter_dir, WEIGHTS_NAME))
        pruned_adapter_weights = {}
        for k, v in trained_adapter_weights.items():
            module_name = ".".join(k.split(".")[:-2])
            value = module_value_dict[module_name]
            is_loraA = "lora_A" in k
            new_v = v[:value].clone() if is_loraA else v[:, :value].clone()
            pruned_adapter_weights[k] = new_v

        os.makedirs(sub_adapter_dir, exist_ok=True)
        torch.save(pruned_adapter_weights, os.path.join(sub_adapter_dir, WEIGHTS_NAME))
        config_path = os.path.join(super_adapter_dir, CONFIG_NAME)
        os.system(f"cp {config_path} {sub_adapter_dir}")
