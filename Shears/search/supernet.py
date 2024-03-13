from typing import TypeVar

from nncf import NNCFConfig
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.search.supernet import TrainedSuperNet
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import (
    create_compressed_model_from_algo_names,
)
from nncf.torch.model_creation import create_nncf_network

TModel = TypeVar("TModel")


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
        heuristic_config = {
            k: v[(len(v) - 1) // 2] for k, v in self._m_handler.width_search_space.items()
        }
        heuristic_config = {ElasticityDim.WIDTH: heuristic_config}
        self.activate_config(heuristic_config)
