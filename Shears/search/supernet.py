from typing import TypeVar

from nncf import NNCFConfig
from nncf.experimental.torch.nas.bootstrapNAS.search.supernet import TrainedSuperNet
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import (
    create_compressed_model_from_algo_names,
)
from nncf.torch.model_creation import create_nncf_network

TModel = TypeVar("TModel")


class TransformerSuperNet(TrainedSuperNet):
    """
    An interface for handling pre-trained transformer-based super-networks. This class can be used to quickly implement
    third party solutions for subnetwork search on existing super-networks.
    """

    @classmethod
    def from_checkpoint(
        cls,
        model: TModel,
        nncf_config: NNCFConfig,
        supernet_elasticity_path: str,
        supernet_weights_path: str,
    ) -> "TrainedSuperNet":
        """
        Loads existing super-network weights and elasticity information, and creates the SuperNetwork interface.

        :param model: base model that was used to create the super-network.
        :param nncf_config: configuration used to create the super-network.
        :param supernet_elasticity_path: path to file containing state information about the super-network.
        :param supernet_weights_path: trained weights to resume the super-network.
        :return: SuperNetwork with wrapped functionality.
        """
        nncf_network = create_nncf_network(model, nncf_config)
        algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
        elasticity_ctrl, model = create_compressed_model_from_algo_names(
            nncf_network, nncf_config, algo_names=[algo_name]
        )
        elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
        return TrainedSuperNet(elasticity_ctrl, model)
