
import sys
import time
import zipfile
from pathlib import Path
import logging
import warnings  # to disable warnings on export to ONNX
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import nncf  # Important - should be imported directly after torch
from nncf.common.utils.logger import set_log_level
set_log_level(logging.ERROR)  # Disables all NNCF info and warning messages
from nncf import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.experimental.torch.nas.bootstrapNAS import EpochBasedTrainingAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS import SearchAlgorithm
from nncf.torch import create_compressed_model, register_default_init_args
from nncf.torch.initialization import wrap_dataloader_for_init
from nncf.torch.model_creation import create_nncf_network

torch.manual_seed(0)

print("Imported PyTorch and NNCF")
