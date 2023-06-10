import os

import numpy as np
import torch
import torch.nn as nn
from mgca.utils.segmentation_loss import MixedLoss
from pytorch_lightning import LightningModule

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Segmenter(LightningModule):
    def __init__(
        self, 
        seg_model: nn.Module,
    ):
        pass
