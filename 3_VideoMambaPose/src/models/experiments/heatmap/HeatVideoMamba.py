"""Combining implementations from ViTPose (heatmap deconv) using Video MambaBlocks
https://github.com/ViTAE-Transformer/ViTPose
"""

import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint
 
# remember that this is einstein operation, which is the special fancy way of reshaping.
from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import HeatmapVideoMamba as hvm

class Deconv(nn.Module):
    """
    This was inspired by the ViTPose deconvolution process
    https://github.com/ViTAE-Transformer/ViTPose/blob/d5216452796c90c6bc29f5c5ec0bdba94366768a/mmpose/models/heads/deconv_head.py#L12
    """
    def __init__(self):
        super().__init__()

        # 1. need to make sure of the shape of the hidden state. Should not be single dimensional. I should be able to have frame per frame.
        self.
    def generate_deconv(self, input):



class HeatMapVideoMambaPose(nn.Module):
    def __init(self):
        super().__init__()
        self.mamba = vm.videomamba_tiny()
        self.decoder = 

    def forward(self, x):
        x = self.mamba(x)
        print(x.shape)
        x = self.decoder(x)

        return x


if __name__ == "__main__":

    # generating a random input
    # (Batch, Channel number, NumFrames, W, H) = (16, 3, 8, 224, 224)

    # Define the dimensions
    batch_size = 16
    num_frames = 8
    height = 224
    width = 224
    channels = 3

    # Generate a random tensor
    test_video = torch.rand(batch_size, channels, num_frames, height, width) # I get an error .... 384, 3, 1, 16, 16

    # Check the shape of the random tensor
    print("Shape of the random tensor:", test_video.shape)


