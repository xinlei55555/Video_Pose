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
import HeatmapVideoMamba as hvm
import HeatMapDeconv as hmd
import HeatMapJointRegressor as hjr

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class HeatMapVideoMambaPose(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.mamba = hvm.videomamba_tiny()

        # decoder into heatmap
        self.deconv = hmd.Deconv()

        # output into joints
        self.joints = hjr.JointOutput()

    def forward(self, x):
        # x = self.mamba(x)
        # print(x)
        x = self.deconv(x)
        # print('before', x.shape)
        # x = self.joints(x)
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
    # I get an error .... 384, 3, 1, 16, 16
    test_video = torch.rand(batch_size, channels, num_frames, height, width)
    

    # Check the shape of the random tensor
    print("Shape of the random tensor:", test_video.shape)

    test_model = HeatMapVideoMambaPose()

    # move the data to the GPU
    test_model = test_model.to(device)
    test_video = test_video.to(device)

    y = test_model(test_video)

    # * note: when I print (B, C, T, H, W), returns 16, 192, 8, 14, 14

    # torch.Size([16, 1568, 192]), i.e. (Batch, 1568 is 8*14*14, 192 is the channel number )
    print(y.shape)
    print(y)
