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
import models.latent_HMR.HMRVideoMamba as hvm
import models.latent_HMR.HMRJointRegressor as hjr

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class HMRVideoMambaPose(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_height = self.config['image_tensor_height']
        self.img_width = self.config['image_tensor_width']

        # encoder
        self.mamba = hvm.videomamba_tiny(img_size=(self.img_width, self.img_height),
                                         patch_size=self.config['patch_size'], embed_dim=self.config['embed_channels'], num_frames=self.config['num_frames'])

        # number of patches (dimensions)
        num_patch_height = int(
            self.config['image_tensor_height'] / self.config['patch_size'])  # 192 / 16 = 12
        num_patch_width = int(
            self.config['image_tensor_width'] / self.config['patch_size'])  # 256 / 16 = 16

        # output into joints
        self.joints = hjr.JointOutput(self.config, input_channels=self.config['joint_number'], joint_number=self.config['joint_number'],
                                      d=self.config['num_frames'], h=num_patch_height, w=num_patch_width, normalize=self.config['normalized'])  # for the JHMBD database

    def forward(self, x):
        # Prints GPU memory summary
        if self.config['full_debug']:
            # Prints GPU memory summary
            print('Memory before (in MB)', torch.cuda.memory_allocated()/1e6)
            # this prints Here is the input format torch.Size([12, 3, 16, 224, 224])k
            print('Here is the input format', x.shape)

        # uses around 7gb of memory for tiny
        x = self.mamba(x)

        if self.config['full_debug']:
            print('Output of the mamba model', x.shape)

        x = self.joints(x)

        if self.config['full_debug']:
            print('Final shape', x.shape)
            # Prints GPU memory summary
            print('Memory after (in MB)', torch.cuda.memory_allocated()/1e6)

        return x
