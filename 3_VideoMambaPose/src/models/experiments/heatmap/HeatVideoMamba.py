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
import HeatMapDeconv2D as hmd2D
import HeatMapDeconv3D as hmd3D
import HeatMapJointRegressor as hjr


class HeatMapVideoMambaPose(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # encoder
        self.mamba = hvm.videomamba_tiny(patch_size=self.config['patch_size'], embed_dim=self.config['embed_channels'], num_frames=self.config['num_frames'])

        # patch sizes are image_size/patch_number
        patch_height = self.config['image_tensor_height'] / \
            self.config['patch_number']
        patch_width = self.config['image_tensor_width'] / \
            self.config['patch_number']

        if config['2d_deconv']:
            self.deconv = hmd2D.Deconv(
                self.config, self.config['num_frames'], patch_height, patch_width, self.config['joint_number'])
        else:
            self.deconv = hmd3D.Deconv(
                self.config, self.config['num_frames'], patch_height, patch_height)
            # 15 is the number of output joints

        # output into joints
        self.joints = hjr.JointOutput(
            self.config, input_channels=self.config['joint_number'], joint_number=self.config['joint_number'], d=1, h=56, w=56, normalize=self.config['normalized'])  # for the JHMBD database

    def forward(self, x):
        if self.config['full_debug']:
            # Prints GPU memory summary
            print('Memory before (in MB)', torch.cuda.memory_allocated()/1e6)
            # this prints Here is the input format torch.Size([12, 3, 16, 224, 224])k
            print('Here is the input format', x.shape)

        # uses around 7gb of memory for tiny
        x = self.mamba(x)

        if self.config['full_debug']:
            print('Output of the mamba model, before the deconvolution', x.shape)

        x = self.deconv(x)

        # the shape of this is a bit too big after the convolutions.
        if self.config['full_debug']:
            print('After deconvolution', x.shape)

        x = self.joints(x)

        if self.config['full_debug']:
            print('Final shape', x.shape)
            # Prints GPU memory summary
            print('Memory after (in MB)', torch.cuda.memory_allocated()/1e6)

        return x


if __name__ == "__main__":

    # generating a random input
    # (Batch, Channel number, NumFrames, W, H) = (16, 3, 8, 224, 224)

    # Define the dimensions
    batch_size = 1  # ! can train on cedar with batch size of 32 (no more)
    # I did 16 memory usage is linearly growing with the input size.
    num_frames = 64
    # *Note: VitPose doesn't use square, to reduce number of pixels. (bounding box with YolO)
    height = 224
    width = 224
    channels = 3

    # Generate a random tensor
    # I get an error .... 384, 3, 1, 16, 16
    # test video for mamba
    test_video = torch.rand(batch_size, channels, num_frames, height, width)

    # this is the test video for the deconv
    # okay, let me not batch this lol
    # test_video = torch.rand(1, 1568, 192) #!this 1568 makes the whole video HUEGEE. which means my deconvolution isn't really working lmao.
    # test_video=torch.rand(1, 64, 192)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # with num_frames 64
    # test_video = torch.rand(1, 12544, 192)

    # this is the joint map regressor:
    # test_video = torhc.rand()

    # Check the shape of the random tensor
    print("Shape of the random tensor:", test_video.shape)

    test_model = HeatMapVideoMambaPose()

    # move the data to the GPU
    test_model = test_model.to(device)
    test_video = test_video.to(device)

    y = test_model(test_video)

    # * note: when I print (B, C, T, H, W), returns 16, 192, 8, 14, 14

    # torch.Size([16, 1568, 192]), i.e. (Batch, 1568 is 8*14*14, 192 is the channel number )
    print('Shape of final tensor', y.shape)
    print(y)
