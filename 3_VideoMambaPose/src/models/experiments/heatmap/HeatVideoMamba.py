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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class HeatMapVideoMambaPose(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.mamba = hvm.videomamba_tiny()

        # decoder into heatmap
        # self.deconv = hmd.Deconv(4, 4, 4) 
        # self.deconv = hmd3D.Deconv(64, 14, 14)
        # 16 batch size.
        self.deconv = hmd2D.Deconv(16, 14, 14, 15) # Shape of final tensor torch.Size([64, 17, 112, 112])
                # 15 is the number of output joints

        # output into joints
        # remember that the output layer has been replaced by 1 for deconv!!!
        self.joints = hjr.JointOutput(input_channels = 15, joint_number=15, d=1, h=56, w=56) # for the JHMBD database

    def forward(self, x):
        # print('Memory before (in MB)', torch.cuda.memory_allocated()/1e6)  # Prints GPU memory summary
        # print('Here is the input format', x.shape) # this prints Here is the input format torch.Size([12, 3, 16, 224, 224])k
        x = self.mamba(x) # uses around 7gb of memory for tiny

        # print('Output of the mamba model, before the deconvolution', x.shape)
        x = self.deconv(x)
        
        # print(self.deconv)
        # the shape of this is a bit too big after the convolutions.
        # print('After deconvolution', x.shape)

        # this should parallelize and apply it to each channel separately
        x = self.joints(x)
        # print('Final shape', x.shape)
        # print('Memory after (in MB)', torch.cuda.memory_allocated()/1e6)  # Prints GPU memory summary
        return x


if __name__ == "__main__":

    # generating a random input
    # (Batch, Channel number, NumFrames, W, H) = (16, 3, 8, 224, 224)

    # Define the dimensions
    batch_size = 1 # ! can train on cedar with batch size of 32 (no more)
    num_frames = 64 # I did 16 memory usage is linearly growing with the input size.
    height = 224 # *Note: VitPose doesn't use square, to reduce number of pixels. (bounding box with YolO)
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
