'''
Inefficient to linearize the channels, and then to regress.
    Instead, want to predit loss on the heatmap.
Also for x, y (not z) (because 3D pose detection is more complex : https://wham.is.tue.mpg.de/)


2 solutions:
1. Either use the heatmap, and directly compute loss. But that would involve transforming the dataset into heatmaps (then taking the mse between each pixel)
2. Use a joint regressor from mmpose, or whatever, but problem is that if its hardcoded, then cannot apply loss.  --> see code sent by soroush.
'''

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


class JointOutput(nn.Module):
    """
    Return the joint output from the Heatmap 

    Similar to deep pose regression:
    https://github.com/ViTAE-Transformer/ViTPose/blob/d5216452796c90c6bc29f5c5ec0bdba94366768a/mmpose/models/heads/deeppose_regression_head.py#L13
    """

    def __init__(self,
                 input_channels=15,
                 joint_number=15,
                 #  d=16,
                 d=1, # remember that after the 2d deconvolution, I have removed the d layer!!!!
                 h=56,
                 w=56
                 ):
        super().__init__()

        # For example, in PyTorch, this method is used to define the layers of the network, such as convolutional layers, linear layers, activation functions, etc.
        # hence need to have the regressor in the initializer if it wants to be saved properly
        # * I need to verify the output layer size
        self.joint_number = joint_number
        self.input_channels = input_channels
        self.c, self.d, self.h, self.w = input_channels, d, h, w
        self.b = 16 # although could change later.
        self.regressor = self.regressors()
        # self.flatten = self.input_flatten()

    # update the shapes that are passed in
    def get_shape(self):
        if len(list(x.shape)) == 5:
            self.b, self.c, self.d, self.h, self.w = x.shape
        else:
            self.b, self.c, self.h, self.w = x.shape
            self.d = 1

    def input_flatten(self, x):
        # first get the shape of the input
        self.get_shape()

        # x has the following sizes: (16,17 channels, 8, 14, 14) --> The 192 channels were initiated from the patching
        # * I want each channel to be processed separately, as a whole. So flatten each layer.
        if len(list(x.size())) == 5:
            return rearrange(x, 'b c d h w -> (b c) (d h w)')  # rearrange
        else:
            return rearrange(x, 'b c h w -> (b c) (h w)')

    def regressors(self):
        # self.get_shape(x)

        # Assuming the input tensor x has shape (batch_size, input_size)
        input_size = self.d * self.h * self.w
        # need to verify the input size! although this is still hugeeeeeee
        layers = [nn.Linear(input_size, self.w),  # reduce one dimension
                  nn.ReLU(),
                  nn.Linear(self.w, 2)]  # I will return 3, which are the values for x, y, z
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_flatten(x)

        # ! unsure Apply regressors to all channels simultaneously
        # This will apply regressors to all channels at once

        # need to apply regressor to each channel
        output = self.regressor(x)

        # need to reshape the output
        output = rearrange(output, '(b c) o -> b c o', b=self.b, c=self.c)
        return output
