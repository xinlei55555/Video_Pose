'''
Inefficient.
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
                 input_channels=17,
                 joint_number=17):
        super().__init__()
        # * I need to verify the output layer size
        self.joint_number = joint_number
        self.input_channels = input_channels

    def get_shape(self, x):
        self.b, self.c, self.d, self.h, self.w = x.shape

    def input_flatten(self, x):
        # x has the following sizes: (16,17 channels, 8, 14, 14) --> The 192 channels were initiated from the patching
        # * I want each channel to be processed separately, as a whole. So flatten each layer.
        if len(list(x.size())) == 5:
            return rearrange(x, 'b c d h w -> c b (d h w)')  # rearrange
        else:
            return rearrange(x, 'c d h w -> c (d h w)')

    def regressors(self):
        # Assuming the input tensor x has shape (batch_size, input_size)
        input_size = self.d * self.h * self.w
        # need to verify the input size! although this is still hugeeeeeee
        layers = [nn.Linear(input_size, self.w), # reduce one dimension
                  nn.ReLU(),
                  nn.Linear(self.w, 3)]  # I will return 3, which are the values for x, y, z
        return nn.Sequential(*layers)

    def define_regressor(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.regressor = self.regressors()

        self.regressor = self.regressor.to(device)

    def forward(self, x):
        self.get_shape(x)
        self.define_regressor()

        x = self.input_flatten(x)

        # ! unsure Apply regressors to all channels simultaneously
        # This will apply regressors to all channels at once
        output = self.regressor(x)

        # # Reshape output to have shape (batch_size, input_channels, 2)
        # output = output.view(x.size(0), self.input_channels, -1)

        return x
