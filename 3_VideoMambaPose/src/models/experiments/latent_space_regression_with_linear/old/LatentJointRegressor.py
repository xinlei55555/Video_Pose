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
                 input_channels=17,
                 joint_number=17): # note: for JHMDB, need to change to 15
        super().__init__()
        # * I need to verify the output layer size
        self.joint_number = joint_number
        self.input_channels = input_channels

    def get_shape(self, x):
        # since we are directly regressing from the mamba model, it should be b (d h w) channel
            # batch, size (dxhxw) and channel
        self.b, self.s, self.c = x.shape

    def input_flatten(self, x):
        # x has the following sizes: (16,17 channels, 8, 14, 14) --> The 192 channels were initiated from the patching
        # * I want each channel to be processed separately, as a whole. So flatten each layer.
        return rearrange(x, 'b s c -> (b) (s c)')  # rearrange

    def regressors(self):
        # Assuming the input tensor x has shape (batch_size, input_size)
        input_size = self.s * self.c
        # added one linear layer, because seemed to be too few.
        layers = [nn.Linear(input_size, self.s), # reduce one dimension
                  nn.ReLU(),
                  nn.Linear(self.s, self.c),
                  nn.ReLU(),
                  nn.Linear(self.c, 2*self.joint_number)]  # I will return 2*num_joints, so that there is x, y for each num_joints.
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

        # need to apply regressor to each channel
        output = self.regressor(x)

        # need to reshape the output
        output = rearrange(output, '(b) (c d) -> b c d', d=2, c=self.joint_number) # so that it becomes like the output of the JHMDB dataset
        return output
