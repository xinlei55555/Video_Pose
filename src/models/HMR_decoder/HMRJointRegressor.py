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

    # remember that after the 2d deconvolution, I have removed the d layer!!!!
    def __init__(self, config, input_channels, joint_number, d, h, w, normalize=True):
        super().__init__()
        # For example, in PyTorch, this method is used to define the layers of the network, such as convolutional layers, linear layers, activation functions, etc.
        # hence need to have the regressor in the initializer if it wants to be saved properly
        self.config = config

        self.joint_number = joint_number
        self.input_channels = input_channels
        self.c, self.d, self.h, self.w = input_channels, d, h, w
        self.dim = self.d * self.h * self.w  # could change later.
        self.c = self.config['embed_channels']
        self.b = self.config['batch_size']

        self.normalize = normalize
        self.dropout = self.config['dropout']

        # need to be defined in the __init__ so that it ignores in evaluation
        self.dropout_layer = nn.Dropout(self.config['dropout_percent'])

        self.regressor = self.regressors(
            dim_hidden=self.config['hidden_channels'], dim_out=self.joint_number * self.config['output_dimensions'])

    # update the shapes that are passed in
    # def get_shape(self, x):
    #     self.b, self.dim, self.c = x.shape
        # note: dim is supposed to be height x width x depth
        # c is supposed to be 192

    def input_flatten(self, x):
        # first get the shape of the input
        # self.get_shape(x)

        # mamba has the following output batch, (num_frames x heigt x width), channel_number
        return rearrange(x, 'b d c -> b (d c)')  # rearrange

    def regressors(self, dim_hidden, dim_out):
        # Assuming the input tensor x has shape (batch_size, input_size)
        input_size = self.dim * self.c

        layers = [nn.Linear(input_size, dim_hidden)]  # use power of 2

        # this applies dropout to all the layers except the last one.
        for _ in range(self.config['num_hidden_layers']):
            if self.dropout:
                layers.append(self.dropout_layer)
            # I will return 3, which are the values for x, y, z
            # here, my number of output dimensinos would be 30, then reshape
            layers.extend([nn.ReLU(), nn.Linear(dim_hidden, dim_hidden)])

        # output layer
        layers.extend([nn.ReLU(), nn.Linear(dim_hidden, dim_out)])
        if self.normalize:
            # restrict values at the end to be between -1 and 1
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x):
        '''Idea
        Merge batch and number of frames together.
        Then for each channel, run a crossattention layer. So similar to the 2D deconv
        then at the end run a joint output.
        '''
        x = self.input_flatten(x)

        # need to apply regressor to each channel. (will parallelize)
        output = self.regressor(x)

        # need to reshape the output
        output = rearrange(
            output, 'b (c o)-> b c o', c=self.joint_number, o=self.config['output_dimensions'])
        return output
