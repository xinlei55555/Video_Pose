"""Combining implementations from ViTPose (heatmap deconv) using Video MambaBlocks
https://github.com/ViTAE-Transformer/ViTPose
"""

from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
import HeatMapDeconvHead as hdh
import HeatmapVideoMamba as hvm
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

# import VideoMamba as vm


class Deconv(nn.Module):
    """
    This was inspired by the ViTPose deconvolution process
    https://github.com/ViTAE-Transformer/ViTPose/blob/d5216452796c90c6bc29f5c5ec0bdba94366768a/mmpose/models/heads/deconv_head.py#L12
    """

    def __init__(self):
        super().__init__()
        # !using ViTPose
        # self.deconv = hdh.DeconvHead(in_channels = 192, out_channels = 3)

        # I will try using mmcv
        # self.deconv = torch.nn.ConvTranspose3d(in_channels=192,
        #                                        out_channels=3,
        #                                        kernel_size=2)

        # * Initialize my layers with mmcv.cnn

        self.conv_layers = self.define_conv_layers()
        self.deconv_layers = self.define_deconv_layers()

    def prep_input(self, x):
        """Conv2d's input is of shape (N, C_in, H, W) 
        where N is the batch size as before, 
        C_in the number of input channels, 
        Depth input
        H is the height and 
        W the width of the image
        """
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=8, h=14, w=14)
        # x has the following sizes: (16, 8, 14, 14, 192 channels) --> The 192 channels were initiated from the patching
        return x

    def define_conv_layers(self,
                           num_conv_layers=1,
                           conv_channels=3, 
                           out_channels=17,
                           num_deconv_kernels=(4, 4, 4)):
        layers = []
        for i in range(num_conv_layers):
            layers.append(
                build_conv_layer(
                    dict(type='Conv3d'),
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=num_conv_kernels[i],
                    stride=1,
                    padding=(num_conv_kernels[i] - 1) // 2))
            layers.append(
                build_norm_layer(dict(type='BN'), conv_channels)[1])
            layers.append(nn.ReLU(inplace=True))
        # add a final output convolution
        layers.append(cfg=dict(type='Conv3d'),
                      in_channels=conv_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=0)

        if len(layers) > 1:
            return nn.Sequential(*layers)
        else:
            return layers[0]

    def _get_deconv_cfg(deconv_kernel):
        """
        This is inspired from the ViTPose Paper Heatmap
        Get configurations for deconv layers."""

        # defines the padding based on the size of the deconvolution kernel
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def define_deconv_layers(self,
                           num_layers=1,
                           deconv_channels=192, 
                           # this is defining the shape of the filter
                           num_filters=(81, 9, 3),

                           # the larger kernel size capture more information from neighbour
                           num_kernels=(4, 3, 2)):
        """The middle deconvolution layers"""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)
    

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv3d'),
                    in_channels=deconv_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    # stride=(2, 2, 2),
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))

            # updating the input channel to be the output of the previous channel
            deconv_channels = planes

        return nn.Sequential(*layers)


    def forward(self, x):
        # print(x)

        # preparing the output from the mamba model
        x = self.prep_input(x)
        
        # deconvolutions
        x = self.deconv_layers(x)

        # heatmap output, through convolutions
        x = self.conv_layers(x)
        return x


class HeatMapVideoMambaPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = hvm.videomamba_tiny()

        self.deconv = Deconv()

    def forward(self, x):
        x = self.mamba(x)
        print('before', x.shape)
        x = self.deconv(x)
        # print(x)

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
