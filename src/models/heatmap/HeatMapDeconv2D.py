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

from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer

class Deconv(nn.Module):
    """
    This was inspired by the ViTPose deconvolution process
    https://github.com/ViTAE-Transformer/ViTPose/blob/d5216452796c90c6bc29f5c5ec0bdba94366768a/mmpose/models/heads/deconv_head.py#L12
    """

    def __init__(self, config, d, h, w, out_channels):
        super().__init__()
        self.config = config

        # h is the number of patches in the height, while w is the number of patches in the width
        self.d, self.h, self.w = d, h, w
        self.out_channels = out_channels

        self.conv_layers = self.define_conv_layers(
            self.config['num_conv'], self.config['conv_channels'], self.out_channels)
        self.deconv_layers = self.define_deconv_layers(
            self.config['num_deconv'], self.config['deconv_channels'])

    def prep_input(self, x):
        """Conv2d's input is of shape (N, C_in, D, H, W) 
        where N is the batch size as before, 
        C_in the number of input channels, 
        Depth input
        H is the height and 
        W the width of the image
        """
        if self.config['full_debug']:
            print('quick debug', self.d, self.h, self.w)
        
        # I want to combine batch and depth, for the 2d
        x = rearrange(x, 'b (d h w) c -> b d c h w',
                      d=self.d, h=self.h, w=self.w)

        # and I can just discard the depth, and keep the last layer of the mamba (at least for the 2D deconv)
        # Select the last element in the 'd' dimension
        if self.config['use_last_frame_only']:
            x = x[:, -1, :, :, :]  # x.shape now should be [batch, c, h, w]
        else:
            # new: will change it so that every frame gets a prediction
            x = rearrange(x, 'b d c h w -> (b d) c h w',
                      d=self.d, h=self.h, w=self.w)
        return x

    def define_conv_layers(self,
                           num_conv_layers=2,
                           # number of input channels
                           conv_channels=256,
                           out_channels=15,
                           num_conv_kernels=(1, 1, 1)):
        layers = []
        for i in range(num_conv_layers):
            layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=num_conv_kernels[i],
                    stride=1,
                    padding=(num_conv_kernels[i] - 1) // 2))
            layers.append(nn.BatchNorm2d(conv_channels))
            # layers.append(
            # build_norm_layer(dict(type='BN'), conv_channels)[1])
            layers.append(nn.ReLU(inplace=True))
        # add a final output convolution
        layers.append(
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=conv_channels,
                out_channels=out_channels,
                kernel_size=num_conv_kernels[0],
                # stride of 1, to go through all
                stride=1,
                padding=0))

        if len(layers) > 1:
            return nn.Sequential(*layers)
        else:
            return layers[0]

    def _get_deconv_cfg(self, deconv_kernel):
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

    # in my case my deconv layers are doubling the size of the images.
    def define_deconv_layers(self,
                             num_layers=2,
                             # I start with 192 channels, which comes from the patching operations at the beginning of mamba (which linearized the data)
                             # Note that for ViTPose, there were 3 channels, that's because ViTPose still works with a VisionTransformers, but deconvolves the data first.
                             deconv_channels=192,
                             # this is defining the shape of the filter
                             #    num_filters=(81, 49, 21),
                             # * I want to deconv to the correct number of channels
                             num_filters=(256, 256),

                             # the larger kernel size capture more information from neighbour
                             #    num_kernels=(4, 3, 2)):
                             #! larger kernel sizes
                             num_kernels=(4, 4)):
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
            kernel, padding, output_padding = self._get_deconv_cfg(
                num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=deconv_channels,
                    out_channels=planes,
                    # * I am defining the kernel_size to be three times the size, so that it performs deconvolution within the video too.
                    # here, i was using 8 as the kernel size for the 3d thingy. Very bad idea, since d is a huge number.
                    kernel_size=(kernel, kernel),
                    stride=(2, 2),
                    # stride=(kernel, kernel, kernel),
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
        # print("before convolution (after deconvolution)", x.shape)

        # heatmap output, through convolutions
        x = self.conv_layers(x)
        return x
