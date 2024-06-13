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
from models.HMR_decoder.utils.pose_transformer import TransformerDecoder

class Mamba_HMR_decoder(nn.Module):
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

        # number of tokens

        # token_dim is the dimension of each token, in my case, its patch_size * patch_size
        self.token_dim = 

        # dim

        self.transformer = TransformerCrossAttn(
            num_tokens, 
            token_dim,
            dim,
            depth,
            heads,
            mlp_dim,
            dim_head: int = 64,
            dropout: float = 0.0,
            emb_dropout: float = 0.0,
            emb_dropout_type: str = 'drop',
            norm: str = "layer",
            norm_cond_dim: int = -1,
            context_dim: Optional[int] = None,
            skip_token_embedding: bool = False,
        )

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
            print("This format has is not supported for the HMR decoder")
            raise NotImplementedError
        else:
            # new: will change it so that every frame gets a prediction
            x = rearrange(x, 'b d c h w -> (b d) c h w',
                      d=self.d, h=self.h, w=self.w)
        return x

    

    def forward(self, x):
        # print(x)

        # preparing the output from the mamba model
        x = self.prep_input(x)

        # transformer block
        x = self.transformer(x)

        
        return x
