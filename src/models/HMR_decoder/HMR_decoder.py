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

        # number of tokens is used to create the positional encoding. 
        self.num_tokens = 1 # this is what HMR smpl_head did...
        # token_dim is the dimension of each token, in my case, its patch_size * patch_size
        # token dimension is the number of tokens that are being inputted, i.e the batch _size * the number of frames
        self.token_dim = 1#self.config['batch_size'] * self.d #self.h * self.w * self.config['embed_channels'] # self.config['path_size'] ** 2.0 
        #! although this batch_size sometimes changes...
        # dim is the embedding dimensions for the transformers (also how many neurons in the intput layer of transformers)
            # * note the same as dim_head.
        # ! dim is the output dimension of the transformers.
        self.dim = self.config['dim'] # not sure about this value, but I think if context dim is not given, then it is equal to dim. So should maybe be the output.
        self.depth = self.config['depth']
        self.heads = self.config['heads'] # number of heads
        self.dim_head = self.config['dim_head']  # dimension of each crossattention head.
        self.mlp_dim = self.config['mlp_dim']
        self.dropout = self.config['dropout_transformer']
        # self.context_dim = self.h * self.w
        self.context_dim = self.config['embed_channels'] # context dim is the number of channels that Mamba outputs

        self.transformer = TransformerDecoder(
            num_tokens=self.num_tokens, 
            token_dim=self.token_dim,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            dim_head=self.dim_head,
            dropout=self.dropout,
            emb_dropout=0.0,
            emb_dropout_type='drop',
            norm="layer",
            norm_cond_dim=-1,
            context_dim=self.context_dim,
            skip_token_embedding=False,
        )

    def prep_input(self, x):
       
        if self.config['full_debug']:
            print('quick debug', self.d, self.h, self.w)
        
        # I want to combine batch and depth, for the 2d
        # Change to token-first (unlike the Deconv layers that Vit Uses)
        x = rearrange(x, 'b (d h w) c -> b d h w c',
                      d=self.d, h=self.h, w=self.w)

        # and I can just discard the depth, and keep the last layer of the mamba (at least for the 2D deconv)
        # Select the last element in the 'd' dimension
        if self.config['use_last_frame_only']:
            x = x[:, -1, :, :, :]  # x.shape now should be [batch, c, h, w]
            print("This format has is not supported for the HMR decoder")
            raise NotImplementedError
        else:
            # new: will change it so that every frame gets a prediction
            x = rearrange(x, 'b d h w c -> (b d) (h w) c',
                      d=self.d, h=self.h, w=self.w)
        return x

    

    def forward(self, x):
        # print(x)

        # preparing the output from the mamba model
        x = self.prep_input(x)

        batch_size = x.shape[0] # this is b * d
        input_token = torch.zeros(batch_size, 1, 1).to(x.device) # just a learnable token!

        # transformer block
        output_tokens = self.transformer(input_token, context=x)

        return output_tokens
