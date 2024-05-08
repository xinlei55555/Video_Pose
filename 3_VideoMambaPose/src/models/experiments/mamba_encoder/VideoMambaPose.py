#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
os.getcwd()


# In[15]:


os.chdir("/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/mamba_encoder")
os.getcwd()


# In[16]:


# get_ipython().system('pip list')


# In[17]:


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


# In[19]:


# generating a random input
# (Batch, Channel number, NumFrames, W, H) = (16, 3, 8, 224, 224)

# Define the dimensions
batch_size = 16
num_frames = 8
height = 224
width = 224
channels = 3

# Generate a random tensor
test_video = torch.rand(batch_size, channels, num_frames, height, width) # I get an error .... 384, 3, 1, 16, 16

# Check the shape of the random tensor
print("Shape of the random tensor:", test_video.shape)


# #### running the video preprocessing

# #### running the mamba model

# In[21]:


import VideoMamba as vm


# In[22]:


class VideoMambaPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = vm.videomamba_tiny() # TODO this is temporary 

    def forward(self, x):
        x = self.encoder(x)

        # adding my own layer, see how deciwatch did it.
        print(x.shape)


# In[24]:


test_model = VideoMambaPose()

y = test_model(test_video)

