import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

from einops import rearrange

from data_format.coco_dataset.CocoImageLoader import COCOLoader, eval_COCOLoader
# from data_format.eval_Cocoloader import eval_COCOLoader
from data_format.AffineTransform import preprocess_video_data, data_augment, normalize_fn

from data_format.coco_dataset.CocoImageLoader import COCOLoader, eval_COCOLoader
from einops import rearrange
import torch

import random

class COCOVideoLoader(Dataset):
    '''
    A dataset to load the COCO videos as images
    '''
    def __init__(self, config, train_set, real_job):
        if train_set != 'test':
            self.image_data = COCOLoader(config, train_set, real_job=real_job)
        if train_set == 'test':
            self.image_data = eval_COCOLoader(config, train_set, real_job=real_job)
        self.real_job = real_job
        
        self.config = config 
        self.frames_num = self.config['num_frames']
        self.tensor_height = self.config['image_tensor_height']
        self.tensor_width = self.config['image_tensor_width']
        self.min_norm = self.config['min_norm']

    def __len__(self):
        return (len(self.image_data)) # nevermind, I make each image a video #// self.frames_num)
        
    def __getitem__(self, index):
        image, joint, bbox, mask = self.image_data[index]
        
        image = rearrange(image, '(d c) h w -> d h w c', d=1)
        joint = joint.unsqueeze(0)
        bbox = bbox.unsqueeze(0)

        # apply rotation data augmentation if needed:
        # https://github.com/ViTAE-Transformer/ViTPose/blob/main/mmpose/datasets/pipelines/top_down_transform.py#L147
        rotation = 0
        if self.train_set and config['data_augmentation']['rotation'] > 0 and random.uniform(0, 1) > config['data_augmentation']['rotation']:
            rotation = config['rotation_val']
        image, joint = preprocess_video_data(image.numpy(), bbox.numpy(), joint.numpy(), (self.tensor_width, self.tensor_height), rotation)

        # perform image data augmentation on train_set, before nromalizing the joint values.
        if self.train_set:    
            image, joint = data_augment(config['data_augmentation'], image, joint, bbox, (self.tensor_width, self.tensor_height), config['flip_types'])

        # normalize the values
        joint = normalize_fn(joint, self.min_norm, self.tensor_height, self.tensor_width)

        # technically, I have depth = 1... do it's like a one frame video.
        image = rearrange(image, 'd c h w -> c d h w')

        # check if all the joint values are between -1 and 1
        if self.config['full_debug'] and not torch.all((joint >= -1) & (joint <= 1)):
            print("Error, some of the normalized values are not between -1 and 1")

        return [image, joint, mask]
        

class eval_COCOVideoLoader(COCOVideoLoader):
    '''A dataformat class for the evaluation dataset of the COCO dataset
    '''
    def __getitem__(self, index):
        initial_image, joint, bbox, mask, image_id, keypoint_id = self.image_data[index]
        image = initial_image.detach().clone() # okay, instead of passing the initial image, i'll pass the index
        # width, heightf
        original_size = torch.tensor([image.shape[2], image.shape[1]])  # Assuming the original size is (height, width)

        # making them all batch size = 1
        # image = image.unsqueeze(0)
        image = rearrange(image, '(d c) h w -> d h w c', d=1)
        joint = joint.unsqueeze(0)
        bbox = bbox.unsqueeze(0)

        processed_image, joint = preprocess_video_data(image.numpy(), bbox.numpy(), joint.numpy(), (self.tensor_width, self.tensor_height))

        # perform image data augmentation, before nromalizing the joint values.
        # image, joint = data_augment(image, joint, bbox, (self.tensor_width, self.tensor_height))

        # normalize the values
        joint = normalize_fn(joint, self.min_norm, self.tensor_height, self.tensor_width)

        processed_image = rearrange(processed_image, 'd c h w -> c d h w')

        # check if all the joint values are between -1 and 1
        if self.config['full_debug'] and not torch.all((joint >= -1) & (joint <= 1)):
            print("Error, some of the normalized values are not between -1 and 1")
        
        return processed_image, joint, mask, image_id, original_size, bbox, index, keypoint_id

if __name__ == '__main__':
    main(config)