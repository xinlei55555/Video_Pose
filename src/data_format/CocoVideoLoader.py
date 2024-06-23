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

from data_format.coco_dataset.CocoImageLoader import COCOLoader
from data_format.AffineTransform import preprocess_video_data

class COCOVideoLoader(Dataset):
    '''
    A dataset to load the COCO videos as images
    '''
    def __init__(self, config, train_set, real_job):
        self.image_data = COCOLoader(config, train_set, real_job=real_job)
        self.real_job = real_job
        # reduce the quantity of data
        # if not self.real_job:
            # the first 128 images
            # self.image_data = self.image_data[:256]
        self.config = config 
        self.frames_num = self.config['num_frames']
        self.tensor_height = self.config['image_tensor_height']
        self.tensor_width = self.config['image_tensor_width']
        self.min_norm = self.config['min_norm']

    def __len__(self):
        # you want to remove some of the information given that 
        return (len(self.image_data) // self.frames_num)
        
    def __getitem__(self, index):
        vid_index = index * self.frames_num

        # slicing to get the video
        video, joints, bboxes = [], [], []
        for idx in range(vid_index, vid_index+self.frames_num):
            image, joint, bbox = self.image_data[idx]
            # making them all batch size = 1
            # image = image.unsqueeze(0)
            image = rearrange(image, '(d c) h w -> d h w c', d=1)
            joint = joint.unsqueeze(0)
            bbox = bbox.unsqueeze(0)
            # some of the bbox have width, and height 0!!!! that means there is nothing in it... (so let me just ignore them in COCOImageLoader)
            image, joint = preprocess_video_data(image.numpy(), bbox.numpy(), joint.numpy(), (self.tensor_width, self.tensor_height), self.min_norm)
            video.append(image[0])
            joints.append(joint[0])
            # bboxes.append(bbox[0])
        video = torch.stack(video)
        joints = torch.stack(joints)
        video = rearrange(video, 'd c h w->c d h w')

        return [video, joints]
        

if __name__ == '__main__':
    main()