import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

from data_format.coco_dataset.CocoImageLoader import COCOLoader
from data_format.AffineTransform import preprocess_video_data

class COCOVideoLoader(Dataset):
    '''
    A dataset to load the COCO videos as images
    '''
    def __init__(self, config, train_set, real_job):
        self.image_data = COCOLoader(config, train_set, real_job=real_job)
        self.config = config 
        self.frames_num = self.config['num_frames']
        self.tensor_height = self.config['image_tensor_height']
        self.tensor_width = self.config['image_tensor_width']
        self.min_norm = self.config['min_norm']
    def __len__(self):
        return (len(self.image_data) // self.frames_num)
    
    def __getitem__(self, index):
        vid_index = index * self.frames_num

        # slicing to get the video
        video, joints, bboxes = [], [], []
        for idx in range(vid_index, vid_index+self.frames_num):
            image, joint, bbox = self.image_data[idx]
            video.append(images)
            joints.append(joint)
            bboxes.append(bbox)
        video = torch.stack(video)
        joints = torch.stack(joints)
        bboxes = torch.stack(bbox)

        # apply transformation on the video
        video, joints = preprocess_video_data(video.numpy(), bboxes.numpy(), joints.numpy(), (self.tensor_width, self.tensor_height), self.min_norm)

        return [video, joints]
        

if __name__ == '__main__':
    main()