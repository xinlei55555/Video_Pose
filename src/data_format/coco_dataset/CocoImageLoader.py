'''
Updated dataloader with using the id of the images, instead of the image id
'''
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


class COCOLoader(Dataset):
    '''
    A dataset loader for COCO dataset.

    Attributes:
        self.config (dict): configuration file
        self.train (string): type of dataset, train, val or test
        self.transform (func): given image transformation to apply
        self.real_job (bool): whether to take a subset of the data
        self.data_dir (string): data directory
        self.anno_file (string): full path of the annotation file
        self.image_dir (string): data of the images)
        self.coco (COCO): coco API object
        self.data (list[dict]): contains information on each datapoint
    '''

    def __init__(self, config, train='train', real_job=False):
        self.config = config
        self.train = train
        self.transform = get_transforms()
        self.real_job = real_job  # Use get method to handle missing key

        # Set up paths
        self.data_dir = config['data_path']

        if train == 'train' or train == 'val':
            dir_name = f'images/{self.train}2017'
            self.anno_file = os.path.join(
                self.data_dir, 'annotations', f'person_keypoints_{self.train}2017.json')

        # if using testing... then i'll just pick val lol
        if train == 'test':
            dir_name = f'images/val2017'
            self.anno_file = os.path.join(
                self.data_dir, 'annotations', f'person_keypoints_val2017.json')

        if not self.real_job:
            dir_name = 'images/val2017'
            self.train = 'val'

        self.image_dir = os.path.join(self.data_dir, dir_name)

        # Initialize COCO API
        self.coco = COCO(self.anno_file)

        # default value for pose detection
        PERSON_CAT_ID = 1
        person_ann_ids = self.coco.getAnnIds(catIds=[PERSON_CAT_ID])
        person_anns = self.coco.loadAnns(ids=person_ann_ids)

        self.data = []
        # loop through each person id.
        for person_ann in person_anns:
            if person_ann['num_keypoints'] > 0:
                self.data.append({
                    "image_id": person_ann['image_id'],
                    "category_id": PERSON_CAT_ID,
                    'keypoints': person_ann['keypoints'],
                    'id': person_ann['id'],
                    'bbox': person_ann['bbox']
                })
        print(
            f"The length of the: {self.train} dataset is : {len(self.data)} divided by the number of frames")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get actual image ID
        person_id = self.data[index]['id']
        image_id = self.data[index]['image_id']

        # Load image, from the given image_id
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        # Apply transformations to become tensor
        image = self.transform(image)

        # reshaping the tensor
        keypoints = torch.tensor(
            self.data[index]['keypoints'], dtype=torch.float32).reshape((17, 3))

        bbox = torch.tensor(self.data[index]['bbox'], dtype=torch.float32)

        # Since I only want the x, y values, and not the visibility flag
        # Remove the last column and store it in mask
        mask = keypoints[:, -1]
        keypoints = keypoints[:, :-1]  # YES YOU WANT THE VISIBILITY TT

        return image, keypoints, bbox, mask


class eval_COCOLoader(COCOLoader):
    '''A dataformat class for the evaluation dataset of the image COCO Loader'''
    def __getitem__(self, index):
        image, keypoints, bbox, mask = super().__getitem__(index)
        return image, keypoints, bbox, mask, self.data[index]['image_id'], self.data[index]['id']

    def get_item_with_id(self, person_id):
        '''
        Returns the information for the given person_id

        Args:
            person_id (int): represents the id of the person in the keypoint values

        Returns:
            image, keypoints, bbox, mask
        '''
        # find the proper id in the data file
        idx = -1
        for el in range(len(self.data)):
            if self.data[el]['id'] == person_id:
                idx = el
                break
    
        # if id was not found.
        if idx == -1:
            print("Error, the person id was not found in the dataset")
            exit()
        return image, keypoints, bbox, mask

def get_transforms():
    # note that there are a lot of imagesssss sizesssss
    return transforms.Compose([
        # transforms.Resize((240, 320)), # resize all the images... but then what about the joints?
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


if __name__ == '__main__':
    # Example config dictionary
    config = {
        'data_dir': '/home/xinleilin/Projects/Video_Pose/data/COCO-Pose/coco',
    }

    # Initialize dataset and dataloader
    train_dataset = COCOLoader(config, train='val', transform=get_transforms())
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

    # Iterate through the data
    for images, keypoints, bboxes in train_loader:
        print(images)
        print(keypoints)
        print(bboxes)
        print(images.shape, keypoints.shape, bboxes.shape)
        break
