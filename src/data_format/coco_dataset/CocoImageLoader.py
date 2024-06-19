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
    '''

    def __init__(self, config, train=False, transform=None, real_job=False):
        self.config = config
        self.train = train
        self.transform = transform
        self.real_job = real_job  # Use get method to handle missing key

        dir_name = 'images/train2017'
        if not self.real_job:
            dir_name = 'images/test2017'
        # Set up paths
        self.data_dir = config['data_dir']
        self.image_dir = os.path.join(self.data_dir, dir_name if self.train else 'images/val2017')
        self.annotation_file = os.path.join(self.data_dir, 'annotations', 'person_keypoints_train2017.json' if self.train else 'person_keypoints_val2017.json')

        # Initialize COCO API
        self.coco = COCO(self.annotation_file)
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # Get image ID
        image_id = self.image_ids[index]

        # Load image
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        # Load keypoints and bounding boxes
        annotation_ids = self.coco.getAnnIds(imgIds=image_info['id'], iscrowd=False)
        annotations = self.coco.loadAnns(annotation_ids)

        # Initialize keypoints and bounding box array
        keypoints = np.zeros((17, 3))  # COCO keypoints: 17 keypoints with (x, y, v)
        bbox = np.zeros((4,))  # Bounding box: [x, y, width, height]

        for annotation in annotations:
            if 'keypoints' in annotation:
                keypoints = np.array(annotation['keypoints']).reshape((17, 3))
                bbox = np.array(annotation['bbox'])  # Extract bounding box
                break  # Assume one person per image for simplicity

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        bbox = torch.tensor(bbox, dtype=torch.float32)

        # Since I only want the x, y values, and not the visibility flag
        keypoints = keypoints[:, :-1]

        return image, keypoints, bbox

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

if __name__ == '__main__':
    # Example config dictionary
    config = {
        'data_dir': '/home/xinleilin/Projects/Video_Pose/data/COCO-Pose/coco',
    }

    # Initialize dataset and dataloader
    train_dataset = COCOLoader(config, train=False, transform=get_transforms())
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

    # Iterate through the data
    for images, keypoints, bboxes in train_loader:
        print(images)
        print(keypoints)
        print(bboxes)
        print(images.shape, keypoints.shape, bboxes.shape)
        break
