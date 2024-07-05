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
        self.image_ids = self.coco.getImgIds()

        self.new_image_ids = []
        # removing all the bboxes that are torch.tensor([0., 0., 0., 0.])
        for index in range(len(self.image_ids)):
            if not real_job:
                # I will only work with 2 videos
                if len(self.new_image_ids) >= 32:
                    print(f'The image_ids chosen are the {self.new_image_ids}')
                    break
            image_id = self.image_ids[index]
            image_info = self.coco.loadImgs(image_id)[0]
            annotation_ids = self.coco.getAnnIds(
                imgIds=image_info['id'], iscrowd=False)
            annotations = self.coco.loadAnns(annotation_ids)

            bbox = np.zeros((4,))  # Bounding box: [x, y, width, height]
            for annotation in annotations:
                if 'keypoints' in annotation:
                    # keypoints = np.array(annotation['keypoints']).reshape((17, 3))
                    bbox = np.array(annotation['bbox'])  # Extract bounding box
                    break  # Assume one person per image for simplicity

            # skip when bbox is empty
            if bbox[0] == bbox[1] == bbox[2] == bbox[3] == 0.:
                continue
            else:
                self.new_image_ids.append(index)

        print(
            f"The length of the: {self.train} dataset is : {len(self.new_image_ids)} divided by the number of frames")

    def __len__(self):
        return len(self.new_image_ids)

    def __getitem__(self, index):
        # Get actual image ID
        image_id = self.image_ids[self.new_image_ids[index]]

        # Load image
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        # Load keypoints and bounding boxes
        annotation_ids = self.coco.getAnnIds(
            imgIds=image_info['id'], iscrowd=False)
        annotations = self.coco.loadAnns(annotation_ids)

        # Initialize keypoints and bounding box array
        # COCO keypoints: 17 keypoints with (x, y, v)
        keypoints = np.zeros((17, 3))
        bbox = np.zeros((4,))  # Bounding box: [x, y, width, height]

        for annotation in annotations:
            if 'keypoints' in annotation:
                keypoints = np.array(annotation['keypoints']).reshape((17, 3))
                bbox = np.array(annotation['bbox'])  # Extract bounding box
                break  # Assume one person per image for simplicity

        # # Skip annotations with zero bounding box
        # while np.array_equal(bbox, np.zeros((4,))):
        #     index = (index + 1) % len(self.image_ids)
        #     continue

        # Apply transformations to become tensor
        image = self.transform(image)

        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        bbox = torch.tensor(bbox, dtype=torch.float32)

        # Since I only want the x, y values, and not the visibility flag
        # Remove the last column and store it in mask
        mask = keypoints[:, -1]
        keypoints = keypoints[:, :-1]  # YES YOU WANT THE VISIBILITY TT

        return image, keypoints, bbox, mask


class eval_COCOLoader(COCOLoader):
    '''A dataformat class for the evaluation dataset of the image COCO Loader'''

    def get_item_with_id(self, image_id):
        '''
        Returns the information for the given image_id

        Args:
            image_id (int)

        Returns:
            image, keypoints, bbox, mask, image_id 
        '''
        # Load image
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        # Load keypoints and bounding boxes
        annotation_ids = self.coco.getAnnIds(
            imgIds=image_info['id'], iscrowd=False)
        annotations = self.coco.loadAnns(annotation_ids)

        # Initialize keypoints and bounding box array
        # COCO keypoints: 17 keypoints with (x, y, v)
        keypoints = np.zeros((17, 3))
        bbox = np.zeros((4,))  # Bounding box: [x, y, width, height]

        for annotation in annotations:
            if 'keypoints' in annotation:
                keypoints = np.array(annotation['keypoints']).reshape((17, 3))
                bbox = np.array(annotation['bbox'])  # Extract bounding box
                break  # Assume one person per image for simplicity

        # # Skip annotations with zero bounding box
        # while np.array_equal(bbox, np.zeros((4,))):
        #     index = (index + 1) % len(self.image_ids)
        #     continue

        # Apply transformations to become tensor
        image = self.transform(image)

        original_size = torch.tensor([image.shape[2], image.shape[1]])  # Assuming the original size is (height, width)

        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        bbox = torch.tensor(bbox, dtype=torch.float32)

        # Since I only want the x, y values, and not the visibility flag
        # Remove the last column and store it in mask
        mask = keypoints[:, -1]
        keypoints = keypoints[:, :-1]  # YES YOU WANT THE VISIBILITY TT

        return image, keypoints, bbox, mask, original_size

    def __getitem__(self, index):
        # calling the __getitem__ from the parent class
        # captures the value from the parent class
        image, keypoints, bbox, mask = super().__getitem__(index)

        # redefining the missing variables.
        image_id = self.image_ids[self.new_image_ids[index]]

        # then adds lines
        return image, keypoints, bbox, mask, image_id


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
