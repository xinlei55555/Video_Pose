'''
Using COCOeval to evaluate the pretrained model with Mamba.
'''
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

import torch 
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from einops import rearrange
import argparse
import sys
import os
import json
import numpy as np

# first import the dataset from data_format
from data_format.CocoVideoLoader import eval_COCOVideoLoader
from data_format.coco_dataset.CocoImageLoader import eval_COCOLoader
from import_config import open_config
from data_format.AffineTransform import denormalize_fn, inverse_process_joint_data

from models.heatmap.HeatVideoMamba import HeatMapVideoMambaPose
from models.HMR_decoder.HMRMambaPose import HMRVideoMambaPose
from models.MLP_only_decoder.MLPMambaPose import MLPVideoMambaPose
from models.HMR_decoder_coco_pretrain.HMRMambaPose import HMRVideoMambaPoseCOCO
from data_format.AffineTransform import box2cs
from inference.visualize_coco import visualize, load_model

import random

def visualize_frame(joint, frame, width, height, bbox, file_name='coco_pretrain_result'):
    """
    Visualize a single frame with joint annotations.
    
    Args:
        joint (torch.Tensor): Tensor containing joint coordinates.
        frame (torch.Tensor): Tensor containing frame data.
        width (int): Width of the frame.
        height (int): Height of the frame.
        bbox (torch.Tensor): Tensor containing bounding box coordinates.
        file_name (str): The name of the file to save the visualization.
    """
    visualize(pose_model, frame, joint, bbox, dataset_info, file_name)

def coco_mask_fn(joints, labels, masks):
    """
    Mask the necessary COCO style joint values.
    
    Args:
        joints (torch.Tensor): Predicted joint values of shape (B, J, 2).
        labels (torch.Tensor): Ground truth joint values of shape (B, J, 2).
        masks (torch.Tensor): Mask tensor of shape (B, J, 1).
    
    Returns:
        torch.Tensor, torch.Tensor: Masked joints and labels.
    """
    # Get the shape of the predicted tensor
    B, J, X = joints.shape

    # Create a boolean mask where mask == 0
    zero_mask = (masks != 0)

    # Expand the mask to match the shape of the last dimension
    zero_mask = zero_mask.unsqueeze(-1).expand(-1, -1, X)

    # Instead of in-place modification, create new tensors with masked values set to 0
    joints = joints * zero_mask.float()
    labels = labels * zero_mask.float()
    return joints, labels

# note: from the val dataset, seems that category_id is always 1
def tensor_to_coco(tensor, image_ids, image_sizes, bboxes, masks, category_id=1, score=1.0):
    """
    Transform a tensor of shape (B, 17, 2) into a COCO result object.
    
    Args:
        tensor (torch.Tensor): The input tensor of shape (B, 17, 2).
        image_ids (list): A list of image IDs with length B.
        image_sizes (np.array): The input image sizes of shape (B, 2) where (w, h).
        bboxes (np.array): The input bounding boxes of shape (B, 4) where 4 is (x, y, w, h).
        category_id (int): The category ID for the keypoints.
        score (float): The confidence score for the detection.
    
    Returns:
        list: A list of dictionaries representing COCO keypoints results.
    """
    B, num_keypoints, _ = tensor.shape
    results = []

    for i in range(B):
        keypoints = tensor[i].reshape(-1).tolist()
        keypoints_with_visibility = []

        # Not all keypoints are visible!!!
        for j in range(0, len(keypoints), 2):
            keypoints_with_visibility.extend([keypoints[j], keypoints[j+1], masks[i][j//2].item()])
        
        # Add the scale and center
        center, scale = box2cs(image_sizes[i], bboxes[i])

        result = {
            "image_id": image_ids[i].item(),  # THIS IS A TENSOR
            "category_id": category_id,
            "keypoints": keypoints_with_visibility,
            "score": score,
        }
        results.append(result)

    return results

def main(config):
    """
    Main function to run the evaluation process.
    
    Args:
        config (dict): Configuration dictionary.
    """
    # Load the mmpose model
    pose_model = init_model(config['config_file'], config['checkpoint_path'], device='cuda:0')
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get('dataset_info', None))

    # Load test data
    test_set = eval_COCOVideoLoader(config, train_set='test', real_job=True)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

    # Variables for storing results
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate through the test dataset
    for i, data in enumerate(test_loader):
        inputs, processed_joints, mask, image_id, image_size, bbox, initial_index = data
        inputs = inputs.to(device)

        # Inference
        pose_results, _ = inference_top_down_model(pose_model, inputs, bbox, format='xywh')
        results.extend(pose_results)

    # Evaluate
    eval_results = pose_model.evaluate(results, config['annotations_path'])
    print(f'mAP is {eval_results["mAP"]}')

if __name__ == '__main__':
    # argparse to get the file path of the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='heatmap/heatmap_beluga.yaml',
                        help='Name of the configuration file')
    args = parser.parse_args()
    config_file = args.config

    # import configurations:
    config = open_config(config_file)

    main(config)
