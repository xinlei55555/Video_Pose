import torch 
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from mmcv.runner import load_checkpoint
from mmpose.apis import init_model, inference_top_down_model, vis_pose_result
from mmpose.datasets.dataset_info import DatasetInfo
from data_format.eval_Cocoloader import eval_COCOVideoLoader
from data_format.AffineTransform import denormalize_fn, inverse_process_joint_data, box2cs

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
    vis_pose_result(pose_model, frame, joint, bbox, dataset_info, file_name)

def tensor_to_coco(tensor, image_ids, image_sizes, bboxes, category_id=1, score=1.0):
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
        for j in range(0, len(keypoints), 2):
            keypoints_with_visibility.extend([keypoints[j], keypoints[j+1], 2])  # Assuming all keypoints are visible
        
        # Add the scale and center
        center, scale = box2cs(image_sizes[i], bboxes[i])

        result = {
            "image_id": image_ids[i].item(),  # THIS IS A TENSOR
            "category_id": category_id,
            "keypoints": keypoints_with_visibility,
            "score": score,
            "center": center.tolist(),
            "scale": scale.tolist(),
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
