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
    visualize(joint, frame, file_name, width, height, bbox, False)


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


def tensor_to_coco(tensor, image_ids, image_sizes, bboxes, masks, person_id, category_id=1, score=1.0):
    """
    Transform a tensor of shape (B, 17, 2) into a COCO result object.

    Args:
        tensor (torch.Tensor): The input tensor of shape (B, 17, 2).
        image_ids (list[torch.Tensor]): A list of image IDs with length B.
        image_sizes (np.array): The input image sizes of shape (B, 2) where (w, h).
        bboxes (np.array): The input bounding boxes of shape (B, 4) where 4 is (x, y, w, h).
        category_id (int): The category ID for the keypoints.
        score (float): The confidence score for the detection.
        person_id (list[torch.Tensor]): A list of the id for each keypoints

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
            keypoints_with_visibility.extend(
                [keypoints[j], keypoints[j+1], masks[i][j//2].item()])

        # Add the scale and center
        center, scale = box2cs(image_sizes[i], bboxes[i])

        result = {
            "image_id": image_ids[i].item(),  # THIS IS A TENSOR
            "category_id": category_id,
            "keypoints": keypoints_with_visibility,
            "score": score,
            "id": person_id[i].item()
        }
        results.append(result)

    return results


def evaluate_coco(dt_annotations, data, stats_name=None, sigmas=None, single_input=None, category_id=1):
    """
    Evaluate the model using COCO evaluation metrics.

    Args:
        dt_annotations (str): Annotation values, as defined in the tensor_to_coco function.
        data (str): Path to the ground truth annotations file.
        coco_data_object (COCOVideoLoader(torch.utils.data.Dataset)): Dataset object for the coco
        sigmas (np.array, optional): Keypoint sigmas for evaluation.
        single_input (int, optional): Specific image ID to evaluate.
        imgIds (list[torch.Tensor], optional): list of image ids

    Returns:
        list: Average Precision (mAP) for keypoints.
    """
    if sigmas is not None:
        print("Sigmas has not yet been implemented")
    # Create COCO objects
    cocoGt = COCO(data)

    with open("outputs/results.txt", "w") as f:
        f.write(json.dumps(dt_annotations))

    # Load the keypoints
    pk_res = cocoGt.loadRes("outputs/results.txt")

    # Define a default keypoint object, using the default sigmas, yet no areas
    eval_coco = COCOeval(cocoGt=cocoGt, cocoDt=pk_res, iouType='keypoints')

    # This runs the mAP on a single input, with the id of the person.
    if single_input is not None:
        eval_coco.params.imgIds = single_input
        print('The image id chosen is: ', single_input)

        # here are the ground truth values for
        image, keypoints, bbox, mask = coco_data_object.image_data.get_item_with_id(
            single_input)

        print('This is me debugging to check for the image shape ', image.shape)
        # Assuming the original size is (b, height, width)
        image_size = torch.tensor([image.shape[2], image.shape[1]])

        # visualize the actual expected input.
        print(
            f"Currently visualizing the annotation file's input value for the given image id: {single_input}")
        visualize_frame(keypoints.unsqueeze(0), image.unsqueeze(
            0), image_size[0].item(), image_size[1].item(), bbox.unsqueeze(0))

        print(
            f'The current ground truth values for the datapoints are keypoints: {keypoints}, bbox: {bbox}, image_size: {image_size}')

    # print the results that are going to be used
    print(pk_res, ' is the result of loading the annotations')

    # Run the evaluation
    eval_coco.evaluate()
    eval_coco.accumulate()
    eval_coco.summarize()

    # Get the mAP for keypoints
    # COCOeval.stats[0] is the mAP for keypoints
    average_precisions = eval_coco.stats
    # print(average_precisions)

    return average_precisions


def testing_loop(model, test_set, dataset_name, device):
    """
    Testing loop to run on the whole COCO dataset.

    Args:
        model (torch.nn.Module): The model to test.
        test_set (Dataset): The dataset to test on.
        dataset_name (str): The name of the dataset.
        device (torch.device): The device to run the model on.

    Returns:
        tuple: A tuple containing the model outputs, image IDs, ground truth joints, masks, image sizes, bounding boxes, and initial indexes.
    """
    model.eval()
    print('\t Memory before (in MB)', torch.cuda.memory_allocated()/1e6)

    outputs_lst = torch.tensor([]).to(device)
    gt_joints = torch.tensor([]).to(device)
    masks = torch.tensor([]).to(device)
    image_ids = []
    image_sizes = torch.tensor([]).to(device)
    bboxes = torch.tensor([]).to(device)
    keypoint_ids = []
    initial_indexes = []

    with torch.no_grad():
        # Go through each batch
        for i, data in enumerate(test_set):
            if dataset_name == 'JHMDB':
                raise NotImplementedError

            if dataset_name == 'COCO':
                inputs, processed_joints, mask, image_id, image_size, bbox, initial_index, keypoint_id = data

                # Skip images where none of the joints are being used
                for m in mask:
                    if (m == 0).all():
                        print("WEIRD")
                        continue

                image_ids.extend(image_id)

            else:
                raise NotImplementedError

            inputs = inputs.to(device)
            processed_joints = processed_joints.to(device)
            mask = mask.to(device)
            image_size = image_size.to(device)
            bbox = bbox.to(device)

            outputs = model(inputs)

            # Merge all the batches
            outputs_lst = torch.cat((outputs_lst, outputs))
            gt_joints = torch.cat((gt_joints, processed_joints))
            masks = torch.cat((masks, mask))
            image_sizes = torch.cat((image_sizes, image_size))
            bboxes = torch.cat((bboxes, bbox))
            keypoint_ids.extend(keypoint_id)
            initial_indexes.extend(initial_index)

        return outputs_lst, image_ids, gt_joints, masks, image_sizes, bboxes, initial_indexes, keypoint_ids


def main(config):
    """
    Main function to run the evaluation process.

    Args:
        config (dict): Configuration dictionary.
    """
    if config['dataset_name'] != 'COCO':
        print("Error the dataset selected is not COCO")
        exit()

    # these are the sigmas used by VITPOSE!
    pose_sigmas = np.array(config['sigmas'])
    stats_name = list(config['stats_name'])

    # Choosing checkpoint
    data_dir = config['data_path']
    batch_size = config['batch_size']
    checkpoint_dir = config['checkpoint_directory']
    checkpoint_name = config['checkpoint_name']
    test_checkpoint = None

    if test_checkpoint is None:
        lst = sorted(list(os.listdir(os.path.join(
            config['checkpoint_directory'], config['checkpoint_name']))))
        test_checkpoint = lst[0]

    print('Chosen checkpoint is', test_checkpoint)
    model_path = os.path.join(
        checkpoint_dir, checkpoint_name, test_checkpoint)

    # configuration
    pin_memory = True  # if only 1 GPU
    num_cpu_cores = os.cpu_count()
    num_workers = config['num_cpus'] * (num_cpu_cores) - 1

    # will be using the testing set to compare between the datasets (I used the val for the training
    test_set = eval_COCOVideoLoader(config, train_set='test', real_job=True)

    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(config, filepath=model_path)

    model = model.to(device)  # to unique GPU
    print('Model loaded successfully as follows: ', model)

    test_outputs, image_ids, test_labels, masks, image_sizes, bboxes, initial_indexes, keypoint_ids = testing_loop(
        model, test_loader, config['dataset_name'], device)

    # rearrange bboxes to be (B, 4)
    bboxes = rearrange(bboxes, 'b t x -> (b t) x')

    # now transform the inputs into COCO objects
    # need to denormalize the values, and keep the image ids
    tensor_width, tensor_height = config['image_tensor_width'], config['image_tensor_height']
    test_outputs = denormalize_fn(
        test_outputs, min_norm=config['min_norm'], h=tensor_height, w=tensor_width)
    test_labels = denormalize_fn(
        test_labels, min_norm=config['min_norm'], h=tensor_height, w=tensor_width)

    # denormalize the affine transforms to adjust to the image sizes to the original sizes
    for i in range(test_outputs.shape[0]):
        _, new_joint = inverse_process_joint_data(bboxes[i].cpu().detach().clone().numpy(), test_outputs[i][0].cpu(
        ).detach().clone().numpy(), (tensor_width, tensor_height), min_norm=config['min_norm'])
        test_outputs[i] = new_joint

        # I'll try with the initial images, unsure
        _, new_joint_label = inverse_process_joint_data(bboxes[i].cpu().detach().clone().numpy(
        ), test_labels[i][0].cpu().detach().clone().numpy(), (tensor_width, tensor_height), min_norm=config['min_norm'])
        test_labels[i] = new_joint_label

    # mask the results that are incorrect
    test_outputs = rearrange(test_outputs, 'b t j x-> (b t) j x')
    test_labels = rearrange(test_labels, 'b t j x-> (b t) j x')
    test_outputs, test_labels = coco_mask_fn(test_outputs, test_labels, masks)

    # evaluation of predicted keypoints
    annotations_path = os.path.join(
        config['data_path'], 'annotations', f'person_keypoints_val2017.json')

    cocoDt = tensor_to_coco(test_outputs, image_ids, image_sizes.cpu(
    ).numpy(), bboxes.cpu().numpy(), masks.cpu(), keypoint_ids)
    result = evaluate_coco(dt_annotations=cocoDt, data=annotations_path,
                           stats_name=stats_name, sigmas=pose_sigmas)
    print(f'mAP is {result}')

    # just for testing purposes, evaluation of the ground truth values
    cocoGt = tensor_to_coco(test_labels, image_ids, image_sizes.cpu(
    ).numpy(), bboxes.cpu().numpy(), masks.cpu(), keypoint_ids)
    result = evaluate_coco(dt_annotations=cocoGt, data=annotations_path,
                           stats_name=stats_name, sigmas=pose_sigmas)
    print(f'mAP is {result} (Should be 1.00)')

    # visualize ground truth and predicted
    # added a dimension, because its only 1 frame.
    image_index = 0
    visualize_frame(test_outputs[image_index].unsqueeze(0).cpu(), test_set.image_data[initial_indexes[image_index].item()][0].unsqueeze(
        0).cpu(), image_sizes[image_index][0].item(), image_sizes[image_index][1].item(), bboxes[image_index].unsqueeze(0).cpu())
    visualize_frame(test_labels[image_index].unsqueeze(0).cpu(), test_set.image_data[initial_indexes[image_index].item()][0].unsqueeze(0).cpu(
    ), image_sizes[image_index][0].item(), image_sizes[image_index][1].item(), bboxes[image_index].unsqueeze(0).cpu(), file_name='ground_truth')


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
