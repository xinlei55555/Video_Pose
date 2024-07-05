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
from data_format.eval_Cocoloader import eval_COCOVideoLoader
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
            # "center": center.tolist(),
            # "scale": scale.tolist(),
            # "bbox": bboxes[i].tolist(), # the bbox when nothing is declared just becomes the maximum and the minimum in the image
            # 'bbox': False,
        }
        results.append(result)

    return results

def reflect_keypoints(bboxes):
    """
    Calculate the midpoint of bounding boxes.
    
    Args:
        bboxes (torch.Tensor): Bounding boxes tensor of shape (B, 4).
    
    Returns:
        torch.Tensor: Tensor with the midpoint of each bounding box.
    """
    mid_points = bboxes[:, 0] + bboxes[:, 2] // 2
    return mid_points

def evaluate_coco(dt_annotations, data, coco_data_object=None, stats_name=None, sigmas=None, single_input=None, imgIds=None):
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
    # Create COCO objects
    if coco_data_object is not None:
        print("using the default coco ground truth values from the image loader")
        coco = coco_data_object.image_data.coco
    else:
        print("using custom loaded coco data object results as the ground truth")
        coco = COCO(data)

    # #TODO !!! I think that might be what was missing...
    # person_ann_ids = coco.getAnnIds(catIds=[1])
    # person_anns = coco.loadAnns(ids=person_ann_ids)
    # print(person_anns)

    with open("outputs/results.txt", "w") as f:
        f.write(json.dumps(dt_annotations))
    
    pk_res = coco.loadRes("outputs/results.txt")
    
    # Load the keypoints
    # pk_res = coco.loadRes(dt_annotations)

    # Define a default keypoint object, using the default sigmas, yet no areas
    eval_coco = COCOeval(cocoGt=coco, cocoDt=pk_res, iouType='keypoints')#, use_area=False, sigmas=sigmas)

    # This runs the mAP on a single input
    if single_input is not None:
        eval_coco.params.imgIds = single_input
        print('The image id chosen is: ', single_input)

        # here are the ground truth values for 
        image, keypoints, bbox, mask, image_size = coco_data_object.image_data.get_item_with_id(single_input)

        # visualize the actual expected input.
        print(f"Currently visualizing the annotation file's input value for the given image id: {single_input}")
        visualize_frame(keypoints.unsqueeze(0), image.unsqueeze(0), image_size[0].item(), image_size[1].item(), bbox.unsqueeze(0))

        print(f'The current ground truth values for the datapoints are keypoints: {keypoints}, bbox: {bbox}, image_size: {image_size}')

    if single_input is None and imgIds is not None:
        print(f'The initial length of imgIds is {len(eval_coco.params.imgIds)} and the new length is {len(imgIds)}')
        eval_coco.params.imgIds = imgIds
        print('here are all the coco parameters: ', dir(eval_coco.params))

    # print the results that are going to be used
    print(pk_res, ' is the result of loading the annotations')

    # [https://github.com/facebookresearch/maskrcnn-benchmark/issues/524]
    # eval_coco.params.iouThrs = np.array([0.25, 0.3 , 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])
     
    # eval_coco.params.useSegm = None # replaced by ioutype
    # eval_coco.params.useCats = 0
    
    
    # Run the evaluation
    eval_coco.evaluate()
    eval_coco.accumulate()
    eval_coco.summarize()
    
    # Get the mAP for keypoints
    average_precisions = eval_coco.stats # COCOeval.stats[0] is the mAP for keypoints
    # print(average_precisions)

    return average_precisions

def calculate_mAP(cocoDt, stats_name, image_ids, annotations_path, pose_sigmas):
    '''
    Calculates the mean Average Precision for the whole dataset

    Args:
        cocoDt (COCO): A coco object which contains the loaded prediction results
        stats_name (list): Names of the different metrics that are employed
        image_ids (list[torch.Tensor]): list of image ids
        annotations_path: path to the annotation file
        pose_sigmas (np.array): sigmas for joint value

    Returns:
        A float representing the mean average precision of the given dataset.
    '''
    # checking the length of the number of stats_names
    if 10 != len(stats_name):
        print("The length of the sigmas is not equal to the number of average precisions calculated")
        exit()

    results = torch.zeros(len(stats_name))
    num_elements = torch.zeros(len(stats_name))
    num_exceptions = 0
    for i in range(len(image_ids)):
        evaluation = evaluate_coco(cocoDt, annotations_path, stats_name=stats_name, sigmas=pose_sigmas, single_input=image_ids[i].item())#$np.array([1.0 for _ in range(17)]))
        # results = torch.tensor([el if el.item() >= 0. for el in evaluation]) 
        # for el in range(evaluation.shape[0]):
        #     if evaluation[el].item() > 0
        #         results[el] += evaluation[el].item()
        #         num_elements[el] += 1
         # Only consider positive evaluation values
        if len(evaluation) != 10:
            print(f"Unexpected evaluation shape: {len(evaluation)}")
            print(evaluation)
            print(image_ids[i])
            num_exceptions += 1
            print("index num", i)
            exit()
        evaluation = torch.tensor(evaluation)
        positive_mask = evaluation > 0
        
        # Accumulate results for positive values
        results += evaluation * positive_mask
        num_elements += positive_mask.float()
    print(f"The number of exceptions was {num_exceptions}")

    average_precisions = results / num_elements
    mean_average_precisions = torch.sum(average_precisions) / (average_precisions.shape[0])
    return mean_average_precisions

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
    initial_indexes = []
    
    with torch.no_grad():
        # Go through each batch
        for i, data in enumerate(test_set):
            if dataset_name == 'JHMDB':
                raise NotImplementedError

            if dataset_name == 'COCO':
                inputs, processed_joints, mask, image_id, image_size, bbox, initial_index = data
                
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
            initial_indexes.extend(initial_index)
        
        return outputs_lst, image_ids, gt_joints, masks, image_sizes, bboxes, initial_indexes

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
        lst = sorted(list(os.listdir(os.path.join(config['checkpoint_directory'], config['checkpoint_name']))))
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

    test_outputs, image_ids, test_labels, masks, image_sizes, bboxes, initial_indexes = testing_loop(model, test_loader, config['dataset_name'], device)

    # rearrange bboxes to be (B, 4)
    bboxes = rearrange(bboxes, 'b t x -> (b t) x')

    # now transform the inputs into COCO objects
    # need to denormalize the values, and keep the image ids
    tensor_width, tensor_height = config['image_tensor_width'], config['image_tensor_height']
    test_outputs = denormalize_fn(test_outputs, min_norm=config['min_norm'], h=tensor_height, w=tensor_width)
    test_labels = denormalize_fn(test_labels, min_norm=config['min_norm'], h=tensor_height, w=tensor_width)

    # denormalize the affine transforms to adjust to the image sizes to the original sizes
    for i in range(test_outputs.shape[0]):
        _, new_joint = inverse_process_joint_data(bboxes[i].cpu().detach().clone().numpy(), test_outputs[i][0].cpu().detach().clone().numpy(), (tensor_width, tensor_height), min_norm=config['min_norm'])
        test_outputs[i] = new_joint

        # I'll try with the initial images, unsure
        _, new_joint_label = inverse_process_joint_data(bboxes[i].cpu().detach().clone().numpy(), test_labels[i][0].cpu().detach().clone().numpy(), (tensor_width, tensor_height), min_norm=config['min_norm'])
        test_labels[i] = new_joint_label

    # mask the results that are incorrect
    test_outputs = rearrange(test_outputs, 'b t j x-> (b t) j x')
    test_labels = rearrange(test_labels, 'b t j x-> (b t) j x')
    test_outputs, test_labels = coco_mask_fn(test_outputs, test_labels, masks)

    # example image index to use in future inference
    image_index = 9
    # i want the image_id = 468965
    image_index = image_ids.index(torch.tensor(468965))
    print("here is the initial index: ", image_ids[image_index].item())

    # cocoDt = tensor_to_coco(test_outputs, image_ids, image_sizes.cpu().numpy(), bboxes.cpu().numpy(), masks[image_index].cpu())

    annotations_path = os.path.join(config['data_path'], 'annotations', f'person_keypoints_val2017.json')
    
    # # calculate the summation
    # result = calculate_mAP(cocoDt=cocoDt, stats_name=stats_name, image_ids=image_ids, annotations_path=annotations_path, pose_sigmas=pose_sigmas)
    # print(f'mAP is {result}')

    # added a dimension, because its only 1 frame.
    visualize_frame(test_outputs[image_index].unsqueeze(0).cpu(), test_set.image_data[initial_indexes[image_index].item()][0].unsqueeze(0).cpu(), image_sizes[image_index][0].item(), image_sizes[image_index][1].item(), bboxes[image_index].unsqueeze(0).cpu())

    # just checking for the labels after normalization and denormalization process
    # cocoGt = tensor_to_coco(test_labels, image_ids, image_sizes.cpu().numpy(), bboxes.cpu().numpy())

    # result = calculate_mAP(cocoDt=cocoGt, stats_name=stats_name, image_ids=image_ids, annotations_path=annotations_path, pose_sigmas=pose_sigmas)
    # print(f'mAP is {result} (Should be 1.00)')
    visualize_frame(test_labels[image_index].unsqueeze(0).cpu(), test_set.image_data[initial_indexes[image_index].item()][0].unsqueeze(0).cpu(), image_sizes[image_index][0].item(), image_sizes[image_index][1].item(), bboxes[image_index].unsqueeze(0).cpu(), file_name='ground_truth')

    # # !TEST now, checking for the initial cocoGt, and seeing how much that has as mAP
    unprocessed_results = []    
    unprocessed_masks = []
    for idx in range(len(initial_indexes)):
        # comes from the output of the imageLoader for COCO
        data = test_set.image_data[initial_indexes[idx].item()]
        unprocessed_results.append(data[1].cpu())
        unprocessed_masks.append(data[3].cpu())
    unprocessed_results = torch.stack(unprocessed_results)
    unprocessed_masks = torch.stack(unprocessed_masks)


    cocoDt_2 = tensor_to_coco(unprocessed_results, image_ids, image_sizes.cpu().numpy(), bboxes.cpu().numpy(), unprocessed_masks.cpu().numpy())

    # # okay, some othe rbug to investigate... but my image_ids seems to change...
    results = evaluate_coco(cocoDt_2, annotations_path, coco_data_object=test_set, stats_name=stats_name, sigmas=pose_sigmas, single_input=image_ids[image_index].item())
    print(f'mAP is {results} (Should be 1.00)')

    print(masks[image_index], 'is the mask values')
    print(cocoDt_2[image_index], 'is the predicted values?')


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
