'''
Using COCOeval to evaluate the pretrained model with Mamba.
'''
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

import torch 
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from einops import rearrange
import argparse
import os

# first import the dataset from data_format
from data_format.eval_Cocoloader import eval_COCOVideoLoader
from data_format.coco_dataset.CocoImageLoader import eval_COCOLoader
from import_config import open_config
from data_format.AffineTransform import denormalize_fn, inverse_process_joint_data

from models.heatmap.HeatVideoMamba import HeatMapVideoMambaPose
from models.HMR_decoder.HMRMambaPose import HMRVideoMambaPose
from models.MLP_only_decoder.MLPMambaPose import MLPVideoMambaPose
from models.HMR_decoder_coco_pretrain.HMRMambaPose import HMRVideoMambaPoseCOCO

from inference.visualize_coco import visualize, load_model

def visualize_frame(joint, frame, width, height, bbox, file_name='coco_pretrain_result'):
    visualize(joint, frame, file_name, width, height, bbox, False)

def coco_mask_fn(joints, labels, masks):
    '''Mask the necessary COCO style joint values'''
    # Get the shape of the predicted tensor
    B, J, X = joints.shape

    # Create a boolean mask where mask == 0
    zero_mask = (masks != 0)

    # Expand the mask to match the shape of the last dimension
    zero_mask = zero_mask.unsqueeze(-1).expand(-1, -1, X)

        # Instead of in-place modification, create new tensors with masked values set to 0
    joints = joints * (zero_mask).float()
    labels = labels * (zero_mask).float()
    return joints, labels


# note: from the val dataset, seems that category_id is always 1
def tensor_to_coco(tensor, image_ids, category_id=1, score=1.0):
    """
    Transform a tensor of shape (B, 17, 2) into a COCO result object.

    Args:
        tensor (torch.Tensor): The input tensor of shape (B, 17, 2).
        image_ids (list): A list of image IDs with length B.
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
        
        result = {
            "image_id": image_ids[i].item(), # THIS IS A TENSOR
            "category_id": category_id,
            "keypoints": keypoints_with_visibility,
            "score": score
            # 'id': i # unsure about this TODO
        }
        results.append(result)

    return results

def reflect_keypoints(joints, width, max_val):
    mid_point = max_val - width // 2
    

def evaluate_coco(dt_annotations, data):
    '''Given a ground truth annotations list and a predicted annotations list, return the mAP'''
    # Create COCO objects
    coco = COCO(data)
    
    # coco = COCO()
    # print(dir(coco))
    # print(coco)
    # # exit()

    # TODO oh wait, loadRes works for annotations for keypoints too ig?
    pk_res = coco.loadRes(dt_annotations)

    # OMG, I would like to adjust my gt annotations though...., because I am not reformmated to the values of the image sizes in the initial image.
    # gt_res = coco.loadRes(gt_annotations) #! TODO Okay, I think the coco Ground truth, I should not reload one, that means I should probably denormalize completely the values I get mbased on image size....

    # define a default keypoint object, using the default sigmas, yet no areas
    eval_coco = COCOeval(cocoGt=coco, cocoDt=pk_res, iouType='keypoints') #, use_area=False, sigmas=None)

    
    # Run the evaluation
    eval_coco.evaluate()
    eval_coco.accumulate()
    eval_coco.summarize()
    
    # Get the mAP for keypoints
    mAP = eval_coco.stats[0]  # COCOeval.stats[0] is the mAP for keypoints
    
    return mAP

def testing_loop(model, test_set, dataset_name, device):
    '''Testing loop to run on the whole COCO dataset
    '''
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
        # go through each batch
        for i, data in enumerate(test_set):
            if dataset_name == 'JHMDB':
                raise NotImplementedError

            if dataset_name == 'COCO':
                inputs, processed_joints, mask, image_id, image_size, bbox, initial_index = data
                image_ids.extend(image_id)
                
            else:
                raise NotImplementedError
            
            inputs = inputs.to(device)
            processed_joints = processed_joints.to(device)
            mask = mask.to(device)
            image_size = image_size.to(device)
            bbox = bbox.to(device)

            outputs = model(inputs)

            # merging all the batches.
            outputs_lst = torch.cat((outputs_lst, outputs))
            gt_joints = torch.cat((gt_joints, processed_joints))
            masks = torch.cat((masks, mask))
            image_sizes = torch.cat((image_sizes, image_size))
            bboxes = torch.cat((bboxes, bbox))
            initial_indexes.extend(initial_index)
        
        return outputs_lst, image_ids, gt_joints, masks, image_sizes, bboxes, initial_indexes

def main(config):
    if config['dataset_name'] != 'COCO':
        print("Error the dataset selected is not COCO")
        exit()

    # choosing checkpoint 
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

    # now transform the inputs into COCO objects
    # need to denormalize the values, and keep the image ids
    tensor_width, tensor_height = config['image_tensor_width'], config['image_tensor_height']
    test_outputs = denormalize_fn(test_outputs, min_norm=config['min_norm'], h=tensor_height, w=tensor_width)
    test_labels = denormalize_fn(test_labels, min_norm=config['min_norm'], h=tensor_height, w=tensor_width)

    # denormalize the affine transforms to adjust to the image sizes to the original sizes
    for i in range(test_outputs.shape[0]):
        _, new_joint = inverse_process_joint_data(bboxes[i][0].cpu().detach().clone().numpy(), test_outputs[i][0].cpu().detach().clone().numpy(), (tensor_width, tensor_height), min_norm=config['min_norm'])
        test_outputs[i] = new_joint

        # I'll try with the initial images, unsure
        _, new_joint_label = inverse_process_joint_data(bboxes[i][0].cpu().detach().clone().numpy(), test_labels[i][0].cpu().detach().clone().numpy(), (tensor_width, tensor_height), min_norm=config['min_norm'])
        test_labels[i] = new_joint_label

    # mask the results that are incorrect
    test_outputs = rearrange(test_outputs, 'b t j x-> (b t) j x')
    test_labels = rearrange(test_labels, 'b t j x-> (b t) j x')
    test_outputs, test_labels = coco_mask_fn(test_outputs, test_labels, masks)

    cocoDt = tensor_to_coco(test_outputs, image_ids)

    # technically useless
    # cocoGt = tensor_to_coco(test_labels, image_ids)
    annotations_path = os.path.join(config['data_path'], 'annotations', f'person_keypoints_val2017.json')
    result = evaluate_coco(cocoDt, annotations_path)
    print(f'mAP is {result}')

    # added a dimension, because its only 1 frame.
    image_index = 343
    visualize_frame(test_outputs[image_index].unsqueeze(0).cpu(), test_set.image_data[initial_indexes[image_index].item()][0].unsqueeze(0).cpu(), image_sizes[image_index][0].item(), image_sizes[image_index][1].item(), bboxes[image_index].cpu())
    visualize_frame(test_labels[image_index].unsqueeze(0).cpu(), test_set.image_data[initial_indexes[image_index].item()][0].unsqueeze(0).cpu(), image_sizes[image_index][0].item(), image_sizes[image_index][1].item(), bboxes[image_index].cpu(), file_name='ground_truth')

    # just checking for the guitar
    cocoGt = tensor_to_coco(test_labels, image_ids)

    result = evaluate_coco(cocoGt, annotations_path)
    print(f'mAP is {result} (Should be 1.00)')


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
