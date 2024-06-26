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

def coco_mask_fn(joints, labels, masks):
    '''Mask the necessary COCO style joint values'''
    # Get the shape of the predicted tensor
    B, T, J, X = joints.shape

    joints = rearrange(joints, 'b t j x-> (b t) j x')
    labels = rearrange(labels, 'b t j x-> (b t) j x')
    # masks = rearrange(masks, 'b t j x-> (b t) j x')

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

def evaluate_coco(dt_annotations, data):
    '''Given a ground truth annotations list and a predicted annotations list, return the mAP'''
    # Create COCO objects
    coco = data.coco
    
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
    with torch.no_grad():
        # go through each batch
        for i, data in enumerate(test_set):
            if dataset_name == 'JHMDB':
                raise NotImplementedError

            if dataset_name == 'COCO':
                inputs, processed_joints, mask, image_id, image_size, bbox = data
                image_ids.extend(image_id)
                
            else:
                raise NotImplementedError
            
            inputs = inputs.to(device)
            processed_joints = processed_joints.to(device)
            mask = mask.to(device)
            image_size = image_size.to(device)
            bbox = bbox.to(device)

            outputs = model(inputs)

            # outputs_lst.append(outputs)
            # gt_joints.append(processed_joints)
            # masks.append(mask)
        
            # merging all the batches.
            outputs_lst = torch.cat((outputs_lst, outputs))
            gt_joints = torch.cat((gt_joints, processed_joints))
            masks = torch.cat((masks, mask))
            image_sizes = torch.cat((image_sizes, image_size))
            bboxes = torch.cat((bboxes, bbox))
        
        return outputs_lst, image_ids, gt_joints, masks, image_sizes, bboxes

def main(config):
    if config['dataset_name'] != 'COCO':
        print("Error the dataset selected is not COCO")
        exit()
    
    data_dir = config['data_path']
    batch_size = config['batch_size']
    checkpoint_dir = config['checkpoint_directory']
    checkpoint_name = config['checkpoint_name']

    # configuration
    pin_memory = True  # if only 1 GPU
    num_cpu_cores = os.cpu_count()
    num_workers = config['num_cpus'] * (num_cpu_cores) - 1

    # will be using the testing set to compare between the datasets (I used the val for the training
    test_set = eval_COCOVideoLoader(config, train_set='test', real_job=True)

    test_loader = DataLoader(test_set, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # choosing the right model:
    if config['model_type'] == 'heatmap':
        model = HeatMapVideoMambaPose(config)

    elif config['model_type'] == 'HMR_decoder':
        model = HMRVideoMambaPose(config)
    
    elif config['model_type'] == 'MLP_only_decoder':
        model = MLPVideoMambaPose(config)
    
    elif config['model_type'] == 'HMR_decoder_coco_pretrain':
        model = HMRVideoMambaPoseCOCO(config)

    else:
        print('Your selected model does not exist! (Yet)')
        return

    model = model.to(device)  # to unique GPU
    print('Model loaded successfully as follows: ', model)

    test_outputs, image_ids, test_labels, masks, image_sizes, bboxes = testing_loop(model, test_loader, config['dataset_name'], device)

    # now transform the inputs into COCO objects
    # need to denormalize the values, and keep the image ids
    tensor_width, tensor_height = config['image_tensor_width'], config['image_tensor_height']
    test_outputs = denormalize_fn(test_outputs, min_norm=config['min_norm'], h=tensor_height, w=tensor_width)
    test_labels = denormalize_fn(test_labels, min_norm=config['min_norm'], h=tensor_height, w=tensor_width)

    # denormalize the affine transforms to adjust to the image sizes to the original sizes
    for i in range(test_outputs.shape[0]):
        _, new_joint = inverse_process_joint_data(bboxes[i][0].cpu().detach().clone().numpy(), test_outputs[i][0].cpu().detach().clone().numpy(), list(image_sizes[i].cpu().detach().clone()), min_norm=config['min_norm'])
        test_outputs[i] = new_joint

    # mask the results that are incorrect
    test_outputs, test_labels = coco_mask_fn(test_outputs, test_labels, masks)

    cocoDt = tensor_to_coco(test_outputs, image_ids)

    # technically useless
    # cocoGt = tensor_to_coco(test_labels, image_ids)

    result = evaluate_coco(cocoDt, eval_COCOLoader(config, train='test', real_job=True))

    print(f'mAP is {result}')

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
