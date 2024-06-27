'''
Using COCOeval to evaluate the pretrained model with Mamba.
'''
from xtcocotools.cocoeval import COCOeval
from xtcocotools.coco import COCO

import torch 
from einops import rearrange

from data_format.eval_Cocoloader import eval

# first import the dataset from data_format
from data_format.eval_Cocoloader import eval_CocoVideoLoader


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
            "image_id": image_ids[i],
            "category_id": category_id,
            "keypoints": keypoints_with_visibility,
            "score": score
        }
        results.append(result)

    return results

def evaluate_coco(gtCOCO, pkCOCO):
    '''Given a ground truth coco object, and a predicted keypoint object, return the mAP'''
    # define a default keypoint object, using the default sigmas, yet no areas
    eval_coco = COCOeval(cocoGt=gtCOCO, cocoDt=pkCOCO, sigmas=None, iouType='keypoints', use_area=False)    

def testing_loop(model, test_set, dataset_name):
    '''Testing loop to run on the whole COCO dataset
    '''
    model.eval()
    print('\t Memory before (in MB)', torch.cuda.memory_allocated()/1e6)
    
    outputs_lst = []
    image_ids = []
    with torch.no_grad():
        # go through each batch
        for i, data in enumerate(test_set):
            if dataset_name == 'JHMDB':
                raise NotImplementedError

            if dataset_name == 'COCO':
                inputs, image_id = data
                image_ids.append(image_id)
                
                # TODO I am not sure if I need the mask, include after
                # mask = mask.to(device)
            else:
                raise NotImplementedError

            outputs = model(inputs)

            outputs_lst.append(outputs)
        
        # stack all the batches into one dataset
        outputs_lst = torch.stack(outputs_lst)

        # merging all the batches.
        outputs_lst = rearrange(outputs_lst, 'n b j x -> (n b) j x')
        
        return outputs_lst, image_ids

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

    # will be using the testing set to compare between the datasets (I used the val for the training)
    annotation_file = os.path.join(data_dir, 'annotations', 'person_keypoints_test2017.json')

    # cocoGT = COCO(annotation_file)

    # then run the entire dataset with the data
    # or, I need to create another data loader, but with the 
    # okay, I will change my dataloadr.
    test_set = eval_COCOVideoLoader(config, train_set='test', real_job=True)

    test_loader = DataLoader(test_set, batch_size=batch_size, 
                            shuffle=False, num_workers=num_worker, pin_memory=pin_memory)

    # choosing the right model:
    if config['model_type'] == 'heatmap':
        model = HeatMapVideoMambaPose(config).to(device)

    elif config['model_type'] == 'HMR_decoder':
        model = HMRVideoMambaPose(config).to(device)
    
    elif config['model_type'] == 'MLP_only_decoder':
        model = MLPVideoMambaPose(config).to(device)
    
    elif config['model_type'] == 'HMR_decoder_coco_pretrain':
        model = HMRVideoMambaPoseCOCO(config).to(rank)

    else:
        print('Your selected model does not exist! (Yet)')
        return

    model = model.to(device)  # to unique GPU
    print('Model loaded successfully as follows: ', model)

    test_outputs, image_ids = testing_loop(model, test_loader, config['dataset_name'])

    # now transform the inputs into COCO objects
    # need to denormalize the values, and keep the image ids

    cocoDt = tensor_to_coco(test_outputs, image_ids)



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
