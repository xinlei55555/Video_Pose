'''
Using COCOeval to evaluate the pretrained model with Mamba.
'''
from xtcocotools.cocoeval import COCOeval
from xtcocotools.coco import COCO


# first import the dataset from data_format
from data_format.CocoVideoLoader


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
    gtCOCO = 
    eval_coco = COCOeval(cocoGt=gtCOCO, cocoDt=pkCOCO, sigmas=None, iouType='keypoints', use_area=False)    


def main(config):
    if config['dataset_name'] != 'COCO':
        print("Error the dataset selected is not COCO")
        exit()
    
    data_dir = config['data_path']
    # will be using the testing set to compare between the datasets (I used the val for the training)
    annotation_file = os.path.join(data_dir, 'annotations', 'person_keypoints_test2017.json')

    cocoGT = COCO(annotation_file)

    # then run the entire dataset with the data
    # or, I need to create another data loader, but with the 
    # okay, I will change my dataloadr.
    



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
