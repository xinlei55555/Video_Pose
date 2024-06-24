import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
import torch.nn as nn
import sys
import os
from PIL import Image
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse

from import_config import open_config

from data_format.CocoVideoLoader import COCOVideoLoader

def load_model(config , filepath, parallel=False):
    # Create the model
    # choosing the right model:
    if config['model_type'] == 'heatmap':
        model = HeatMapVideoMambaPose(config)

    elif config['model_type'] == 'HMR_decoder':
        model = HMRVideoMambaPose(config)
    
    elif config['model_type'] == 'MLP_only_decoder':
        model = MLPVideoMambaPose(config)

    elif config['model_type'] == 'HMR_decoder_coco_pretrain':
            model = mHMRVideoMambaPoseCOCO(config).to(rank)

    else:
        print('Your selected model does not exist! (Yet)')
        return

    # load the dictionary from checkpoint, and load the weights into the model.
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

    # loading model that was trained with DDP
    if parallel:
        checkpoint = adapt_model_parallel(checkpoint)

    # loading this requires me to check from the initial save
    # strict = False makes it so that even though some layer are missing, it will work (although idk why some layesr are missing)
    model.load_state_dict(checkpoint)

    # Set model to evaluation mode
    model.eval()

    return model


def adapt_model_parallel(checkpoint):
    # in case we load a DDP model checkpoint to a non-DDP model
    model_dict = checkpoint
    pattern = re.compile('module.')
    for k, v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    return model_dict


def inference(model, input_tensor):
    # Disable gradient computation for inference
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)

    return output


def get_data_points(dataset, index):
    image, joint, bbox = dataset[index]
    return image, joint, bbox


def visualize(joints, frames, file_name, width, height, bboxes=None, use_last_frame_only=False):
    '''1: neck
    2: belly
    3: face
    4: right shoulder
    5: left  shoulder
    6: right hip
    7: left  hip
    8: right elbow
    9: left elbow
    10: right knee
    11: left knee
    12: right wrist
    13: left wrist
    14: right ankle
    15: left ankle'''

    num_frames, num_joints = frames.shape[0], joints.shape[0]
    if num_frames != num_joints:
        print("Error, the number of joints does not equal the number of frames")
        raise NotImplementedError
    
    else:
        print('Visualization Information: ')
        print('\t', len(list(joints)), 'is the number of joints you have')
        print('\t', len(list(frames)), 'is the number of frames that you have')
        print('\t', 'The passed width and height are ', width, height)
    # generate a new folder name
    idx = 1
    num_char = len(file_name)
    while os.path.exists(os.path.join('inference/results', file_name)):
        if idx == 1:
            file_name = file_name + str(idx)
        else:
            file_name = file_name[:num_char] + str(idx)
        idx += 1
    file_name = os.path.join('inference/results', file_name)
    os.mkdir(file_name)

    # Create a video writer to save the output
    for frame_idx in range(num_frames):
        # Get the joints for the current frame
        joints_per_frame = joints[frame_idx]
        image = frames[frame_idx]

        image = rearrange(image, 'c h w->h w c')
        
        # converting and changing to cpu before plotting
        image = image.clone().to('cpu')
        image = image.numpy()

        # Scale image data to the range [0, 1] for floats
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image - image.min()) / (image.max() - image.min())

        joints_per_frame = joints_per_frame.clone().to('cpu')
        joints_per_frame = joints_per_frame.numpy()

        # Plotting the joints onto the image
        plt.figure()
        plt.imshow(image)
        plt.scatter(
            joints_per_frame[:, 0], joints_per_frame[:, 1], c='red', s=10, label='Joints')

        # Annotate the joints with their indices
        for i, pos in enumerate(joints_per_frame):
            x, y = pos[0], pos[1]
            plt.text(x, y, str(i+1), color="blue",
                     fontsize=12, ha='right', va='bottom')
        
        # Draw bounding boxes if provided
        if bboxes is not None:
            bbox = bboxes[frame_idx]
            x_min, y_min, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), w, h, fill=False, edgecolor='green', linewidth=2))

        plt.title(f'Frame {frame_idx + 1}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.xlim(0, width)
        plt.ylim(height, 0)  # Invert y-axis to match image coordinate system
        plt.legend()

        # Save the current plot to an image file
        plt.savefig(f'{file_name}/frame_{frame_idx}.png')
        plt.close()

    print(f">>> Video saved as {file_name}")


def main(config):
    # some default values.
    ground_truth = False
    resized_ground_truth = False
    predicted = True
    joints_exist = True

    # this is the index to callf rom the dataloader
    index_used = 0 
    
    # finding a checkpoint and model path:
    # test_checkpoint = 'HMR_decoder_initial_run_0.0401.pt'
   
    if test_checkpoint is None:
        lst = sorted(list(os.listdir(os.path.join(config['checkpoint_directory'], config['checkpoint_name']))))
        test_checkpoint = lst[0]
    print('Chosen checkpoint is', test_checkpoint)
    model_path = os.path.join(
        config['checkpoint_directory'], config['checkpoint_name'], test_checkpoint)

    
    # defining the sizes of the videos:
    width, height = config['image_width'], config['image_height']
    tensor_width, tensor_height = config['image_tensor_width'], config['image_tensor_height']
    frames_per_vid = config['num_frames']

    # load video and ground truth joints
    # I just want a testing value, no need to load all
    dataset = COCOVideoLoader(config, train_set=False, real_job=False)
    image, joint, bbox = get_data_points(dataset, index_used)

    # add batch_size dimension
    image = image.unsqueeze(0)
    joint = joint.unsqueeze(0)
    bbox = bbox.unsqueeze(0)

    # visualizing ground truth
    if ground_truth:
        visualize(joints, frames, 'Normalized_ground_truth',
                  width, height, bboxes, config['use_last_frame_only'])

    if predicted:
        # load the model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the model from the .pt file
        model = load_model(config, model_path, config['parallelize'])
        model = model.to(device)

        # show models + gradients (if wanted)
        if config['full_debug']:
            print(model)
            if config['show_gradients']:
                for name, param in model.named_parameters():
                    print(f'Parameter: {name}')
                    print(f'Values: {param.data}')
                    if param.grad is not None:
                        print(f'Gradients: {param.grad}')
                    else:
                        print('No gradient computed for this parameter')
                        print('---')

     
          
        batch_outputs = inference(model, image.to(device))
        batch_outputs = batch_outputs.to('cpu')

        # creating the final tensors:
        final_videos = rearrange(batch_videos, 'b c d h w -> (b d) c h w')
        final_outputs = rearrange(batch_outputs, 'b d j x -> (b d) j x')
        ground_truth_joints = rearrange(batch_ground_truth_joints, 'b d j x->(b d) j x')

        # denormalize the final_outputs to suit the new screen size
        final_outputs = denormalize_fn(final_outputs, config['min_norm'], h=tensor_height, w=tensor_width)
        ground_truth_joints = denormalize_fn(ground_truth_joints, config['min_norm'], h=tensor_height, w=tensor_width)
        
        # creating the directory, if doesn't exist
        os.makedirs(f'inference/results/{config["model_type"]}/{config["checkpoint_name"]}', exist_ok=True)
        
        # visualization
        visualize(final_outputs, final_videos, f"{config['model_type']}/{config['checkpoint_name']}/Predicted", tensor_width, tensor_height, None, False)

        # showing the ground truth with the resized joints
        if resized_ground_truth:
            visualize(ground_truth_joints, final_videos, f"{config['model_type']}/{config['checkpoint_name']}/resized_ground_truth", tensor_width, tensor_height, None, False)



if __name__ == '__main__':
    from data_format.AffineTransform import denormalize_fn, bounding_box, inference_yolo_bounding_box, preprocess_video_data#, inverse_process_joints_data, inverse_process_joint_data, preprocess_video_data
    from models.heatmap.HeatVideoMamba import HeatMapVideoMambaPose
    from models.HMR_decoder.HMRMambaPose import HMRVideoMambaPose
    from models.MLP_only_decoder.MLPMambaPose import MLPVideoMambaPose

    # argparse to get the file path of the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='heatmap/heatmap_beluga.yaml',
                        help='Name of the configuration file')
    args = parser.parse_args()
    config_file = args.config

    # import configurations:
    config = open_config(config_file)

    main(config)
