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



def load_model(config , filepath, parallel=False):
    # Create the model
    # choosing the right model:
    if config['model_type'] == 'heatmap':
        model = HeatMapVideoMambaPose(config)

    elif config['model_type'] == 'HMR_decoder':
        model = HMRVideoMambaPose(config)
    
    elif config['model_type'] == 'MLP_only_decoder':
        model = MLPVideoMambaPose(config)

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


def get_input_and_label(use_videos, joint_path, video_path):
    # for the sake of testing, I will juhhst hard copy some files and the respective joint outputs
    # I'll also make sure they were in the test files
    joints = scipy.io.loadmat(os.path.join(joint_path, 'joint_positions.mat'))
    joints = torch.tensor(joints['pos_img'])

    joints = rearrange(joints, 'd n f->f n d')
    video = video_to_tensors(video_path, use_videos)

    return joints, video


def image_to_tensor(image_path):
    '''Returns a torch tensor for a given image associated with the path'''
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor = transform(image)
    return tensor


def video_to_tensors(video_path, use_videos):
    '''
    Returns a tensor with the following:
    (n_frames, num_channels (3), 224, 224)
    '''

    # goes through the each image.
    if not use_videos:
        image_tensors = []

        # reorder the images
        filenames = []
        for filename in os.listdir(video_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filenames.append(filename)
        filenames.sort()
        for filename in filenames:
            file_path = os.path.join(video_path, filename)
            if os.path.isfile(file_path):
                image_tensor = image_to_tensor(file_path)
                image_tensors.append(image_tensor)

        # Concatenates a sequence of tensors along a new dimension.
        batch_tensor = torch.stack(image_tensors)

    else:
        print("Currently using the video to open the files")
        if not os.path.isfile(video_path):
            print(f"The video file {video_path} does not exist.")

        # Initialize a list to store the frames
        frames = []

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video was opened successfully
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")

        # Read frames until the end of the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame from BGR to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert frame to tensor and add it to the list
            frame_tensor = torch.tensor(frame, dtype=torch.float32)
            frames.append(frame_tensor)

        # Release the video capture object
        cap.release()

        # Stack all frames into a single tensor
        batch_tensor = torch.stack(frames)
        print('The shape of your video is', batch_tensor.shape)

        # Transpose the tensor to have the shape (frames, channels, height, width)
        batch_tensor = rearrange(batch_tensor, 'n h w c-> n c h w')

    return batch_tensor

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

    # some default paths for test and train datapoints:
    # video_path = 'inference/test_visualization/20_good_form_pullups_pullup_f_nm_np1_ri_goo_0.avi'
    # joint_path = 'inference/test_visualization/20_good_form_pullups_pullup_f_nm_np1_ri_goo_0'
    # video_path = 'inference/test_visualization/11_4_08ErikaRecurveBack_shoot_bow_u_nm_np1_ba_med_0.avi'
    # joint_path = 'inference/test_visualization/11_4_08ErikaRecurveBack_shoot_bow_u_nm_np1_ba_med_0'
    if config['use_videos']:
        video_path = 'inference/test_visualization/practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_12.avi'
        # video_path = 'inference/test_visualization/HowtoswingaBaseballbat_swing_baseball_f_nm_np1_le_bad_0.avi'
    else:
        video_path = 'inference/test_visualization/practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_12 copy'
    joint_path = 'inference/test_visualization/practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_12'
    # joint_path = 'inference/test_visualization/HowtoswingaBaseballbat_swing_baseball_f_nm_np1_le_bad_0'
    
    # finding a checkpoint and model path:
    # test_checkpoint = 'HMR_decoder_initial_run_0.0401.pt'
    # test_checkpoint = 'heatmap_0.0413.pt'
    # test_checkpoint = 'HMR_decoder_1_train_input_transformer_0.0531.pt'
    # test_checkpoint = 'HMR_decoder_angle_velocity_1_train_input_transformer_0.0535.pt'
    # test_checkpoint = 'HMR_decoder_angle_velocity_1_train_input_transformer_2.3679.pt'
    # test_checkpoint = 'HMR_decoder_angle_velocity_1_train_input_transformer_2.0946.pt'
    test_checkpoint  = 'HMR_decoder_angle_velocity_1_train_input_transformer_0.0469.pt'
    test_checkpoint = 'HMR_decoder_angle_velocity_1_train_input_transformer_0.0537.pt'
    test_checkpoint = 'HMR_decoder_new_velocity_1_train_input_transformer_4.8700.pt'
    test_checkpoint = 'HMR_decoder_new_mjpje_velocity_1_train_input_transformer_5.6138.pt'
    test_checkpoint = 'HMR_decoder_new_velocity_1_train_input_transformer_1.5393.pt'
    test_checkpoint = None
    # test_checkpoint = 'HMR_decoder_new_velocity_10_angle_1_mse_5_train_input_transformer_3.5849_epoch_150.pt'
    test_checkpoint = 'HMR_decoder_new_velocity_10_angle_1_mse_5_train_input_transformer_4.2295_epoch_100.pt'

    if test_checkpoint is None:
        lst = sorted(list(os.listdir(os.path.join(config['checkpoint_directory'], config['checkpoint_name']))))
        test_checkpoint = lst[0]
    print('Chosen checkpoint is', test_checkpoint)
    model_path = os.path.join(
        config['checkpoint_directory'], config['checkpoint_name'], test_checkpoint)

    # defining the jump
    jump = config['jump']
    if not config['use_last_frame_only'] and jump != config['num_frames']:
        print("The Jump does not match the parameter for last frame!")
        raise NotImplementedError
    
    # defining the sizes of the videos:
    width, height = config['image_width'], config['image_height']
    tensor_width, tensor_height = config['image_tensor_width'], config['image_tensor_height']
    frames_per_vid = config['num_frames']

    # load video and ground truth joints
    joints, frames = get_input_and_label(config['use_videos'], joint_path, video_path)

    # get the bounding boxes
    if joints_exist:
        bboxes = bounding_box(joints)
    # elsewise, use the yolo algorithm
    else:
        bboxes = inference_yolo_bounding_box(joints)
        
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

        # predicting only the last frame in a video
        if config['use_last_frame_only']:
            # skip the first 15 frames, so only keep from index 15 included onwards
            frames = frames[15:].detach.clone()
            # I'll do it later, but basically what I was doing with the old visualization.
            raise NotImplementedError
        # predicting all 16 frames of a video
        else:
            # then I need to remove the last frames, which are not in the modulo 16
            max_num = len(frames) - (len(frames) % frames_per_vid)
            frames = frames[:max_num].detach().clone()

            batch_videos = []
            batch_ground_truth_joints = []
            for frame_idx in range(0, max_num, frames_per_vid):
                # get subvideos of 16 frames at once.
                if frame_idx + frames_per_vid > len(frames):
                    print('max_num does not seem to be the correct value')
                    break
                print(frame_idx, frame_idx+frames_per_vid)
                input_video = frames.detach().clone()[frame_idx: frame_idx + frames_per_vid]
                ground_truth_joints = joints.detach().clone()[frame_idx: frame_idx + frames_per_vid]
                bboxes_indexed = bboxes.detach().clone()[frame_idx: frame_idx + frames_per_vid]

                # preprocess the videos
                input_video = rearrange(input_video, 'd c h w -> d h w c')          
                print(input_video.shape, 'is the shape of the input video')
                input_video, ground_truth_joints = preprocess_video_data(input_video.detach().clone().numpy(), bboxes_indexed.detach().clone().numpy(), ground_truth_joints.detach().clone().numpy(), (tensor_width, tensor_height), config['min_norm'])

                # rearrange to channel first                
                input_video = rearrange(input_video, 'd c h w-> c d h w')
                
                batch_videos.append(input_video)
                batch_ground_truth_joints.append(ground_truth_joints)
            # make into a single torch tensor
            batch_videos = torch.stack(batch_videos)
            batch_ground_truth_joints = torch.stack(batch_ground_truth_joints)
            
            # run inference
            batch_videos = batch_videos.to(device)
            batch_outputs = inference(model, batch_videos.detach().clone())
            batch_outputs = batch_outputs.to('cpu')

            # creating the final tensors:
            final_videos = rearrange(batch_videos, 'b c d h w -> (b d) c h w')
            final_outputs = rearrange(batch_outputs, 'b d j x -> (b d) j x')
            ground_truth_joints = rearrange(batch_ground_truth_joints, 'b d j x->(b d) j x')

            # denormalize the final_outputs to suit the new screen size
            final_outputs = denormalize_fn(final_outputs, config['min_norm'], h=tensor_height, w=tensor_width)
            ground_truth_joints = denormalize_fn(ground_truth_joints, config['min_norm'], h=tensor_height, w=tensor_width)
            
            # creating the directory, if doesn't exist
            os.makedirs(f'inference/results/{config["checkpoint_name"]}', exist_ok=True)
            
            # visualization
            visualize(final_outputs, final_videos, f"{config['checkpoint_name']}/Predicted", tensor_width, tensor_height, None, False)

            # showing the ground truth with the resized joints
            if resized_ground_truth:
                visualize(ground_truth_joints, final_videos, f"{config['checkpoint_name']}/resized_ground_truth", tensor_width, tensor_height, None, False)



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
