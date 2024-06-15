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
    # change to correct directory:
    num_frames = min(
        len(list(joints)), len(list(frames)))  # Number of frames in the video

    num_frames_per_video = len(list(frames)) - len(list(joints))

    print('\t', len(list(joints)), 'is the number of joints you have')
    print('\t', len(list(frames)), 'is the number of frames that you have')

    print('\t', 'The passed width and height are ', width, height)

    # generate a new folder name
    idx = 1
    num_char = len(file_name)
    while os.path.exists(os.path.join('inference/results', file_name)):
        if idx == 1:
            file_name = str(idx) + file_name
        else:
            file_name = str(idx) + file_name[-num_char:]
        idx += 1
    file_name = os.path.join('inference/results', file_name)
    os.mkdir(file_name)

    # NOTE: even though in the dataloader, I am using a more efficient way, I won't because anyways here, the overhead copying isn't that much
    # Create a video writer to save the output
    for frame_idx in range(num_frames):
        # Get the joints for the current frame
        joints_per_frame = joints[frame_idx]

        # Create a blank 320x240 image (white background)
        # note that for the visualization, the frame number are going to be different
        # if use_last_frame_only:
        image = frames[frame_idx]
        # since for the ones that were not 16 frames at once, I started the prediction at frame 0
        # else:
            # image = frames[frame_idx - num_frames_per_video]

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
        # testing for if its the normalization that made the values very
        # plt.scatter(joints_per_frame[:, 0]*320, joints_per_frame[:, 1]*240, c='red', s=10, label='Joints')
        plt.scatter(
            joints_per_frame[:, 0], joints_per_frame[:, 1], c='red', s=10, label='Joints')

        # Annotate the joints with their indices
        for i, pos in enumerate(joints_per_frame):
            # print('pos ', pos)
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

    print(f">> Video saved as {file_name}")

def debug_parameters(config, model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)

def main(config):
    ground_truth = False
    resized_ground_truth = True
    predicted = True
    # better_visualization will always be true except if last frame ig, cuz its much better
    better_visualization = True


    joints_exist = True
    jump = config['jump']
    if not config['use_last_frame_only'] and jump != config['num_frames']:
        print("The Jump does not match the parameter for last frame!")
        raise NotImplementedError
    # video_path = 'inference/test_visualization/20_good_form_pullups_pullup_f_nm_np1_ri_goo_0.avi'
    # joint_path = 'inference/test_visualization/20_good_form_pullups_pullup_f_nm_np1_ri_goo_0'
    # video_path = 'inference/test_visualization/11_4_08ErikaRecurveBack_shoot_bow_u_nm_np1_ba_med_0.avi'
    # joint_path = 'inference/test_visualization/11_4_08ErikaRecurveBack_shoot_bow_u_nm_np1_ba_med_0'
    video_path = 'inference/test_visualization/practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_12.avi'
    joint_path = 'inference/test_visualization/practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_12'
    
    test_checkpoint = 'HMR_decoder_initial_run_0.0496.pt'
    if test_checkpoint is None:
        lst = sorted(list(os.listdir(os.path.join(config['checkpoint_directory'], config['checkpoint_name']))))
        test_checkpoint = lst[0]
    print('Chosen checkpoint is', test_checkpoint)
    model_path = os.path.join(
        config['checkpoint_directory'], config['checkpoint_name'], test_checkpoint)

    # load the whole joint file and the video
    joints, frames = get_input_and_label(config['use_videos'], joint_path, video_path)

    width, height = config['image_width'], config['image_height']
    tensor_width, tensor_height = config['image_tensor_width'], config['image_tensor_height']

    if config['full_debug']:
        print('Here are some joint values', joints[0])

    # get the bounding boxes
    if joints_exist:
        bboxes = bounding_box(joints)
    # elsewise, use the yolo algorithm
    else:
        bboxes = inference_yolo_bounding_box(joints)

    if ground_truth:
        # ground truth
        visualize(joints, frames, 'normalized_pull_ups',
                  width, height, bboxes, config['use_last_frame_only'])

    # i'll try to fix just the normal visualize predict
    if predicted:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the model from the .pt file
        model = load_model(config, model_path, config['parallelize'])
        model = model.to(device)

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
    
        frames_per_vid = config['num_frames']

        # all frames, except first 15 (because each video is 16 frames) with 15 joints, and x y
        if config['use_last_frame_only']:
            outputs = torch.zeros(len(frames)-15, 15, 2)
            max_num = len(frames) - 15

        else:
            max_num = len(frames) - (len(frames) % 16)
            # here, the output should be a multiple of 16 
            # outputs = torch.zeros(len(frames) - (len(frames) % 16), 15, 2)
            # final_frames = torch.zeros(len(frames) - (len(frames) % 16), 3, config['image_tensor_height'], config['image_tensor_width'])
            # how about just use an empty torch tensor to which I contatenate values.
            outputs = torch.tensor([])
            final_frames = torch.tensor([])
        # need to reformat the output, find the bounding box, and apply the output
        # If I have the ground truth data, then I will rely on that for the bounding box
        # if only using the last frame
        print(len(list(outputs)), 'is the length of the outputs ')
        for frame in range(15, max_num, jump):
            print(frame, 'is the current frame index')
            input_frame = frames[frame-(frames_per_vid)+1:frame+1]
            input_frame = rearrange(input_frame, 'd c h w -> d h w c')                   

            input_frame, _ = preprocess_video_data(input_frame.numpy(), bboxes.numpy(), joints.numpy(), (tensor_width, tensor_height), config['min_norm'])

            input_frame = rearrange(input_frame, '(b d) c h w -> b c d h w', b=1)
            print(input_frame.shape, 'is the shape of the input frames')

            input_frame = input_frame.to(device)

            # videos.append(input_frame) # need cuda GPU!
            output = inference(model, input_frame)

            output = output.to('cpu')


            if config['use_last_frame_only']:
                # I think output outputs a batch size of 1, so there is one more dimension
                _, output = inverse_process_joint_data(bboxes[frame].numpy(), output[0].numpy(), (tensor_width, tensor_height), config['min_norm'], False)

                outputs[frame-15] = output

            elif better_visualization:
                input_frame = input_frame.to('cpu')
                input_frame = rearrange(input_frame[0], 'c d h w -> d h w c')

                # just denormalize, and nothing else
                output = output[0]
                output = denormalize_fn(output, config['min_norm'], tensor_height, tensor_width)
                input_frame = rearrange(input_frame, 'b h w c->b c h w')
                # for i in range(output.shape[0]):
                #     outputs[frame + i + 1 - frames_per_vid] = output[i]
                #     print('Currently filling: ', frame + i + 1 - frames_per_vid)
                #     final_frames[frame + i + 1 - frames_per_vid] = input_frame[i]
                print(output.shape, 'is the shape of the output of your model')
                outputs = torch.cat((outputs, output))
                final_frames = torch.cat((final_frames, input_frame))

            else:
                input_frame = input_frame.to('cpu')
                input_frame = rearrange(input_frame[0], 'c d h w -> d h w c')
                output = output[0]
                # okay, my inverse process jointdata does NOT work. I don't think it inverses the x values for the width correctly.
                # !Hence, I will just take the processed frames and simply denormalize the joint values, without inverse affine whatever.
                # I think output outputs a batch size of 1, so there is one more dimension
                # input_frame, output = inverse_process_joints_data(bboxes.numpy(), output.numpy(), (tensor_width, tensor_height), config['min_norm'], input_frame.numpy())
                # print(input_frame.shape)
                # exit()
                # input_frame = rearrange(input_frame, 'd h w c -> d c h w ')

                # outputs[frame+1-frames_per_vid:frames+1] = output
                # for i in range(output.shape[0]):
                #     outputs[frame + i + 1 - frames_per_vid] = output[i]
                #     print('Currently filling: ', frame + i + 1 - frames_per_vid)
                #     final_frames[frame + i + 1 - frames_per_vid] = input_frame[i]
                outputs = torch.cat([outputs, output])
                final_frames = torch.cat([final_frames, input_frame])
        print(outputs.shape, 'is the shape of your outputs')
        print('Are the last two outputs the same?: ', outputs[0] == outputs[1])
        # prints the last output
        # print('output', output)

        # visualize
        frames = rearrange(frames, 'd c h w -> d h w c')                   
        frames, test = preprocess_video_data(frames.numpy(), bboxes.numpy(), joints.numpy(), (tensor_width, tensor_height), config['min_norm'])
        if resized_ground_truth:
            # visualizing the joints that were normalized
            test = denormalize_fn(test, config['min_norm'], tensor_height, tensor_width)
            visualize(test, frames, 'aaa', tensor_width, tensor_height, None, False)
            # exit()

        if better_visualization:
            # visualize(outputs, final_frames, 'predicted', width,
            #       height, None, config['use_last_frame_only'])
            print(outputs.shape, 'is the shape of the output')
            
            # outputs = denormalize_fn(outputs, config['min_norm'], tensor_height, tensor_width)
            print(outputs[0])
            visualize(outputs, frames, 'predicted', tensor_width, tensor_height, None, False)

        else:
            visualize(outputs, final_frames, 'predicted', width,
                  height, bboxes, config['use_last_frame_only'])


if __name__ == "__main__":
    # config = open_config(file_name='heatmap_beluga_idapt_local.yaml',
   # importing all possible models:
    from data_format.AffineTransform import denormalize_fn, bounding_box, inference_yolo_bounding_box, inverse_process_joints_data, inverse_process_joint_data, preprocess_video_data
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
