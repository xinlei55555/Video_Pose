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

# change the system directory
sys.path.append(
    '/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap')
sys.path.append('/mnt/DATA/Personnel/Other learning/Programming/Professional_Opportunities/KITE - Video Pose ViT/KITE - Video Pose Landmark Detection/3_VideoMambaPose/src/models/experiments/heatmap')
from HeatVideoMamba import HeatMapVideoMambaPose


def load_model(filepath):
    # Create the model
    model = HeatMapVideoMambaPose()

    # load the dictionary from checkpoint, and load the weights into the model.
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

    # loading this requires me to check from the initial save
    # strict = False makes it so that even though some layer are missing, it will work (although idk why some layesr are missing)
    model.load_state_dict(checkpoint)

    # Set model to evaluation mode
    model.eval()

    return model


def inference(model, input_tensor):
    # Disable gradient computation for inference
    with torch.no_grad():
        output = model(input_tensor)

    return output

# i'll finish the code on my local machine


def get_input_and_label(joint_path, video_path, path='/home/linxin67/scratch/JHMDB'):
    # for the sake of testing, I will just hard copy some files and the respective joint outputs
    # I'll also make sure they were in the test files
    joints = scipy.io.loadmat(joint_path+'/joint_positions.mat')
    joints = torch.tensor(joints['pos_img'])
    joints = rearrange(joints, 'd n f->f n d')

    video = video_to_tensors(video_path)

    return joints, video


def image_to_tensor(image_path):
    '''Returns a torch tensor for a given image associated with the path'''
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        # notice that all the images are 320x240. Hence, resizing all to 224 224 is generalized, and should be equally skewed
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(image)
    return tensor


def video_to_tensors(video_path='/home/linxin67/scratch/JHMDB/Rename_Images/'):
    '''
    Returns a tensor with the following:
    (n_frames, num_channels (3), 224, 224)
    '''
    # directory_path = os.path.join(path, action, video)
    video_path = video_path
    image_tensors = []

    for filename in os.listdir(video_path   ):
        file_path = os.path.join(video_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_tensor = image_to_tensor(file_path)
            image_tensors.append(image_tensor)

    # Concatenates a sequence of tensors along a new dimension.
    batch_tensor = torch.stack(image_tensors)
    return batch_tensor


def visualize(joints, frames, file_name):
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
    # ground truth:
    # Extract x and y values

    # Sample PyTorch tensor with joint coordinates (x, y) for each frame
    # Replace this with your actual tensor
    num_frames = min(len(list(joints)), len(list(frames)))  # Number of frames in the video

    width = 320
    height = 240

    # width = 720
    # height = 512
    # width = 480
    # height= 360

    os.mkdir(file_name)

    # Create a video writer to save the output
    video_writer = cv2.VideoWriter(
        f'{file_name}{width}x{height}/{file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), num_frames, (width, height))
    for frame_idx in range(num_frames):
        # Get the joints for the current frame

        joints_per_frame = joints[frame_idx]

        # Create a blank 320x240 image (white background)
        image = frames[frame_idx]

        # apply transformation to undo the resize
        transform = transforms.Compose([
            # notice that all the images are 320x240. Hence, resizing all to 224 224 is generalized, and should be equally skewed
            transforms.ToPILImage(),
            transforms.Resize((height, width)), # mayb they had the wrong size
            transforms.ToTensor()
        ])
        image = transform(image)
        image = rearrange(image, 'c w d->w d c')

        # converting and changing to cpu before plotting
        image = image.clone().to('cpu')
        image = image.numpy()

        joints_per_frame = joints_per_frame.clone().to('cpu')
        joints_per_frame = joints_per_frame.numpy()

        # Plotting the joints onto the image
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        # testing for if its the normalization that made the values very
        # plt.scatter(joints_per_frame[:, 0]*320, joints_per_frame[:, 1]*240, c='red', s=10, label='Joints')
        plt.scatter(joints_per_frame[:, 0], joints_per_frame[:, 1], c='red', s=10, label='Joints')


        # Annotate the joints with their indices
        for i, pos in enumerate(joints_per_frame):
            # print('pos ', pos)
            x, y = pos[0], pos[1]
            plt.text(x, y, str(i+1), color="blue",
                     fontsize=12, ha='right', va='bottom')
            

        plt.title(f'Frame {frame_idx + 1}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.xlim(0, width)
        plt.ylim(height, 0)  # Invert y-axis to match image coordinate system
        plt.legend()

        # Save the current plot to an image file
        plt.savefig(f'{file_name}/frame_{frame_idx}.png')
        plt.close()

        # Read the saved image
        frame_image = cv2.imread(f'{file_name}/frame_{frame_idx}.png')

        # Ensure the size is correct
        frame_image = cv2.resize(frame_image, (width, height))
# 
        # Write the frame to the video
        video_writer.write(frame_image)

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {file_name}.mp4")


def main():
    # model_path = '/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap/checkpoints/heatmap_22069.0820.pt'
    # model_path = '/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap/checkpoints/heatmap_27345.4473.pt'
    model_path='/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap/checkpoints/heatmap_8652135.9131.pt'
    # model_path = '/mnt/DATA/Personnel/Other learning/Programming/Professional_Opportunities/KITE - Video Pose ViT/KITE - Video Pose Landmark Detection/3_VideoMambaPose/src/models/experiments/heatmap/checkpoints/heatmap_8639121.0703.pt'
    action_path = 'test_visualization/Pirates_5_wave_h_nm_np1_fr_med_8'
    joint_path = 'test_visualization/Copy-of-Pirates_5_wave_h_nm_np1_fr_med_8'
    
    # load the whole joint file and the video
    joints, frames = get_input_and_label(joint_path, action_path, model_path)

    print(joints)

    # ground truth
    visualize(joints, frames, 'ground_truth_pirate')

    return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model from the .pt file
    model = load_model(model_path)
    model = model.to(device)

    print(model)

    frames_per_vid = 16
    outputs = torch.zeros(len(frames)-15, 15, 2) # all frames, except first 15 (because each video is 16 frames) with 15 joints, and x y
    for frame in range(15, len(frames)):
        input_frame = frames[frame-(frames_per_vid)+1:frame+1]

        input_frame = rearrange(input_frame, 'd c h w -> c d h w')

        input_frame = input_frame.to(device)          
        
        # videos.append(input_frame) # need cuda GPU!
        output = inference(model, input_frame)

        outputs[frame-15] = output

    # outputs = torch.as_tensor(outputs)

    # prints the last output
    print('output', output)

    # visualize
    visualize(outputs, frames, 'predicted')



if __name__ == "__main__":
    main()
