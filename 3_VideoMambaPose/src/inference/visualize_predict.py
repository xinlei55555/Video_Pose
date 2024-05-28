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
    # ground truth:
    # Extract x and y values

    # Sample PyTorch tensor with joint coordinates (x, y) for each frame
    # Replace this with your actual tensor
    num_frames = min(len(list(joints)), len(list(frames)))  # Number of frames in the video

    # Create a video writer to save the output
    video_writer = cv2.VideoWriter(
        f'{file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), num_frames, (320, 240))

    for frame_idx in range(num_frames):
        # Get the joints for the current frame

        joints_per_frame = joints[frame_idx].numpy()

        # Create a blank 320x240 image (white background)
        image = frames[frame_idx]

        # apply transformation to undo the resize
        transform = transforms.Compose([
            # notice that all the images are 320x240. Hence, resizing all to 224 224 is generalized, and should be equally skewed
            transforms.ToPILImage(),
            transforms.Resize((240, 320)),
            transforms.ToTensor()
        ])
        image = transform(image)
        image = rearrange(image, 'c w d->w d c').numpy()

        # Plotting the joints onto the image
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.scatter(joints_per_frame[:, 0], joints_per_frame[:, 1], c='red', s=10, label='Joints')

        # Annotate the joints with their indices
        for i, pos in enumerate(joints_per_frame):
            # print('pos ', pos)
            x, y = pos[0], pos[1]
            plt.text(x, y, str(i), color="blue",
                     fontsize=12, ha='right', va='bottom')

        plt.title(f'Frame {frame_idx + 1}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.xlim(0, 320)
        plt.ylim(240, 0)  # Invert y-axis to match image coordinate system
        plt.legend()

        # Save the current plot to an image file
        plt.savefig('frame.png')
        plt.close()

        # Read the saved image
        frame_image = cv2.imread('frame.png')
        # Ensure the size is correct
        frame_image = cv2.resize(frame_image, (320, 240))

        # Write the frame to the video
        video_writer.write(frame_image)

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {file_name}.mp4")


def main():
    # model_path = '/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap/checkpoints/heatmap_22069.0820.pt'
    # model_path = '/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap/checkpoints/heatmap_27345.4473.pt'
    # model_path='/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap/checkpoints/heatmap_8652135.9131.pt'
    model_path = '/mnt/DATA/Personnel/Other learning/Programming/Professional_Opportunities/KITE - Video Pose ViT/KITE - Video Pose Landmark Detection/3_VideoMambaPose/src/models/experiments/heatmap/checkpoints/heatmap_8639121.0703.pt'
    action_path = '/mnt/DATA/Personnel/Other learning/Programming/Professional_Opportunities/KITE - Video Pose ViT/KITE - Video Pose Landmark Detection/3_VideoMambaPose/data/JHMDB/test_visualization/20_good_form_pullups_pullup_f_nm_np1_ri_goo_2'
    joint_path = '/mnt/DATA/Personnel/Other learning/Programming/Professional_Opportunities/KITE - Video Pose ViT/KITE - Video Pose Landmark Detection/3_VideoMambaPose/data/JHMDB/test_visualization/Copy-of-20_good_form_pullups_pullup_f_nm_np1_ri_goo_2'
    # Load the model from the .pt file
    model = load_model(model_path)
    print(model)

    # load the whole joint file and the video
    joints, frames = get_input_and_label(joint_path, action_path, model_path)

    # ground truth
    # visualize(joints, frames, 'ground_truth')

    frames_per_vid = 16
    joints_outputted = []
    for frame in range(15, len(frames)):
        input_frame = frames[frame-(frames_per_vid)+1:frame+1]

        input_frame = rearrange(input_frame, 'd c h w -> c d h w')         
        
        # videos.append(input_frame) # need cuda GPU!
        output = inference(model, input_frame)

    # predict
    print('output', output)

    # visualize
    visualize(output, frames, 'predicted')



if __name__ == "__main__":
    main()
