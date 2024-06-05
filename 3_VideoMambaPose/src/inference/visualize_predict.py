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

from import_config import open_config

# config = open_config(file_name='heatmap_beluga_idapt_local.yaml',
#  folder_path='/home/xinlei/Projects/KITE_MambaPose/Video_Pose/3_VideoMambaPose/configs/heatmap')
config = open_config(file_name='heatmap_beluga.yaml',
                     folder_path='/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/configs/heatmap')

# these are hard coded just for ht ecase
sys.path.append(
    '/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap')
sys.path.append('/mnt/DATA/Personnel/Other learning/Programming/Professional_Opportunities/KITE - Video Pose ViT/KITE - Video Pose Landmark Detection/3_VideoMambaPose/src/models/experiments/heatmap')
# change the system directory

sys.path.append(config['project_dir'])

# do not hit ctrl shift -i, or it will put this at the top
from HeatVideoMamba import HeatMapVideoMambaPose
from DataFormat import denormalize_default, det_denormalize_values, denormalize_fn, normalize_fn

def load_model(filepath, parallel=False):
    # Create the model
    model = HeatMapVideoMambaPose(config)

    # load the dictionary from checkpoint, and load the weights into the model.
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

    # TODO fix this, in case
    if parallel:
        checkpoint = adapt_model_parallel(checkpoint)

    # loading this requires me to check from the initial save
    # strict = False makes it so that even though some layer are missing, it will work (although idk why some layesr are missing)
    model.load_state_dict(checkpoint)

    # Set model to evaluation mode
    model.eval()

    return model

# loading model that was trained with DDP


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


def get_input_and_label(joint_path, video_path, normalized=True, default=False, path='/home/linxin67/scratch/JHMDB'):
    # for the sake of testing, I will juhhst hard copy some files and the respective joint outputs
    # I'll also make sure they were in the test files
    joints = scipy.io.loadmat(os.path.join(joint_path, 'joint_positions.mat'))
    if normalized and default:
        joints = torch.tensor(joints['pos_world'])
        print('The joints taken are the normalized ones')
    else:
        joints = torch.tensor(joints['pos_img'])
        print('The joints taken are not normalized')

    joints = rearrange(joints, 'd n f->f n d')
    video = video_to_tensors(config, video_path)

    return joints, video


def image_to_tensor(config, image_path):
    '''Returns a torch tensor for a given image associated with the path'''
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        # notice that all the images are 320x240. Hence, resizing all to 224 224 is generalized, and should be equally skewed
        transforms.ToTensor(),
        transforms.Resize(
            (config['image_tensor_height'], config['image_tensor_width']))
    ])
    tensor = transform(image)
    return tensor


def video_to_tensors(config, video_path='/home/linxin67/scratch/JHMDB/Rename_Images/'):
    '''
    Returns a tensor with the following:
    (n_frames, num_channels (3), 224, 224)
    '''
    # directory_path = os.path.join(path, action, video)
    video_path = video_path
    image_tensors = []

    for filename in os.listdir(video_path):
        file_path = os.path.join(video_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_tensor = image_to_tensor(config, file_path)
            image_tensors.append(image_tensor)

    # Concatenates a sequence of tensors along a new dimension.
    batch_tensor = torch.stack(image_tensors)
    return batch_tensor


def visualize(joints, frames, file_name, width, height, normalized=True):
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

    print('The passed width and height are ', width, height)

    # generate a new folder name
    idx = 1
    while os.path.exists(os.path.join('results', file_name)):
        if idx == 1:
            file_name = str(idx) + file_name
        else:
            file_name = str(idx) + file_name[1:]
        idx += 1
    file_name = os.path.join('results', file_name)
    os.mkdir(file_name)

    # NOTE: even though in the dataloader, I am using a more efficient way, I won't because anyways here, the overhead copying isn't that much
    # Create a video writer to save the output
    for frame_idx in range(num_frames):
        # Get the joints for the current frame
        joints_per_frame = joints[frame_idx]

        # Create a blank 320x240 image (white background)
        image = frames[frame_idx]

        # apply transformation to undo the resize
        transform = transforms.Compose([
            # notice that all the images are 320x240. Hence, resizing all to 224 224 is generalized, and should be equally skewed
            # transforms.ToPILImage(),
            transforms.Resize((height, width)),  # mayb they had the wrong size
            # transforms.ToTensor()
        ])
        image = transform(image)
        image = rearrange(image, 'c w d->w d c')

        # converting and changing to cpu before plotting
        image = image.clone().to('cpu')
        image = image.numpy()

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

        plt.title(f'Frame {frame_idx + 1}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.xlim(0, width)
        plt.ylim(height, 0)  # Invert y-axis to match image coordinate system
        plt.legend()

        # Save the current plot to an image file
        plt.savefig(f'{file_name}/frame_{frame_idx}.png')
        plt.close()

    print(f"Video saved as {file_name}")


def main(config):
    # action_path = 'test_visualization/Pirates_5_wave_h_nm_np1_fr_med_8'
    # joint_path = 'test_visualization/Copy-of-Pirates_5_wave_h_nm_np1_fr_med_8'

    test_checkpoint = 'heatmap_55.9462.pt'
    model_path = os.path.join(
        config['checkpoint_directory'], config['checkpoint_name'], test_checkpoint)

    action_path = 'test_visualization/20_good_form_pullups_pullup_f_nm_np1_ri_goo_2'
    joint_path = 'test_visualization/Copy-of-20_good_form_pullups_pullup_f_nm_np1_ri_goo_2'
    normalized = config['normalized']
    default = config['default']

    print(f'The joints are normalized: {normalized}.\n \
          The joints are normalized with the default: {default}')

    # load the whole joint file and the video
    joints, frames = get_input_and_label(
        joint_path, action_path, normalized, default, model_path)

    # width and height
    # if default and normalized:
    #     width, height = det_denormalize_values(normalized_joints, joints)
    # else:
    width, height = config['image_width'], config['image_height']

    # # denormalization
    # if normalized and config['default']:
    #     # this path is not ready, as I have not tested the scale values.
    #     print('Denormalization might not work due to absence of scale')
    #     joints = denormalize_default(joints, heigth, width)

    #     # actually, no, if not default, since this is ground truth, I just took the pos_world
    #     # joints = denormalize_fn(joints, height, width)

    print(joints[0])

    # ground truth
    visualize(joints, frames, 'normalized_pull_ups',
              width, height, normalized=normalized)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model from the .pt file
    model = load_model(model_path, config['parallelize'])
    model = model.to(device)

    print(model)

    frames_per_vid = 16
    # all frames, except first 15 (because each video is 16 frames) with 15 joints, and x y
    outputs = torch.zeros(len(frames)-15, 15, 2)
    for frame in range(15, len(frames)):
        input_frame = frames[frame-(frames_per_vid)+1:frame+1]

        input_frame = rearrange(input_frame, 'd c h w -> c d h w')

        input_frame = input_frame.to(device)

        # videos.append(input_frame) # need cuda GPU!
        output = inference(model, input_frame)

        outputs[frame-15] = output

    # outputs = torch.as_tensor(outputs)``

    # prints the last output
    print('output', output)

    # here, we should denormalize the outputs
    if normalized:
        if config['default']:
            # this path is not ready, as I have not tested the scale values.
            print('Denormalization might not work due to absence of scale')
            outputs = denormalize_default(joints, heigth, width)

        # actually, no, if not default, since this is ground truth, I just took the pos_world
        else:
            outputs = denormalize_fn(outputs, height, width)

    # visualize
    visualize(outputs, frames, 'predicted', width,
              height, normalized=normalized)


if __name__ == "__main__":
    main(config)
