import torch
import torch.nn as nn
import os

import pickle
import pandas as pd
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import scipy

from einops import rearrange

import matplotlib
from matplotlib import pyplot as plt

from import_config import open_config

# Defined as global functions, to be able to use them without initializing JHMDBLoad (notably during inference)
def normalize_fn(x, h=240.0, w=320.0):
    # x has num_frames, joint_numbers, (x, y)
    x[:, :, 0] = (x[:, :, 0] / (w / 2.0)) - 1.0  # bewteen -1 and 1
    x[:, :, 1] = (x[:, :, 1] / (h / 2.0)) - 1.0
    return x


def denormalize_fn(x, h=240.0, w=320.0):
    # actually, you should do denormalization AFTER the loss funciton. so when doing inference.
    x[:, :, 0] = (x[:, :, 0] + 1.0) * (w / 2.0)  # bewteen -1 and 1
    x[:, :, 1] = (x[:, :, 1] + 1.0) * (h / 2.0)
    return x


def denormalize_default(x, h=240.0, w=320.0, scale=1):
    # since the data was normalize with respect to the w, and h, then if I use my new width, h, would it change somethign?
    '''
    Default initial normalizer:
    (4) pos_world is the normalization of pos_img with respect to the frame size and puppet scale,Â
        the formula is as below

        pos_world(1,:,:) = (pos_img(1,:,:)/W-0.5)*W/H    ./scale;
        pos_world(2,:,:) = (pos_img(2,:,:)/H-0.5)       ./scale;
    '''
    x[:, :, 0] = (x[:, :, 0] * scale / (w/h) + 0.5) * w  # bewteen -0.5 and 0.5
    x[:, :, 1] = (x[:, :, 1] * scale + 0.5) * h
    return x


def det_denormalize_values(x_norm, x_init, scale):
    print(x_init.shape, x_norm.shape)
    h = int(x_init[0][1][1].item() / (0.5 + x_norm[0][1][1].item()))
    w = int(x_init[0][1][0].item() * 2 - 2 * x_norm[0][1][0].item() * h)
    # to change later
    w = abs(w)
    h = abs(h)
    return w, h


class JHMDBLoad(Dataset):
    '''
    train_annotations: training set annotations
    test_annotations: testing set annotations
        dict_keys(['labels', 'gttubes', 'nframes', 'train_videos', 'test_videos', 'resolution'])
        labels are the labels that are int he actions
        gttubes == ground truth tubes
        nframes is the number of frames

    '''

    # use 16, because transformers can already do 8
    # also we cannot just load all the frames directly into memory, because not enough GPU, but here less than 64GB should be okay
    def __init__(self, config, train_set, real_job=True, jump=1, normalize=(True, True)):
        self.config = config
        self.skip = config['skip']
        self.use_videos = config['use_videos']
        self.normalized, self.default = normalize
        self.frames_per_vid = self.config['num_frames']
        self.train_set = train_set

        # determines whether to take the whole set of data, or just part of it
        self.real_job = real_job

        self.annotations = self.unpickle_JHMDB()
        self.nframes = self.annotations['nframes']

        if self.train_set:
            self.actions, self.train, _ = self.get_names_train_test_split()
        else:
            self.actions, _, self.test = self.get_names_train_test_split()

        # I will remove all 'wave' actions, because the data seems corrupted
        self.arr = []
        if self.train_set:
            # frames with joint values
            self.frames_with_joints = [(self.video_to_tensors(
                action_name, file_name, self.use_videos),
                self.rearranged_joints(
                action_name, file_name))
                for action_name, file_name, n_frames in self.train if action_name != 'wave']
            # arr where arr[idx] = idx in the self.frames_with_joints
            self.jump = jump
            for k in range(len(self.frames_with_joints)):
                video, joints = self.frames_with_joints[k]

                if len(list(video)) != len(list(joints)):
                    print('Wrong length! Video: ', len(list(video)))
                    print('Joints: ', len(list(joints)))

                else:
                    for i in range(self.frames_per_vid, len(list(video)), self.jump):
                        # 3-tuple: (index in self.train_frames_with_joints, index in the video, joint values)
                        self.arr.append([k, i, joints])
        else:
            self.frames_with_joints = [(self.video_to_tensors(
                action_name, file_name, self.use_videos),
                self.rearranged_joints(
                action_name, file_name))
                for action_name, file_name, n_frames in self.test if action_name != 'wave']

            self.jump = jump
            for k in range(len(self.frames_with_joints)):
                video, joints = self.frames_with_joints[k]

                # and if such an occurence happens, then SKIP the file!!!!
                if len(list(video)) != len(list(joints)):
                    print('Wrong length! Video: ', len(list(video)))
                    print('Joints: ', len(list(joints)))

                else:
                    # start looping at frames per vid number
                    # if you are using jump, then need to define start and endpoint
                    for i in range(self.frames_per_vid, len(list(video)), self.jump):
                        # 3-tuple: (index in self.train_frames_with_joints, index in the video, joint values)
                        self.arr.append([k, i, joints])

    def __len__(self):
        return len(list(self.arr))

    def __getitem__(self, index):
        '''
        Each file will have nframe-self.n_frames_in_vid datapoints. Each video will be 8 frames long.
        The goal will always be to predict the last frame of the 8.
        To determine the index:
        1. I will load all the frames.
        2. Then, I will generate the 8 previous frames for a given index on the fly. 


        #
        Answer: I will load everything, and parse from there.
        '''
        video_num, frame_num, joint_values = self.arr[index][0], self.arr[index][1], self.arr[index][2]

        # slicing with pytorch tensors.
        video = self.frames_with_joints[video_num][0][frame_num +
                                                      1-self.frames_per_vid:frame_num+1]

        # need to rearrange so that channel number is in front.
        video = rearrange(video, 'd c h w -> c d h w')

        # show this for debug:
        if self.config['full_debug']:
            print('The shape of the video is', video.shape)
            print(
                f'index: {index}, video_num: {video_num}, frame_num: {frame_num}, len(joint_values), {len(list(joint_values))}')

        return [video, joint_values[frame_num]]

    # this folder is useless
    def unpickle_JHMDB(self, path="/home/linxin67/scratch/JHMDB_old/annotations"):
        os.chdir(path)

        # Open the first pickled file
        with open("JHMDB-GT.pkl", 'rb') as pickled_one:
            try:
                # other times it is 'utf-8!!!
                train = pickle.load(pickled_one, encoding='latin1')
            except UnicodeDecodeError as e:
                print("UnicodeDecodeError:", e)
        return train

    def read_joints_full_video(self, action, video, path="/home/linxin67/scratch/JHMDB/"):
        '''
        Given an action and a video returns the joints for each frame

        Returns a dictionary
        dict_keys(['__header__', '__version__', '__globals__', 'pos_img', 'pos_world', 'scale', 'viewpoint'])

        Each file is the following dimension:
        (2, 15 (num joints), n_frames)

        First, there are two dimension, which is x, y
        Then, 
        In pos image, each array has n number of values, where n is the number of frames in the video.
        '''
        os.chdir(path)
        mat = scipy.io.loadmat(
            f'{path}joint_positions/{action}/{video}/joint_positions.mat')
        return mat

    def rearranged_joints(self, action, video, path='/home/linxin67/scratch/JHMDB/'):
        '''
        Return a torch tensor with num frames, num joints, (x,y) joints.
        '''
        joint_dct = self.read_joints_full_video(action, video, path)

        # pos_world was already normalized with respect to the image. (unlike pos_img)
        if self.normalized and self.default:
            torch_joint = torch.tensor(joint_dct['pos_world'])
            torch_joint = rearrange(torch_joint, 'd n f->f n d')
        # then use custom normalization
        elif self.normalized and not self.default:
            torch_joint = torch.tensor(joint_dct['pos_img'])
            torch_joint = rearrange(torch_joint, 'd n f->f n d')
            torch_joint = normalize_fn(torch_joint)
        # then no normalization
        else:
            torch_joint = torch.tensor(joint_dct['pos_img'])
            # rearrange for training and normalization.
            torch_joint = rearrange(torch_joint, 'd n f->f n d')

        if self.config['full_debug']:
            print(f'normalized: {self.normalized}, default: {self.default}')
            print('example joint values', torch_joint[0][0])

        return torch_joint

    def get_num_frames(self, action, video):
        return self.nframes[os.path.join(action, video)]

    def get_names_train_test_split(self, path="/home/linxin67/scratch/JHMDB/"):
        '''
        Returns three lists:
        1. First one with all the possible actions.
        2. The training set (with 3-tuple: (action_name, file_name, n_frames))
        3. The test set (idem)
        '''
        directory = os.path.join(path, 'splits')
        os.chdir(directory)

        actions = []
        train = []
        test = []

        stop_list = True
        # looping through gives you each action
        for action in os.listdir(directory):
           # I just want to look at the ones with 1 after.
            if action[-5] != '1':
                continue

            # value only becomes False if I break it inside
            if not stop_list:
                break

            action_split = os.path.join(directory, action)

            # skipping all actions that are in the blacklist.
            if action[:-16] in self.skip:
                continue
            actions.append(action[:-16])  # remove the _test_split<int>.txt

            # checking if it is a file
            if os.path.isfile(action_split):
                df = pd.read_csv(action_split, sep=' ', header=None)
                # looping and separating
                for index, row in df.iterrows():
                    file_name = row[0]
                    value = int(row[1])

                    # if only testing, then just take the minimum number of actions
                    if not self.real_job and (len(train) > 20 or len(test) > 1 or (len(actions) > 1)):
                        print("length of actions", len(actions))
                        stop_list = False
                        break

                    if value == 1 and self.train_set:
                        # remove the .avi
                        train.append(
                            (action[:-16], file_name[:-4], self.get_num_frames(action[:-16], file_name[:-4])))
                    elif value == 2 and not self.train_set:
                        test.append(
                            (action[:-16], file_name[:-4], self.get_num_frames(action[:-16], file_name[:-4])))
                    elif value not in [1, 2]:
                        print(type(value), value)
                        print("unknownIndexError")

        print("The length of actions, train and test are",
              len(actions), ", ", len(train), ", ", len(test))
        return actions, train, test

    def image_to_tensor(self, image_path):
        '''Returns a torch tensor for a given image associated with the path'''
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            # notice that all the images are 320x240. Hence, resizing all to 224 224 is generalized, and should be equally skewed
            transforms.Resize(
                (self.config['image_tensor_height'], self.config['image_tensor_width']), antialias=True),
            transforms.ToTensor()
        ])
        tensor = transform(image)
        return tensor

    def video_to_tensors(self, action, video, use_videos, path='/home/linxin67/scratch/JHMDB/'):
        '''
        Returns a tensor with the following:
        (n_frames, num_channels (3), 224, 224)
        '''

        # goes through the each image.
        # !!!! do not use IMAGES BECAUSE OS.LISTDIR DOES NOT GO THROUGH THEM IN THE RIGHT ORDER!
        if not use_videos:
            video_path = os.path.join(path, 'Rename_Images', action, video)
            image_tensors = []

            filenames = []
            for filename in os.listdir(video_path):
                if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filenames.append(filename)
            filenames.sort()
            for filename in filenames:
                file_path = os.path.join(video_path, filename)
                image_tensor = self.image_to_tensor(file_path)
                image_tensors.append(image_tensor)

            # Concatenates a sequence of tensors along a new dimension.
            batch_tensor = torch.stack(image_tensors)
            
        else:
            # Check if the video file exists
            video = video + '.avi' # adding the .avi extension.
            video_path = os.path.join(path, 'ReCompress_Videos', action, video)

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

            # Transpose the tensor to have the shape (frames, channels, height, width)
            batch_tensor = batch_tensor.permute(0, 3, 1, 2)

            # apply the transformation to the whole video (batched) input must be (B, C, H, W)
            transform = transforms.Compose([transforms.Resize(
                (self.config['image_tensor_height'], self.config['image_tensor_width']), antialias=True)])
            
            batch_tensor = transform(batch_tensor)

        return batch_tensor


if __name__ == '__main__':
    config = open_config()
    num_epochs = config['epoch_number']
    batch_size = config['batch_size']
    num_workers = config['num_cpus'] - 1
    normalize = config['normalized']
    default = config['default']  # custom normalization.
    follow_up = (config['follow_up'], config['previous_training_epoch'],
                 config['previous_checkpoint'])
    jump = config['jump']
    real_job = config['real_job']
    checkpoint_dir = config['checkpoint_directory']
    checkpoint_name = config['checkpoint_name']

    train = JHMDBLoad(config, train_set=True, real_job=real_job,
                      jump=jump, normalize=(True, False))
    print(train.arr)
    train_loader = DataLoader(train, batch_size=16,
                              shuffle=True, num_workers=2, pin_memory=False)
    # in real context, would definitely need to move the training set in the GPU
    # print(train.arr)
    # print("len(train), ", len(train))
    # print("len(arr), ", len(train.arr))
    # print("len(frames_with_joints)", len(train.frames_with_joints))
    # print(train[len(train)-1])
    # print(len(train[len(train)-1][0]))
    # print(len(train[len(train)-1][1]))

    # test = JHMDBLoad(train_set=False)
    # print(test[len(test)-1])
    # print(len(test))
    # print(test[len(test)-1].shape)
    # print([type(x) for x in data.train_annotations.keys()]) # a list of strings.
    # print(data.test_annotations.keys()) # a dictionary

    # dict_keys(['labels', 'gttubes', 'nframes', 'train_videos', 'test_videos', 'resolution'])
    # labels are the labels that are int he actions
    # gttubes == ground truth tubes
    # nframes is the number of frames

    # when I print the data[ggtubes], i get a dictionary with many elements.
    # Ithink the first position of the array is the frame number.

    # print(type(data.train_annotations['gttubes'])) # notice that this is ANOTHER dictionary
    # print(data.train_annotations['gttubes'].keys()) # each key represents a video!

    # print(data.train_annotations['gttubes']['pour/Bartender_School_Students_Practice_pour_u_cm_np1_fr_med_1'][8]) # and I think each video is annotated such that the first index is the joint value, and the rest of the array are the positions

    # this is another dictionary.
    # print(type(data.get_names_train_test_split()))
    # print(len(data.get_names_train_test_split()[0]), len(
    #     data.get_names_train_test_split()[1]), len(data.get_names_train_test_split()[2]))
    # print(data.get_names_train_test_split()[1])

    # example_joint = data.rearranged_joints(action='pour', video='Bartender_School_Students_Practice_pour_u_cm_np1_fr_med_1')
    # print(example_joint)
    # print(example_joint.shape)
    # print(len(example_joint['pos_img'][0][0]))
    # print(type(example_joint))
    # print(example_joint.keys())
    # print(data.get_num_frames(action='pour', video='Bartender_School_Students_Practice_pour_u_cm_np1_fr_med_1'))

    # test = data.video_to_tensors(action='pour', video='Bartender_School_Students_Practice_pour_u_cm_np1_fr_med_1')
    # print(test.shape)
    # print(test[0])

    # data.draw_joint_on_image(
    #     action='pour', video='Bartender_School_Students_Practice_pour_u_cm_np1_fr_med_1')

    # then the rest is handled by pytorch:
    # Example usage
    # data = [...]  # Your dataset (list of samples)
    # custom_dataset = CustomDataset(data)
    # data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

    # # Iterate over batches of data
    # for batch in data_loader:
    #     # Process each batch of data
    #     pass
