import torch
import torch.nn as nn
import os

import pickle
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import scipy

from einops import rearrange

import matplotlib
from matplotlib import pyplot as plt


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
    def __init__(self, train_set=True, frames_per_vid=16, joints=True, unpickle=True, real_job=True, jump=1):
        self.frames_per_vid = frames_per_vid
        self.train_set = train_set

        # determines whether to take the whole set of data, or just part of it
        self.real_job = real_job

        if unpickle:
            # if self.train_set:
            self.annotations = self.unpickle_JHMDB()
            self.nframes = self.annotations['nframes']
            # else:
            #     self.test_annotations = self.unpickle_JHMDB()
            #     # this is another dictionary
            #     # I also reformatted the dictionary nframes:video_name
            #     self.nframes_test = self.test_annotations['nframes']

        if joints:
            if self.train_set:
                self.actions, self.train, _ = self.get_names_train_test_split()
            else:
                self.actions, _, self.test = self.get_names_train_test_split()

        self.arr = []
        if self.train_set:
            # frames with joint values
            self.frames_with_joints = [(self.video_to_tensors(
                                                action_name, file_name),
                                            self.rearranged_joints(
                                                action_name, file_name))
                                            for action_name, file_name, n_frames in self.train]
            # arr where arr[idx] = idx in the self.frames_with_joints
            self.jump=jump# this is the number of frames to skip between datapoints
            for k in range(len(self.frames_with_joints)):
                video, joints = self.frames_with_joints[k]

                if len(list(video)) != len(list(joints)):
                    print('Wrong length! Video: ', len(list(video)))
                    print('Joints: ', len(list(joints)))
                
                else:
                    # going through each frame in the video
                    for i in range(self.frames_per_vid, len(list(video)), self.jump): 
                        # 3-tuple: (index in self.train_frames_with_joints, index in the video, joint values)
                        self.arr.append([k, i, joints])
        else:
            self.frames_with_joints = [(self.video_to_tensors(
                                                action_name, file_name),
                                            self.rearranged_joints(
                                                action_name, file_name))
                                            for action_name, file_name, n_frames in self.test]

            self.jump = jump # this is the partition/the number of frames skipped between each video
            for k in range(len(self.frames_with_joints)):
                video, joints = self.frames_with_joints[k]

                # and if such an occurence happens, then SKIP the file!!!!
                if len(list(video)) != len(list(joints)):
                    print('Wrong length! Video: ', len(list(video)))
                    print('Joints: ', len(list(joints)))
                    

                else:
                    # start looping at frames per vid number
                    for i in range(self.frames_per_vid, len(list(video)), self.jump): # if you are using jump, then need to define start and endpoint
                            # 3-tuple: (index in self.train_frames_with_joints, index in the video, joint values)
                        self.arr.append([k, i, joints])

    # some default torch methods:
    def __len__(self):
        # if self.train_set:
        #     return len(self.train_arr)
        return len(list(self.arr))

    # should return the image/video at that index, as well as the label for the video. (Should I make a sliding window, or a striding window and return the value of the first frame)
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

        #!!!! TODOOOO I THink there is an index error hereeee
        video = self.frames_with_joints[video_num][0][frame_num+1-self.frames_per_vid:frame_num+1]
        # video = torch.tensor() # I think it was already a toch tensor
        video = rearrange(video, 'd c h w -> c d h w') # need to rearrange so that channel number is in front.
        # print('The shape of the video is', video.shape)
        # torch.Size([16, 3, 224, 224]) -> torch.Size([3, 16, 224, 224])

        # show this for debug:
        # print(f'index: {index}, video_num: {video_num}, frame_num: {frame_num}, len(joint_values), {len(list(joint_values))}')
        return [video, joint_values[frame_num]]

        
    # this folder is useless
    def unpickle_JHMDB(self, path="/home/linxin67/scratch/JHMDB_old/annotations"):
        os.chdir(path)  

        # Open the first pickled file
        # with open('JHMDB-GT.pkl', 'rb') as pickled_one:
        
        with open("JHMDB-GT.pkl", 'rb') as pickled_one:
            try:
                # other times it is 'utf-8!!!
                train = pickle.load(pickled_one, encoding='latin1')
            except UnicodeDecodeError as e:
                print("UnicodeDecodeError:", e)
        return train

        # else:
        #     # Open the second pickled file
        #     with open('UCF101v2-GT.pkl', 'rb') as pickled_two:
        #         try:
        #             # other times it is 'utf-8!!!
        #             test = pickle.load(pickled_two, encoding='latin1')

        #         except UnicodeDecodeError as e:
        #             print("UnicodeDecodeError:", e)
        #     return test

    # first a function that given an action and a video returns the joints for each frame
    def read_joints_full_video(self, action, video, path="/home/linxin67/scratch/JHMDB/"):
        '''Returns a dictionary
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
        torch_joint = torch.tensor(joint_dct['pos_img'])
        torch_joint = rearrange(torch_joint, 'd n f->f n d')
        return torch_joint

    def get_num_frames(self, action, video):
        # os.chdir(path+'annotations')
        # if self.train_set and action+'/'+video in self.nframes_train:
        return self.nframes[action+'/'+video]
        # else:
        #     return self.nframes_test[action+'/'+video]

    def get_names_train_test_split(self, path="/home/linxin67/scratch/JHMDB/"):
        '''
        Returns three lists:
        1. First one with all the possible actions.
        2. The training set (with 3-tuple: (action_name, file_name, n_frames))
        3. The test set (idem)
        '''
        directory = path+'splits'
        os.chdir(directory)

        actions = []
        train = []
        test = []

        value = True
        # looping through gives you each action
        for action in os.listdir(directory):
           # I just want to look at the ones with 1 after.
            if action[-5] != '1':
                continue

            # value only becomes False if I break it inside
            if not value:
                break

            action_split = os.path.join(directory, action)
            actions.append(action[:-16])  # remove the _test_split<int>.txt
            # checking if it is a file
            if os.path.isfile(action_split):
                df = pd.read_csv(action_split, sep=' ', header=None)
                # looping and separating
                for index, row in df.iterrows():
                    file_name = row[0]
                    value = int(row[1])
                    
                    # if only testing, then just take the minimum number of actions
                    if not self.real_job and (len(train) > 20 or len(test) > 1 or len(actions) > 1):
                        print("length of actions", len(actions))
                        value = False
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
        print("The length of actions, train and test are", len(actions), ", ", len(train), ", ", len(test))
        return actions, train, test

        # looping through each split of each action gives you the name of each video.

        # I'll load all of that into one file, which has action, video, train/test on each row

    # then we need a function to crop the images to 224x224, and need to generate batches of 8 frames in a row. (videos)
    # this will also need to transform the images into three channels (use torch.vision transforms.)

    def image_to_tensor(self, image_path):
        '''Returns a torch tensor for a given image associated with the path'''
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            # notice that all the images are 320x240. Hence, resizing all to 224 224 is generalized, and should be equally skewed
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        tensor = transform(image)
        return tensor

    def video_to_tensors(self, action, video, path='/home/linxin67/scratch/JHMDB/Rename_Images/'):
        '''
        Returns a tensor with the following:
        (n_frames, num_channels (3), 224, 224)
        '''
        directory_path = os.path.join(path, action, video)
        image_tensors = []

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_tensor = self.image_to_tensor(file_path)
                image_tensors.append(image_tensor)

        # Concatenates a sequence of tensors along a new dimension.
        batch_tensor = torch.stack(image_tensors)
        return batch_tensor

    #! this function has not been tested, but since I am not yet in production, I will test it tomorrow.
    def draw_joint_on_image(self, action, video, frame_number=1, path='/home/linxin67/scratch/JHMDB/'):
        video = self.video_to_tensors(action, video)
        frame = video[frame_number]

        # print(frame.shape) # torch.Size([3, 224, 224])

        # transform back, note that if the input is a torch tensor, then no need to use compose
        transform = transforms.Compose([
            # need to first transform the torch tensor to pil image.
            transforms.ToPILImage(),
            # should be height, width
            # notice that all the images are 320x240. Hence, resizing all to 224 224 is generalized, and should be equally skewed
            transforms.Resize((240, 320)),
            transforms.ToTensor()
        ])

        frame = transform(frame)

        # so need to rearrange to h, w, c or else plt won't work
        frame = frame.permute(1, 2, 0)
        plt.imshow(frame)
        plt.show()

        # adding the joints
        draw = ImageDraw.Draw(frame_pil)

        joints = self.rearranged_joints(action, video, path)
        joint = joints[frame_number]  # num_joints, [x,y]

        rearrange(joint, 'n x -> x n')
        x_coords, y_coords = joint[0], joint[1]

        plt.figure()
        plt.scatter(x_coords, y_coords, c='red', marker='o')
        plt.xlim(0, 320)
        plt.ylim(240, 0)  # Invert y-axis for correct orientation
        plt.xlabel('X Coordinates')
        plt.ylabel('Y Coordinates')
        plt.title('Joint Coordinates')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    train = JHMDBLoad(train_set=True, frames_per_vid=16, joints=True, unpickle=True, real_job=False)
    # in real context, would definitely need to move the training set in the GPU
    print(train.arr)
    print("len(train), ", len(train))
    print("len(arr), ", len(train.arr))
    print("len(frames_with_joints)", len(train.frames_with_joints))
    print(train[len(train)-1])
    print(len(train[len(train)-1][0]))

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
