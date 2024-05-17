import torch
import torch.nn as nn
import os

import pickle
import pandas as pd

from torch.utils.data import Dataset, DataLoader

import scipy

from einops import rearrange


class load_JHMDB(Dataset):
    '''
    train_annotations: training set annotations
    test_annotations: testing set annotations
        dict_keys(['labels', 'gttubes', 'nframes', 'train_videos', 'test_videos', 'resolution'])
        labels are the labels that are int he actions
        gttubes == ground truth tubes
        nframes is the number of frames

    '''

    def __init__(self, joints=False, unpickle=False):
        if unpickle:
            self.train_annotations, self.test_annotations = self.unpickle_JHMDB()
        # this is another dictionary
        self.nframes_train = self.train_annotations['nframes']
        self.nframes_test = self.test_annotations['nframes']

        # in the end, implement this
        if joints:
            self.actions, self.train, self.test = self.get_names_train_test_split()
            # self.joints = 
        #     self.joint_values = torch.zeros()
        #     # collect all joint values. In the same way, collect all the titles
        #     # store them into csv files or wtv. with the joint values, train test;

        #     # store the images in torch format.

        #     self.joints = self.read_joints()

    # some default torch methods:
    def __len__(self):
        return len(self.train) + len(self.test);

    # should return the image/video at that index, as well as the label for the video. (Should I make a sliding window, or a striding window and return the value of the first frame)
    def __getitem__(self, index):
        '''
        Each file will have nframe-7 datapoints. Each video will be 8 frames long.
        The goal will always be to predict the last frame of the 8.
        To determine the index:
        1. I will load all the frames.
        2. Then, I will generate the 8 previous frames for a given index on the fly. 
        '''
        pass
        # if index < 
        # return 

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

        # Open the second pickled file
        with open('UCF101v2-GT.pkl', 'rb') as pickled_two:
            try:
                # other times it is 'utf-8!!!
                test = pickle.load(pickled_two, encoding='latin1')

            except UnicodeDecodeError as e:
                print("UnicodeDecodeError:", e)
        return train, test

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
        mat = scipy.io.loadmat(f'{path}joint_positions/{action}/{video}/joint_positions.mat')
        return mat

    def rearranged_joints(self, action, video, path='/home/linxin67/scratch/JHMDB/'):
        '''
        Return a torch tensor with num frames, num joints, (x,y) joints.
        '''
        joint_dct = self.read_joints_full_video(action, video, path)
        torch_joint = torch.tensor(joint_dct['pos_img'])
        torch_joint = rearrange(torch_joint, 'd n f->f n d')
        return torch_joint
    
    # def read_joints(self, action, video, line_num, path='/home/linxin67/scratch/JHMDB/'):
    #     joint_file = self.read_joints(action, video)
    #     return mat[]

    # def read_all_joints(self, path='/hom/linxin67/scratch/JHMDB/'):
    #     os.chdir(path)

    #     return (action, video, )
    def get_num_frames(self, action, video):
        # os.chdir(path+'annotations')
        if action+'/'+video in self.nframes_train:
            return self.nframes_train[action+'/'+video]
        else:
            return self.nframes_test[action+'/'+video]
            
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

        # looping through gives you each action
        for action in os.listdir(directory):
            # print(action[-1], action[-5], action[-4])
            # I just want to look at the ones with 1 after.
            if action[-5] != '1':
                continue
            action_split = os.path.join(directory, action)
            actions.append(action[:-16])  # remove the _test_split<int>.txt
            # checking if it is a file
            if os.path.isfile(action_split):
                df = pd.read_csv(action_split, sep=' ', header=None)
                # looping and separating
                for index, row in df.iterrows():
                    file_name = row[0]
                    value = int(row[1])
                    if value == 1:
                        # remove the .avi
                        train.append((action[:-16], file_name[:-4], self.get_num_frames(action[:-16], file_name[:-4])))
                    elif value == 2:
                        test.append((action[:-16], file_name[:-4], self.get_num_frames(action[:-16], file_name[:-4])))
                    else:
                        print(type(value), value)
                        print("unknownIndexError")
        return actions, train, test

        # looping through each split of each action gives you the name of each video.

        # I'll load all of that into one file, which has action, video, train/test on each row

    # then we need a function to crop the images to 224x224, and need to generate batches of 8 frames in a row. (videos)
    # this will also need to transform the images into three channels (use torch.vision transforms.)

    def crop(self, action, video):
        pass
        # self.nframes_train[]
        # self.nframes_test = self.test_annotations['nframes']

        # crop images. (but first, need to determine if we are given bounding boxes, and if I need to pass patchify my input.)
        # finally, store the values into csv or wtv, and choose them randomly to be able to batch together the training

    def run_batch(self):
        pass


if __name__ == '__main__':
    data = load_JHMDB(joints=True, unpickle=True)
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

    example_joint = data.rearranged_joints(action='pour', video='Bartender_School_Students_Practice_pour_u_cm_np1_fr_med_1')
    print(example_joint)
    print(example_joint.shape)
    # print(len(example_joint['pos_img'][0][0]))
    # print(type(example_joint))
    # print(example_joint.keys())
    # print(data.get_num_frames(action='pour', video='Bartender_School_Students_Practice_pour_u_cm_np1_fr_med_1'))



    # then the rest is handled by pytorch:
    # Example usage
    # data = [...]  # Your dataset (list of samples)
    # custom_dataset = CustomDataset(data)
    # data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

    # # Iterate over batches of data
    # for batch in data_loader:
    #     # Process each batch of data
    #     pass