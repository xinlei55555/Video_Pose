import torch
import torch.nn as nn
import os

import pickle
import pandas as pd

from torch.utils.data import Dataset, DataLoader


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
            self.actions, self.train, self.test = self.get_names_train_test_split
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
        return 

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
    def read_joints(self, action, video, path="/home/linxin67/scratch/JHMDB/"):
        os.chdir(path)
        mat = scipy.io.loadmat(f'{path}{action}/{video}/joint_positions.mat')
        return mat
    
    def read_joints(self, action, video, line_num, path='/home/linxin67/scratch/JHMDB/'):
        joint_file = self.read_joints(action, video)
        return mat[]

    # def read_all_joints(self, path='/hom/linxin67/scratch/JHMDB/'):
    #     os.chdir(path)

    #     return (action, video, )

    def get_names_train_test_split(self, path="/home/linxin67/scratch/JHMDB/"):
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
                        train.append((action[:-16], file_name[:-4]))
                    elif value == 2:
                        test.append((action[:-16], file_name[:-4]))
                    else:
                        print(type(value), value)
                        print("unknownIndexError")
        return actions, train, test

        # looping through each split of each action gives you the name of each video.

        # I'll load all of that into one file, which has action, video, train/test on each row

    # then we need a function to crop the images to 224x224, and need to generate batches of 8 frames in a row. (videos)
    # this will also need to transform the images into three channels (use torch.vision transforms.)

    def crop(self, action, video, path='/home/linxin67/scratch/JHMDB/'):
        os.chdir(path)
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
    print(type(data.get_names_train_test_split()))
    print(len(data.get_names_train_test_split()[0]), len(
        data.get_names_train_test_split()[1]), len(data.get_names_train_test_split()[2]))
    print(data.get_names_train_test_split()[1])



    # then the rest is handled by pytorch:
    # Example usage
    # data = [...]  # Your dataset (list of samples)
    # custom_dataset = CustomDataset(data)
    # data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

    # # Iterate over batches of data
    # for batch in data_loader:
    #     # Process each batch of data
    #     pass