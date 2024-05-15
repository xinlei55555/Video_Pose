from unpickle_data import unpickle_JHMDB

import torch
import torch.nn as nn
import os

import pickle

class load_JHMDB(nn.Module):
    def __init__(self, unpickle=False, path="/home/linxin67/scratch/JHMDB/annotations"):
        os.chdir(path)

        if unpickle:
            self.train_annotations, self.test_annotations = self.unpickle_JHMDB()
        
        if joints:
            self.joints = 

    def unpickle_JHMDB(self, path="/home/linxin67/scratch/JHMDB_old/annotations"):
        os.chdir(path)

        # Open the first pickled file
        # with open('JHMDB-GT.pkl', 'rb') as pickled_one:

        with open("JHMDB-GT.pkl",'rb') as pickled_one:
            try:
                train=pickle.load(pickled_one, encoding='latin1') #other times it is 'utf-8!!!
            except UnicodeDecodeError as e:
                print("UnicodeDecodeError:", e)

        # Open the second pickled file
        with open('UCF101v2-GT.pkl', 'rb') as pickled_two:
            try:
                test=pickle.load(pickled_two, encoding='latin1') #other times it is 'utf-8!!!
                
            except UnicodeDecodeError as e:
                print("UnicodeDecodeError:", e)
        return train, test
            
    def run_batch(self):
        pass
    


if __name__=='__main__':
    data = load_JHMDB()
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

    print(data.train_annotations['gttubes']['pour/Bartender_School_Students_Practice_pour_u_cm_np1_fr_med_1'][8]) # and I think each video is annotated such that the first index is the joint value, and the rest of the array are the positions

