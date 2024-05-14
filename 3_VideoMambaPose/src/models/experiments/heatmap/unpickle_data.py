import pickle
import os

def unpickle_JHMDB(path="/home/linxin67/scratch/JHMDB/annotations"):
    os.chdir(path)

    # Open the first pickled file
    # with open('JHMDB-GT.pkl', 'rb') as pickled_one:

    with open("JHMDB-GT.pkl",'rb') as pickled_one:
        try:
            # data=pickled_one.read()
            # data=data.decode('utf-8')
            # print(data[1:1000])

            data=pickle.load(pickled_one, encoding='latin1') #other times it is 'utf-8!!!
            print(data)
        except UnicodeDecodeError as e:
            print("UnicodeDecodeError:", e)

    # Open the second pickled file
    with open('UCF101v2-GT.pkl', 'rb') as pickled_two:
        try:
            data=pickle.load(pickled_two, encoding='latin1') #other times it is 'utf-8!!!
            print(type(data))
        except UnicodeDecodeError as e:
            print("UnicodeDecodeError:", e)
            



