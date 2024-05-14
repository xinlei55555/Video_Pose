import pickle
import os

os.chdir("/home/linxin67/scratch/JHMDB/annotations")

# Open the first pickled file
# with open('JHMDB-GT.pkl', 'rb') as pickled_one:
with open("JHMDB-GT.pkl",'rb') as pickled_one:
    try:
        data=pickled_one.read()
        data=data.decode('utf-8')
        print(data)
    except UnicodeDecodeError as e:
        print("UnicodeDecodeError:", e)

# Open the second pickled file
with open('UCF101v2-GT.pkl', 'rb') as pickled_two:
    try:
        data=pickled_two.read()
        data=data.decode('utf-8')
        print(type(data))
    except UnicodeDecodeError as e:
        print("UnicodeDecodeError:", e)



