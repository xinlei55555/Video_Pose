# run through it and go through every index that is 35.
import torch
from torch.utils.data import Dataset, DataLoader

from DataFormat import load_JHMDB

print('Started')

# The length of actions, train and test are 21 ,  660 ,  0, if its 1/8 of the data.
train_set = load_JHMDB(train_set=True, real_job=True, jump=8)

test_set = load_JHMDB(train_set=False, real_job=True, jump=8)

# train_set, test_set = train_set.to(device), test_set.to(device) # do not load the data here to the gpu

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# i'll take 1/8 of the dataset lol, although there is actually no need! it was able to load it perfectly
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers) 
    

