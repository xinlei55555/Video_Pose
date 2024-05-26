import torch
import torch.nn as nn

from HeatVideoMamba import HeatMapVideoMambaPose
from torch.utils.data import Dataset, DataLoader

from DataFormat import load_JHMDB

import os

# wandb stuff
import wandb
wandb.init(
    project="1heatmap_video_mamba",

    config={
        "learning_rate":0.001,
        "architecture":"12 Video BiMamba blocks + 3 layers 2D Deconvolutions + 1 layers Convolution + Joint Regressor (Linear + Relu + Linear)",
        "dataset":"JHMDB, no cropping.",
        "epochs":200,
    }
)

class PoseEstimationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted, target):
        """
        Args:
            predicted (torch.Tensor): The predicted joint positions or heatmaps.
            target (torch.Tensor): The ground truth joint positions or heatmaps.

        Returns:
            torch.Tensor: Computed loss.
        """
        # !TODO I need to change that, because it's not just MSE, I am taking the mse of a 3D value, idk if mse works.
        loss = self.mse_loss(predicted, target)
        return loss


def training_loop(n_epochs, optimizer, model, loss_fn, train_set, test_set, device):
    checkpoints_dir = 'checkpoints'
    os.chdir("/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap")
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        model.train() # so that the model keeps updating its weights.
        train_loss = 0.0
        print('train batch for epoch # ', epoch)
        for i, data in enumerate(train_set):

            # update device based on GPU usage.
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'

            train_inputs, train_labels = data

            # should load individual batches to GPU
            train_inputs, train_labels = train_inputs.to(device), train_labels.to(device) 

            # first make an initial guess as to the weights (Note: training is done in parallel)
            train_outputs = model(train_inputs)
            
            if epoch == 1:
                print('The shape of the outputs is ', train_outputs.shape)
                print('The shape of the labels are ', train_labels.shape)

            # determine the loss using the loss_fn which is passed into the training loop. 
            # Note: Need to pass float! (not double)
            loss_train = loss_fn(train_outputs.float(), train_labels.float())


            # optimizer changes the weight and biases to zero, before starting the training again.
            optimizer.zero_grad()

            # this is what computes the derivative of the loss
            loss_train.backward()  # !this will accumulate the gradients at the leaf nodes

            # then, the optimizer will update the values of the weights based on all the derivatives of the losses computed by loss_train.backward()
            optimizer.step()

            train_loss += loss_train

            torch.cuda.empty_cache()  # Clear cache to save memory

        wandb.log({"training loss": train_loss})
        
        model.eval() # so that the model does not change the values of the parameters
        test_loss = 0.0
        with torch.no_grad(): # reduce memory while torch is using evaluation mode
            print('test batch for epoch # ', epoch)
            for i, data in enumerate(test_set):
                # update device based on GPU usage, I think this is important to avoid parallelization error
                # device = 'cuda' if torch.cuda.is_available() else 'cpu'

                test_inputs, test_labels = data
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device) 

                # repeat for the validation
                test_outputs = model(test_inputs)

                # get the loss again for the validation
                loss_val = loss_fn(test_outputs.float(), test_labels.float())

                test_loss += loss_val

                torch.cuda.empty_cache()  # Clear cache to save memory

            wandb.log({"testing loss": test_loss})


        if epoch == 1 or epoch % 50 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Validation loss {test_loss.item():.4f}")

        if test_loss < best_val_loss:
            best_val_loss = loss_val

            # Save the model checkpoint, since this is classification, there isn't really an accuracy...
            # ! delete the previous ones, because takes lots of space
            # os.rmdir(checkpoints_dir)
            # os.makedirs(checkpoints_dir, exist_ok=True)

            # save model locally
            checkpoint_path = os.path.join(checkpoints_dir, f'heatmap_{best_val_loss:.4f}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Best model saved at {checkpoint_path}')
            print("Model parameters are of the following size", len(list(model.parameters())))
    wandb.finish()


# Example usage:
# Assuming `model` is an instance of `HeatMapVideoMambaPose`
# and `target` is the ground truth tensor.
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Initialize the model and loss function
model = HeatMapVideoMambaPose().to(device)
print(model)
loss_fn = PoseEstimationLoss()

# on z
batch_size = 16 # I'll maybe reduce the batch size to 12, just to be safe lol

num_workers = 4 # ! keep it low for testing purposes, but for training, increase to 4
# num_frames = x64x # i'll actually be using 16
# height = 224
# width = 224
# channels = 3

# * testing purposes:
# Generate a random input tensor
# test_video = torch.rand(batch_size, channels, num_frames, height, width)

# # Check the shape of the random tensor
# print("Shape of the random tensor:", test_video.shape)
# -----------------

# ! loading the data, will need to set real_job to False when training
train_set = load_JHMDB(train_set=True, real_job=True)
test_set = load_JHMDB(train_set=False, real_job=True)

# train_set, test_set = train_set.to(device), test_set.to(device) # do not load the data here to the gpu

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# half the batch size for the test loader.
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# defining model
model = HeatMapVideoMambaPose()

# making sure to employ parallelization!!! ANd it workkkeddd
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# move the data to the GPU
model = model.to(device)

# Forward Pass
# y = model(test_video)

# * note: (B, C, T, H, W) returns 16, 192, 8, 14, 14
# torch.Size([16, 1568, 192]), i.e. (Batch, 1568 is 8*14*14, 192 is the channel number )
# print(y.shape)
# print(y)

# Example target tensor (should be of the same shape as predicted_output)
# target_tensor = None  # TODO define this later

# showing the parameters:
# list(model.parameters())

# Compute loss
# loss = loss_fn(predicted_output, target_tensor)
# print(f"Loss: {loss.item()}")

# optimizer
optimizer = torch.optim.Adam(model.parameters())

# Training loop
loss_fn = PoseEstimationLoss()

# ! will increase the number of epochs when not training
training_loop(1, optimizer, model, loss_fn, train_loader, test_loader, device)
