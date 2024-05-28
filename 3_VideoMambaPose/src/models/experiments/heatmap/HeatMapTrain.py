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
        "learning_rate": 0.001,
        "architecture": "12 Video BiMamba blocks + 3 layers 2D Deconvolutions + 1 layers Convolution + Joint Regressor (Linear + Relu + Linear)",
        "dataset": "JHMDB, no cropping.",
        "epochs": 300,
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


def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint)  # this depends on how I saved the model
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    return model


def training_loop(n_epochs, optimizer, model, loss_fn, train_set, test_set, device, follow_up=(False, 1, None)):
    checkpoints_dir = 'checkpoints'
    os.chdir("/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap")
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_val_loss = float('inf')

    start_epoch = 1
    # then load checkpoint to follow up on the model
    if follow_up[0]:
        checkpoint_path = follow_up[2]
        model = load_checkpoint(checkpoint_path, model)
        start_epoch = follow_up[1]

    for epoch in range(start_epoch, n_epochs + start_epoch):
        model.train()  # so that the model keeps updating its weights.
        train_loss = 0.0
        # print('train batch for epoch # ', epoch)

        if epoch == start_epoch:
            # Prints GPU memory summary
            print('Memory before (in MB)', torch.cuda.memory_allocated()/1e6)

        if epoch == 1:
            print(f'The length of the train_set is {len(train_set)}')
            print(f'The length of the test_set is {len(test_set)}')

        for i, data in enumerate(train_set):

            # update device based on GPU usage.
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'

            train_inputs, train_labels = data

            # should load individual batches to GPU
            train_inputs, train_labels = train_inputs.to(
                device), train_labels.to(device)

            # first make an initial guess as to the weights (Note: training is done in parallel)
            train_outputs = model(train_inputs)

            if epoch == start_epoch:
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

            # .item transforms loss from pytorch tensor to python value
            train_loss += loss_train.item()

            if epoch == start_epoch:
                # Prints GPU memory summary
                print('Memory after train_batch (in MB)',
                      torch.cuda.memory_allocated()/1e6)

            torch.cuda.empty_cache()  # Clear cache to save memory

        model.eval()  # so that the model does not change the values of the parameters
        test_loss = 0.0
        with torch.no_grad():  # reduce memory while torch is using evaluation mode
            # print('test batch for epoch # ', epoch)
            for i, data in enumerate(test_set):
                # update device based on GPU usage, I think this is important to avoid parallelization error
                # device = 'cuda' if torch.cuda.is_available() else 'cpu'

                test_inputs, test_labels = data
                test_inputs, test_labels = test_inputs.to(
                    device), test_labels.to(device)

                # repeat for the validation
                test_outputs = model(test_inputs)

                # get the loss again for the validation
                loss_val = loss_fn(test_outputs.float(), test_labels.float())

                test_loss += loss_val.item()

                if epoch == start_epoch:
                    # Prints GPU memory summary
                    print('Memory after test_batch (in MB)',
                          torch.cuda.memory_allocated()/1e6)

                torch.cuda.empty_cache()  # Clear cache to save memory

        # the shown loss should be for individual elements in the batch size
        show_loss_train, show_loss_test = train_loss / \
            len(train_set), test_loss/len(test_set)

        wandb.log({"Pointwise training loss": show_loss_train})
        wandb.log({"Pointwise testing loss": show_loss_train})

        print(f"Epoch {epoch}, Pointwise Training loss {float(show_loss_train)},"
              f" Pointwise Validation loss {float(show_loss_test)}")
        print(
            f"Full training loss: {float(train_loss)}, Full test loss: {float(test_loss)}")

        # I use the full loss when comparing, to avoid having too small numbers.
        if test_loss < best_val_loss:
            best_val_loss = test_loss

            # Save the model checkpoint, since this is classification, there isn't really an accuracy...
            # ! delete the previous ones, because takes lots of space
            # os.rmdir(checkpoints_dir)
            # os.makedirs(checkpoints_dir, exist_ok=True)

            # save model locally
            checkpoint_path = os.path.join(
                checkpoints_dir, f'heatmap_{best_val_loss:.4f}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            # torch.save({
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'epoch': epoch,
            #     'loss': loss.item(),
            # }, checkpoint_path)
            print(f'Best model saved at {checkpoint_path}')
            print("Model parameters are of the following size",
                  len(list(model.parameters())))
    wandb.finish()


def main():
    # currently, only taking the first GPU, but later will use DDP will need to change.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the model and loss function
    model = HeatMapVideoMambaPose().to(device)

    # making sure to employ parallelization!!!
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # move the data to the GPU
    model = model.to(device)

    print(model)

    loss_fn = PoseEstimationLoss()

    batch_size = 16
    num_workers = 0
    # num_frames = x64x # i'll actually be using 16
    # height = 224
    # width = 224
    # channels = 3

    # ! loading the data, will need to set real_job to False when training
    train_set = load_JHMDB(train_set=True, real_job=True, jump=1)
    test_set = load_JHMDB(train_set=False, real_job=True, jump=1)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    training_loop(300, optimizer, model, loss_fn,
                  train_loader, test_loader, device,
                  (True, 50, '/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap/checkpoints/heatmap_22069.0820.pt'))


if __name__ == '__main__':
    main()
