import torch
import torch.nn as nn

from HeatVideoMamba import HeatMapVideoMambaPose

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


def training_loop(n_epochs, optimizer, model, loss_fn, train_inputs, val_inputs, train_labels, val_labels):
    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        # first make an initial guess as to the weights (Note: training is done in parallel)
        train_outputs = model(train_inputs)

        # determine the loss using the loss_fn which is passed into the training loop
        loss_train = loss_fn(train_outputs, train_labels)

        # repeat for the validation
        val_outputs = model(val_inputs)

        # get the loss again for the validation
        loss_val = loss_fn(val_outputs, val_labels)

        # optimizer changes the weight and biases to zero, before starting the training again.
        optimizer.zero_grad()

        # this is what computes the derivative of the loss
        loss_train.backward()  # !this will accumulate the gradients at the leaf nodes

        # then, the optimizer will update the values of the weights based on all the derivatives of the losses computed by loss_train.backward()
        optimizer.step()

        wandb.log({"loss": loss})

        if epoch == 1 or epoch % 50 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                  f" Validation loss {loss_val.item():.4f}")

        if loss_val < best_val_loss:
            best_val_loss = loss_val

            # Save the model checkpoint, since this is classification, there isn't really an accuracy...
            # ! delete the previous ones, because takes lots of space
            os.rmdir(checkpoints_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

            # save model locally
            checkpoint_path = os.path.join(checkpoint_dir, f'heatmap_{best_val_loss:.4f}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Best model saved at {checkpoint_path}')
    wandb.finish()


# Example usage:
# Assuming `model` is an instance of `HeatMapVideoMambaPose`
# and `target` is the ground truth tensor.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the model and loss function
model = HeatMapVideoMambaPose().to(device)
print(model)
loss_fn = PoseEstimationLoss()

batch_size = 16
num_frames = 64
height = 224
width = 224
channels = 3

# Generate a random input tensor
test_video = torch.rand(batch_size, channels, num_frames, height, width)

# Check the shape of the random tensor
print("Shape of the random tensor:", test_video.shape)

# defining model
model = HeatMapVideoMambaPose()

# move the data to the GPU
model = model.to(device)
test_video = test_video.to(device)

# Forward Pass
y = model(test_video)

# * note: (B, C, T, H, W) returns 16, 192, 8, 14, 14
# torch.Size([16, 1568, 192]), i.e. (Batch, 1568 is 8*14*14, 192 is the channel number )
print(y.shape)
print(y)

# Example target tensor (should be of the same shape as predicted_output)
target_tensor = None  # TODO define this later

# showing the parameters:
list(model.parameters())

# Compute loss
loss = loss_fn(predicted_output, target_tensor)
print(f"Loss: {loss.item()}")

# optimizer
torch.optim.Adam(model.parameters())

# Training loop
loss_fn = PoseEstimationLoss()
training_loop(200, optimizer, model, loss_fn, train_inputs,
              val_inputs, train_labels, val_labels)
