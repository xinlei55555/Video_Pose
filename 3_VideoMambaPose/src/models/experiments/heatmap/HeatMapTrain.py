import torch
import torch.nn as nn

from HeatVideoMamba import  HeatMapVideoMambaPose

class PoseEstimationLoss(nn.Module):
    def __init__(self):
        super(PoseEstimationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted, target):
        """
        Args:
            predicted (torch.Tensor): The predicted joint positions or heatmaps.
            target (torch.Tensor): The ground truth joint positions or heatmaps.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        loss = self.mse_loss(predicted, target)
        return loss

# Example usage:
# Assuming `model` is an instance of `HeatMapVideoMambaPose`
# and `target` is the ground truth tensor.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the model and loss function
model = HeatMapVideoMambaPose().to(device)
loss_fn = PoseEstimationLoss()

batch_size = 16
num_frames = 8
height = 224
width = 224
channels = 3

# Generate a random input tensor
test_video = torch.rand(batch_size, channels, num_frames, height, width)

# Check the shape of the random tensor
print("Shape of the random tensor:", test_video.shape)

# defining model
test_model = HeatMapVideoMambaPose()

# move the data to the GPU
test_model = test_model.to(device)
test_video = test_video.to(device)

# Forward Pass
y = test_model(test_video)

# * note: (B, C, T, H, W) returns 16, 192, 8, 14, 14
# torch.Size([16, 1568, 192]), i.e. (Batch, 1568 is 8*14*14, 192 is the channel number )
print(y.shape)
print(y)

# Example target tensor (should be of the same shape as predicted_output)
target_tensor = None #TODO define this later

# Compute loss
loss = loss_fn(predicted_output, target_tensor)
print(f"Loss: {loss.item()}")