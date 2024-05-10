import torch
import torch.nn as nn

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

# Initialize the model and loss function
model = HeatMapVideoMambaPose().to(device)
loss_fn = PoseEstimationLoss()

# Example input tensor
input_tensor = torch.rand(batch_size, channels, num_frames, height, width).to(device)

# Forward pass
predicted_output = model(input_tensor)

# Example target tensor (should be of the same shape as predicted_output)
target_tensor = torch.rand_like(predicted_output).to(device)

# Compute loss
loss = loss_fn(predicted_output, target_tensor)
print(f"Loss: {loss.item()}")