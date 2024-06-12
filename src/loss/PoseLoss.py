import torch
import torch.nn as nn

class PoseEstimationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.config = config
        self.velocity_loss = VelocityLoss(config)

    def forward(self, predicted, target):
        """
        Args:
            predicted (torch.Tensor): The predicted joint positions or heatmaps.
            target (torch.Tensor): The ground truth joint positions or heatmaps.

        Returns:
            torch.Tensor: Computed loss.
        """
        loss = self.mse_loss(predicted, target)
        if self.config['show_predictions']: 
            print(f'The target values are : ', target)
            print(f'The predicted values are : ', predicted)
        return loss

class VelocityLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, predicted, target):
        """
        Args:
            predicted (torch.Tensor): The predicted joint positions or heatmaps.
            target (torch.Tensor): The ground truth joint positions or heatmaps.

        Returns:
            torch.Tensor: Computed loss.
        """
        pass

