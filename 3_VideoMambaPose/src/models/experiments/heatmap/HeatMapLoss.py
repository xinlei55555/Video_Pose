import torch
import torch.nn as nn

class PoseEstimationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.config = config

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
        if self.config['show_predictions']: 
            print(f'The target values are : ', target)
            print(f'The predicted values are : ', predicted)
        return loss