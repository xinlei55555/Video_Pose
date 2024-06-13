import torch
import torch.nn as nn

from einops import rearrange

class PoseEstimationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.velocity_loss = self.velocity_loss_fn

    def velocity_loss_fn(self, predicted, target):
        """
        Args:
            predicted (torch.Tensor): The predicted joint positions or heatmaps.
            target (torch.Tensor): The ground truth joint positions or heatmaps.

        Returns:
            torch.Tensor: Computed loss.
                """
        # n_frames = predicted.shape[-3]
        if not self.config['use_last_frame_only']:
            predicted = rearrange(predicted, '(b d) j x -> b d j x',
                                d=self.config['num_frames'])
            target = rearrange(target, '(b d) j x -> b d j x',
                            d=self.config['num_frames'])

        # inputted shape should be (B, Num_frames, Joint_number, 2)
        # MSE is the squared difference of all the values between each two frames
        # this gives me the difference between each two frames within a video
        predicted_differences = self.mse_loss(
            predicted[..., 1:, :, :], predicted[..., :-1, :, :])
        target_differences = self.mse_loss(
            target[..., 1:, :, :], target[..., :-1, :, :])

        # then we want the difference between both
        return abs(target_differences - predicted_differences)

    def forward(self, predicted, target):
        """
        Args:
            predicted (torch.Tensor): The predicted joint positions or heatmaps.
            target (torch.Tensor): The ground truth joint positions or heatmaps.

        Returns:
            torch.Tensor: Computed loss.
        """
        # since now I am taking in predicted and targets for each frame in a video
        # inputted shape should be (B, Num_frames, Joint_number, 2)
        if not self.config['use_last_frame_only']:
            predicted = rearrange(predicted, 'b d j x -> (b d) j x')
            target = rearrange(target, 'b d j x -> (b d) j x')

        calculated_loss = 0.0

        if 'mse' in self.config['losses']:
            calculated_loss += self.mse_loss(predicted, target)

        if 'velocity' in self.config['losses']:
            calculated_loss += self.velocity_loss(predicted, target)

        if self.config['show_predictions']:
            print(f'The target values are : ', target)
            print(f'The predicted values are : ', predicted)

        return calculated_loss
