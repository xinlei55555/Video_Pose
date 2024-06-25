import torch
import torch.nn as nn

from einops import rearrange

import math

class PoseEstimationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.mpjpe = self.loss_mpjpe
        self.velocity_loss = self.velocity_loss_fn
        self.angle_loss = self.angle_loss_fn

    def velocity_loss_fn(self, predicted, target):
        """
        Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
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
        predicted_differences = predicted[..., 1:, :, :] - predicted[..., :-1, :, :]
        target_differences = target[..., 1:, :, :] - target[..., :-1, :, :]

        # then we want the difference between both, THIS IS THE SAME AS MSE
        difference_norms = torch.norm(target_differences - predicted_differences, dim=-3)
        return torch.mean(difference_norms)

    def angle_loss_fn(self, predicted, target):
        """
        Args:
            predicted (torch.Tensor): The predicted joint positions or heatmaps.
            target (torch.Tensor): The ground truth joint positions or heatmaps.

        Returns:
            torch.Tensor: Computed loss.
        """
        def compute_angle(joints):
            """
            Compute the angle between consecutive joints.
            Args:
                joints (torch.Tensor): The joint positions of shape (B, J, 2).
            Returns:
                torch.Tensor: The angles between consecutive joints.
            """
            vector = joints[:, 1:, :] - joints[:, :-1, :]
            angles = torch.atan2(vector[..., 1], vector[..., 0])
            return angles

        predicted_angles = compute_angle(predicted)
        target_angles = compute_angle(target)

        # normalize the angle values
        predicted_angles /= 360.0
        target_angles /= 360.0

        return self.mse_loss(predicted_angles, target_angles)

    def loss_mpjpe(self, predicted, target):
        """
        Mean per-joint position error (i.e. mean Euclidean distance),
        often referred to as "Protocol #1" in many papers.
        """
        assert predicted.shape == target.shape
        # this is mean squared
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

    def forward(self, predicted, target, mask=None):
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
        


        # I need to mask all the values that are not visible in the dataset
        if mask is not None:
            # T, J, X = predicted.shape
            # for i in range(T):
            #     for idx in range(J):
            #         if mask[i][idx] == 0.:
            #             # ignore those values.
            #             predicted[i][idx][0] = predicted[i][idx][1] = target[i][idx][0] = target[i][idx][1] = 0.

            # Get the shape of the predicted tensor
            T, J, X = predicted.shape

            # Create a boolean mask where mask == 0
            zero_mask = (mask == 0)

            # Expand the mask to match the shape of the last dimension
            zero_mask = zero_mask.unsqueeze(-1).expand(-1, -1, X)

            # Instead of in-place modification, create new tensors with masked values set to 0
            predicted = predicted * (~zero_mask).float()
            target = target * (~zero_mask).float()

        calculated_loss = 0.0

        if 'mse' in self.config['losses']:
            calculated_loss += self.mse_loss(predicted, target) * self.config['losses']['mse']

        if 'velocity' in self.config['losses']:
            calculated_loss += self.velocity_loss(predicted, target) * self.config['losses']['velocity']

        if 'angle' in self.config['losses']:
            calculated_loss += self.angle_loss(predicted, target) * self.config['losses']['angle']

        if 'mpjpe' in self.config['losses']:
            calculated_loss += self.mpjpe(predicted, target) * self.config['losses']['mpjpe']

        if self.config['show_predictions']:
            print(f'The target values are : ', target)
            print(f'The predicted values are : ', predicted)

            print('Here is the calculated loss', calculated_loss)

        if math.isnan(calculated_loss) or math.isfinite(calculated_loss):
            print(f'The target values are : ', target)
            print(f'The predicted values are : ', predicted)
            # breaks.
            sys.exit(1)

        return calculated_loss
