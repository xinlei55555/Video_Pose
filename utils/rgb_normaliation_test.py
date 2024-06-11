def rgb_normalization(self, input_tensor):
        '''Takes a torch tensor, and normalizes the RGB channels to have values between 0 and 1.
        The mean values established in this are simply the usual imagenet values
        For images, and videos, directly applies the normalization.'''        
        # Define mean and std tensors
        # Example video tensor of shape (B, frames_num, C, H, W)
        mean = torch.tensor([0.485, 0.456, 0.406],
                            dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                            dtype=torch.float32).view(1, 3, 1, 1)
        
        rgb = torch.tensor([256.0, 256.0, 256.0], dtype=torch.float32.view(1, 3, 1, 1))

        # Apply normalization using broadcasting
        output_tensor = (input_tensor / rgb - mean) / std

        return output_tensor

if __name__=='__main__':
    x = torch.rand()