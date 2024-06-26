import torch

joint_for_frame = torch.tensor([[[0.0, 0.0],[0., 0.], [0., 0.]]])
# if not torch.all((joint_for_frame >= -1) & (joint_for_frame <= 1)):
#     print("Error, some of the normalized values are not between -1 and 1")

def compute_angle(joints):
    """
    Compute the angle between consecutive joints.
    Args:
        joints (torch.Tensor): The joint positions of shape (B, J, 2).
    Returns:
        torch.Tensor: The angles between consecutive joints.
    """
    vector = joints[:, 1:, :] - joints[:, :-1, :]
    
    # Calculate the angles using atan2
    angles = torch.atan2(vector[..., 1], vector[..., 0])
    
    # Set angle to 0 where the joints values are the same
    same_joints = (vector == 0).all(dim=-1)
    angles[same_joints] = 0
    
    return angles

print(compute_angle(joint_for_frame))
