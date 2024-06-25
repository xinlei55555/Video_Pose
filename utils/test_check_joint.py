import torch

joint_for_frame = torch.tensor([[1.2, 1.5],[0.5, 2.4], [1.2, 0.5]])
if not torch.all((joint_for_frame >= -1) & (joint_for_frame <= 1)):
    print("Error, some of the normalized values are not between -1 and 1")
