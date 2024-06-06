import torch

# yes, denormalization works.
def normalize_fn(x, h=240.0, w=320.0):
    # x has num_frames, joint_numbers, (x, y)
    x[:, :, 0] = (x[:, :, 0] / (w / 2.0)) - 1.0  # bewteen -1 and 1
    x[:, :, 1] = (x[:, :, 1] / (h / 2.0)) - 1.0
    return x


def denormalize_fn(x, h=240.0, w=320.0):
    # actually, you should do denormalization AFTER the loss funciton. so when doing inference.
    x[:, :, 0] = (x[:, :, 0] + 1.0) * (w / 2.0)  # bewteen -1 and 1
    x[:, :, 1] = (x[:, :, 1] + 1.0) * (h / 2.0)
    return x
print(denormalize_fn(normalize_fn(torch.tensor([[[1.0, 2.0]]]))))