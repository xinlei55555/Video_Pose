import torch
import torch.nn as nn
from einops import rearrange

predicted = torch.randn(16, 16, 15, 2)

# indeed, difference should be 0!! because target is just predicted with 3 everywhere
target = predicted + 3
loss = nn.MSELoss()
predicted_differences = predicted[..., 1:, :, :] - predicted[..., :-1, :, :]
print(predicted_differences.shape, predicted_differences)
# predicted = rearrange(predicted, 'b d j x -> (b d) j x')
# target = rearrange(target, 'b d j x -> (b d) j x')
predicted_differences1 = loss(
            predicted[..., 1:, :, :], predicted[..., :-1, :, :])
target_differences1 = loss(
            target[..., 1:, :, :], target[..., :-1, :, :])

predicted = rearrange(predicted, 'b d j x -> (b d) j x')
target = rearrange(target, 'b d j x -> (b d) j x')
predicted_differences = loss(
            predicted[..., 1:, :, :], predicted[..., :-1, :, :])
target_differences = loss(
            target[..., 1:, :, :], target[..., :-1, :, :])

print(predicted_differences1 == predicted_differences) # not the same!!!! on is computed between each frame of each batch (including between videos), the other is computed within each video

        # then we want the difference between both
print(abs(target_differences1 - predicted_differences1))