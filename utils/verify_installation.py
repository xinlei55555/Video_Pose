import torch
from causal_conv1d import causal_conv1d_fn
from mamba_ssm import Mamba

batch, dim, seq, width = 10, 5, 17, 4
x = torch.zeros((batch, dim, seq)).to('cuda')
weight = torch.zeros((dim, width)).to('cuda')
bias = torch.zeros((dim, )).to('cuda')

print(causal_conv1d_fn(x, weight, bias, None))

# -------------- cchecking if installation
# import torch


batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
print('everything worked', y, x, y.shape, x.shape)
