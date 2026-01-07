import torch

X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

Y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

dataset = (X, Y)