import torch
from torch import nn

n_in = 10
n_out = 10

rand_input = torch.randn(n_in)
rand_output = torch.randn(n_out)

model = nn.Sequential(
	nn.Linear(n_in, 32),
	nn.LeakyReLU(),
	nn.Linear(32, n_out)
)

out = model(rand_input)
# loss = nn.functional.mse_loss(out, rand_output)
loss = torch.sum(out - out)
loss.backward()

for parameter in model.parameters():
	print(parameter.grad)
