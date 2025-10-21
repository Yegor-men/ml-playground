import torch
from torch import nn


# ======================================================================================================================
# ======================================================================================================================
class SigmoidSurrogate(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input_, slope):
		ctx.save_for_backward(input_)
		ctx.slope = float(slope)

		out = (input_ > 0).float()
		return out

	@staticmethod
	def backward(ctx, grad_output):
		(input_,) = ctx.saved_tensors
		k = ctx.slope

		s = torch.sigmoid(k * input_)
		grad_input = grad_output * (k * s * (1.0 - s))

		return grad_input, None


def sigmoid_surrogate(slope: float = 25):
	def inner(x: torch.Tensor):
		return SigmoidSurrogate.apply(x, slope)

	return inner


# ======================================================================================================================


class ATanSurrogate(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input_, alpha):
		ctx.save_for_backward(input_)
		ctx.alpha = float(alpha)

		out = (input_ > 0).float()
		return out

	@staticmethod
	def backward(ctx, grad_output):
		(input_,) = ctx.saved_tensors
		alpha = ctx.alpha
		# grad = (alpha / 2) / (1 + (pi/2 * alpha * u)^2) * grad_output
		denom = 1.0 + (torch.pi / 2.0 * alpha * input_).pow(2)
		grad = grad_output * (alpha / 2.0) / denom
		return grad, None


def atan_surrogate(alpha: float = 2.0):
	def inner(x: torch.Tensor):
		return ATanSurrogate.apply(x, alpha)

	return inner


# ======================================================================================================================
# ======================================================================================================================

class Synaptic(nn.Module):
	def __init__(
			self,
			num_neurons: int,
			alpha: float = 0.5,
			beta: float = 0.5,
			threshold: float = 1.0,
			surrogate_function=atan_surrogate(),
	):
		super().__init__()
		self.out_features = int(num_neurons)
		self.surrogate_function = surrogate_function

		def sigmoid_inverse(tensor: torch.Tensor) -> torch.Tensor:
			# return torch.log(tensor / (1 - tensor))
			return torch.logit(tensor)

		def softplus_inverse(tensor: torch.Tensor) -> torch.Tensor:
			return torch.log(torch.expm1(tensor))

		with torch.no_grad():
			alpha = sigmoid_inverse(torch.ones(num_neurons) * alpha)
			beta = sigmoid_inverse(torch.ones(num_neurons) * beta)
			threshold = softplus_inverse(torch.ones(num_neurons) * threshold)

		self.alpha = nn.Parameter(alpha)
		self.beta = nn.Parameter(beta)
		self.threshold = nn.Parameter(threshold)

		self.zero_states()

	def zero_states(self):
		self.syn = None
		self.mem = None

	def detach_states(self):
		self.syn = self.syn.detach()
		self.mem = self.mem.detach()

	def param_to_dim(self, parameter, dim: int):
		if dim == 1:
			return parameter
		if dim == 2:
			return parameter.view(1, -1)
		if dim == 3:
			return parameter.view(-1, 1, 1)
		if dim == 4:
			return parameter.view(1, -1, 1, 1)

		raise ValueError(f"Dimension {dim} not in (1, 2, 3, 4) for MLP or CNN batched or unbatched")

	def forward(self, x):
		if self.syn is None:
			self.syn = torch.zeros_like(x)
		if self.mem is None:
			self.mem = torch.zeros_like(x)

		alpha = self.param_to_dim(nn.functional.sigmoid(self.alpha), x.dim())
		beta = self.param_to_dim(nn.functional.sigmoid(self.beta), x.dim())
		threshold = self.param_to_dim(nn.functional.softplus(self.threshold), x.dim())

		self.syn = self.syn * alpha + x
		self.mem = self.mem * beta + self.syn
		out_spikes = self.surrogate_function(self.mem - threshold)
		self.mem = self.mem - out_spikes * threshold

		return out_spikes


# ======================================================================================================================
from tqdm import tqdm


class SNN(nn.Module):
	def __init__(self, n_in, n_hidden, n_out):
		super().__init__()

		self.lin1 = nn.Linear(n_in, n_hidden)
		self.lif1 = Synaptic(n_hidden, 0.5, 0.5, 1.0, atan_surrogate(2.0))

		self.lin2 = nn.Linear(n_hidden, n_hidden)
		self.lif2 = Synaptic(n_hidden, 0.5, 0.5, 1.0, atan_surrogate(2.0))

		self.lin3 = nn.Linear(n_hidden, n_out)
		self.lif3 = Synaptic(n_out, 0.5, 0.5, 1.0, atan_surrogate(2.0))

	def zero_states(self):
		self.lif1.zero_states()
		self.lif2.zero_states()
		self.lif3.zero_states()

	def detach_states(self):
		self.lif1.detach_states()
		self.lif2.detach_states()
		self.lif3.detach_states()

	def forward(self, x):
		x = self.lin1(x)
		x = self.lif1(x)

		x = self.lin2(x)
		x = self.lif2(x)

		x = self.lin3(x)
		x = self.lif3(x)

		return x


n_in = 10
n_hidden = 32
n_out = 10

rand_in = torch.rand(n_in)
rand_out = torch.rand(n_out).round_()

model = SNN(n_in, n_hidden, n_out)

optimizer = torch.optim.AdamW(model.parameters())

num_timesteps = 100
num_epochs = 100

losses = []

for e in tqdm(range(num_epochs), total=num_epochs):
	model.zero_states()
	model.zero_grad()

	average_output = torch.zeros(n_out)

	for t in range(num_timesteps):
		spike_input = torch.bernoulli(rand_in)
		model_out = model(spike_input)
		average_output = average_output + model_out

	average_output = average_output / num_timesteps
	loss = nn.functional.mse_loss(average_output, rand_out)
	loss.backward()
	losses.append(loss.item())
	optimizer.step()

import matplotlib.pyplot as plt

plt.plot(losses, label="Loss")
plt.legend()
plt.title("Loss")
plt.show()

with torch.no_grad():
	for t in tqdm(range(num_timesteps), total=num_timesteps):
		spike_input = torch.bernoulli(rand_in)
		model_out = model(spike_input)

		loss = nn.functional.mse_loss(model_out, rand_out)

		print(f"S: {model_out}, L: {loss.item()}")
