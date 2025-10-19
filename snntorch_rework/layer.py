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

	def forward(self, x, syn, mem):
		alpha = self.param_to_dim(nn.functional.sigmoid(self.alpha), x.dim())
		beta = self.param_to_dim(nn.functional.sigmoid(self.beta), x.dim())
		threshold = self.param_to_dim(nn.functional.softplus(self.threshold), x.dim())

		syn1 = syn * alpha + x
		mem1 = mem * beta + syn1
		x1 = self.surrogate_function(mem1 - threshold)
		mem1 = mem1 - x1 * threshold

		return x1, syn1, mem1


# ======================================================================================================================


import torch, torch.nn as nn

# choose atan surrogate if desired
lif = Synaptic(16, alpha=0.9, beta=0.85, threshold=1.0,
			   surrogate_function=atan_surrogate(alpha=2.0))

# toy linear upstream
lin = nn.Linear(10, 16)
x = torch.randn(2, 10)  # batch of 2
pre = lin(x)  # [B, F]
syn = torch.zeros_like(pre)
mem = torch.zeros_like(pre)

spk, syn, mem = lif(pre, syn, mem)
loss = spk.sum()
loss.backward()

print("lin.weight.grad:", lin.weight.grad is not None)
print("lif.raw_alpha.grad:", lif.alpha.grad is not None)
