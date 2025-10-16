import torch
from torch import nn


class HH_layer(nn.Module):
	def __init__(
			self,
			in_features: int,
			out_features: int,
	):
		super().__init__()

		self.in_features = int(in_features)
		self.out_features = int(out_features)

		self.l1 = nn.Linear(in_features, out_features)  # spike -> current
		self.l2 = nn.Sequential(
			nn.Linear(out_features, out_features),
			nn.Softplus()
		)  # voltage -> spike

		# Neuron Capacitance (size of the neuron)
		self.c_m = nn.Parameter(torch.randn(out_features))

		# Gate Conductance
		self.g_na = nn.Parameter(torch.randn(out_features))
		self.g_k = nn.Parameter(torch.randn(out_features))
		self.g_l = nn.Parameter(torch.randn(out_features))

		# Gate Reversal Potentials (what the gates drag the voltage to)
		self.e_na = nn.Parameter(torch.randn(out_features))
		self.e_k = nn.Parameter(torch.randn(out_features))
		self.e_l = nn.Parameter(torch.randn(out_features))

		# v_half (at what voltage the gates are 50% open)
		self.v_half_m = nn.Parameter(torch.randn(out_features))
		self.v_half_h = nn.Parameter(torch.randn(out_features))
		self.v_half_n = nn.Parameter(torch.randn(out_features))

		# k (slope of sigmoid for p_inf)
		self.k_m = nn.Parameter(torch.randn(out_features))
		self.k_h = nn.Parameter(torch.randn(out_features))
		self.k_n = nn.Parameter(torch.randn(out_features))

		# tau_min (minimum time constant, fastest that the gate can respond)
		self.tau_min_m = nn.Parameter(torch.randn(out_features))
		self.tau_min_h = nn.Parameter(torch.randn(out_features))
		self.tau_min_n = nn.Parameter(torch.randn(out_features))

		# tau_amp (extra slowness added near the center)
		self.tau_amp_m = nn.Parameter(torch.randn(out_features))
		self.tau_amp_h = nn.Parameter(torch.randn(out_features))
		self.tau_amp_n = nn.Parameter(torch.randn(out_features))

		# tau_width (width of the extra slow region)
		self.tau_width_m = nn.Parameter(torch.randn(out_features))
		self.tau_width_h = nn.Parameter(torch.randn(out_features))
		self.tau_width_n = nn.Parameter(torch.randn(out_features))

		self.zero_states()

	@torch.no_grad()
	def zero_states(self, mode: str = "p_inf", v: float | torch.Tensor = 0.0):
		if isinstance(v, (int, float)):
			v_val = torch.full_like(self.v, float(v))
		else:
			v_val = v.to(device=self.v.device, dtype=self.v.dtype)
			if v_val.shape != self.v.shape:
				raise ValueError("v must be scalar or same shape as self.v")

		self.v.copy_(v_val)

		if mode == "zeros":
			self.m.zero_()
			self.h.zero_()
			self.n.zero_()
		elif mode == "p_inf":
			self.m.copy_(self.p_inf(self.v, self.v_half_m, self.k_m))
			self.h.copy_(self.p_inf(self.v, self.v_half_h, self.k_h))
			self.n.copy_(self.p_inf(self.v, self.v_half_n, self.k_n))
		elif mode == "random":
			self.m.copy_(torch.rand_like(self.m))
			self.h.copy_(torch.rand_like(self.h))
			self.n.copy_(torch.rand_like(self.n))
		else:
			raise ValueError("mode must be 'zeros', 'p_inf', or 'random'")

	def p_inf(self, v, v_half, k):
		k_safe = nn.functional.softplus(k) + 1e-9
		return torch.sigmoid((v - v_half) / k_safe)

	def tau_of(self, v, tau_min, tau_amp, tau_center, tau_width):
		# tau_center is v_half
		tau_min_p = nn.functional.softplus(tau_min) + 1e-9
		tau_amp_p = nn.functional.softplus(tau_amp)
		tau_width_p = nn.functional.softplus(tau_width) + 1e-9
		return tau_min_p + tau_amp_p * torch.exp(-((v - tau_center) ** 2) / tau_width_p)

	def forward(self, in_spikes, delta_t: float = 1e-3):
		m_inf = self.p_inf(self.v, self.v_half_m, self.k_m)
		tau_m = self.tau_of(self.v, self.tau_min_m, self.tau_amp_m, self.v_half_m, self.tau_width_m)

		h_inf = self.p_inf(self.v, self.v_half_h, self.k_h)
		tau_h = self.tau_of(self.v, self.tau_min_h, self.tau_amp_h, self.v_half_h, self.tau_width_h)

		n_inf = self.p_inf(self.v, self.v_half_n, self.k_n)
		tau_n = self.tau_of(self.v, self.tau_min_n, self.tau_amp_n, self.v_half_n, self.tau_width_n)

		self.m = self.m + ((m_inf - self.m) / (tau_m + 1e-9)) * delta_t
		self.h = self.h + ((h_inf - self.h) / (tau_h + 1e-9)) * delta_t
		self.n = self.n + ((n_inf - self.n) / (tau_n + 1e-9)) * delta_t

		self.m = self.m.clamp(0.0, 1.0)
		self.h = self.h.clamp(0.0, 1.0)
		self.n = self.n.clamp(0.0, 1.0)

		i_syn = self.l1(in_spikes)
		i_na = nn.functional.softplus(self.g_na) * (self.m ** 3) * self.h * (self.v - self.e_na)
		i_k = nn.functional.softplus(self.g_k) * (self.n ** 4) * (self.v - self.e_k)
		i_l = nn.functional.softplus(self.g_l) * (self.v - self.e_l)

		c_m = nn.functional.softplus(self.c_m) + 1e-9
		dv_dt = (i_syn - i_na - i_k - i_l) / c_m
		self.v = self.v + dv_dt * delta_t

		out_spikes = self.l2(self.v)
		return out_spikes
