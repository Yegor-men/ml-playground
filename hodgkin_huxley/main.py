import torch
from torch import nn


class HH_layer(nn.Module):
	def __init__(
			self,
			in_features: int,
			out_features: int,
			delta_t: float = 1e-3,
	):
		super().__init__()

		self.in_features = int(in_features)
		self.out_features = int(out_features)
		self.delta_t = delta_t

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

		# self.register_buffer("v", torch.zeros(self.out_features))
		# self.register_buffer("m", self.p_inf(self.v, self.v_half_m, self.k_m).detach())
		# self.register_buffer("h", self.p_inf(self.v, self.v_half_h, self.k_h).detach())
		# self.register_buffer("n", self.p_inf(self.v, self.v_half_n, self.k_n).detach())

		self.v, self.m, self.h, self.n = None, None, None, None
		self.zero_states()

	@torch.no_grad()
	def zero_states(self, mode: str = "p_inf", v: float | torch.Tensor = 0.0):
		self.v = torch.zeros(self.out_features).to(self.l1.weight)
		self.m = self.p_inf(self.v, self.v_half_m, self.k_m).detach()
		self.h = self.p_inf(self.v, self.v_half_h, self.k_h).detach()
		self.n = self.p_inf(self.v, self.v_half_n, self.k_n).detach()

	def p_inf(self, v, v_half, k):
		k_safe = nn.functional.softplus(k) + 1e-9
		return torch.sigmoid((v - v_half) / k_safe)

	def tau_of(self, v, tau_min, tau_amp, tau_center, tau_width):
		# tau_center is v_half
		tau_min_p = nn.functional.softplus(tau_min) + 1e-9
		tau_amp_p = nn.functional.softplus(tau_amp)
		tau_width_p = nn.functional.softplus(tau_width) + 1e-9
		return tau_min_p + tau_amp_p * torch.exp(-((v - tau_center) ** 2) / tau_width_p)

	def forward(self, in_spikes):
		self.v = self.v.to(in_spikes)
		self.m = self.m.to(in_spikes)
		self.h = self.h.to(in_spikes)
		self.n = self.n.to(in_spikes)

		m_inf = self.p_inf(self.v, self.v_half_m, self.k_m)
		tau_m = self.tau_of(self.v, self.tau_min_m, self.tau_amp_m, self.v_half_m, self.tau_width_m)

		h_inf = self.p_inf(self.v, self.v_half_h, self.k_h)
		tau_h = self.tau_of(self.v, self.tau_min_h, self.tau_amp_h, self.v_half_h, self.tau_width_h)

		n_inf = self.p_inf(self.v, self.v_half_n, self.k_n)
		tau_n = self.tau_of(self.v, self.tau_min_n, self.tau_amp_n, self.v_half_n, self.tau_width_n)

		self.m = self.m + (((m_inf - self.m) / (tau_m + 1e-9)) * delta_t).clamp(0.0, 1.0)
		self.h = self.h + (((h_inf - self.h) / (tau_h + 1e-9)) * delta_t).clamp(0.0, 1.0)
		self.n = self.n + (((n_inf - self.n) / (tau_n + 1e-9)) * delta_t).clamp(0.0, 1.0)

		i_syn = self.l1(in_spikes)
		i_na = nn.functional.softplus(self.g_na) * (self.m ** 3) * self.h * (self.v - self.e_na)
		i_k = nn.functional.softplus(self.g_k) * (self.n ** 4) * (self.v - self.e_k)
		i_l = nn.functional.softplus(self.g_l) * (self.v - self.e_l)

		dv_dt = (i_syn - i_na - i_k - i_l) / (nn.functional.softplus(self.c_m) + 1e-9)
		self.v = self.v + dv_dt * self.delta_t

		out_spikes = self.l2(self.v)

		return out_spikes


# ======================================================================================================================
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# --- Constants ---
NUM_CLASSES = 10
IMAGE_SIZE = 28 * 28  # Flattened MNIST image

# --- Transform ---
transform = transforms.Compose([
	transforms.ToTensor(),  # [1, 28, 28]
	transforms.Lambda(lambda x: x.view(-1))  # Flatten to [784]
])


# --- One-hot encoding helper ---
def one_hot_encode(labels, num_classes=NUM_CLASSES):
	return F.one_hot(labels, num_classes=num_classes).float()


# --- Custom collate function (no batching) ---
def collate_fn(batch):
	# Each batch is actually one sample when batch_size=1
	(image, label) = batch[0]
	label = one_hot_encode(torch.tensor(label), NUM_CLASSES)  # [10]
	return image, label


# --- Dataloaders (always single sample) ---
def get_mnist_single_dataloaders():
	train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
	test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

	train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

	return train_loader, test_loader


train_dataloader, test_dataloader = get_mnist_single_dataloaders()
# ======================================================================================================================
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

middle_size = 64
num_timesteps = 100
num_epochs = 1
delta_t = 1e-3

model = nn.Sequential(
	HH_layer(in_features=784, out_features=middle_size),
	HH_layer(in_features=middle_size, out_features=middle_size),
	HH_layer(in_features=middle_size, out_features=10),
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

train_losses, test_losses = [], []
garbate_losses = []

for e in range(num_epochs):
	# Train
	model.train()
	train_loss = 0
	for (image, label) in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Train - E{e + 1}"):
		image, label = image.to(device), label.to(device)
		temp_loss = 0
		optimizer.zero_grad()
		for layer in model:
			layer.zero_states()

		for t in range(num_timesteps):
			model_out = model(image)
			loss = nn.functional.mse_loss(model_out, label)
			temp_loss += loss.item()
			loss.backward()
			for layer in model:
				layer.v = layer.v.detach()
				layer.m = layer.m.detach()
				layer.h = layer.h.detach()
				layer.n = layer.n.detach()

		with torch.no_grad():
			for param in model.parameters():
				param.grad.div_(num_timesteps)

		optimizer.step()
		temp_loss /= num_timesteps
		train_loss += temp_loss
		garbate_losses.append(temp_loss)

	train_loss /= len(train_dataloader)
	train_losses.append(train_loss)

	# Test
	model.eval()
	test_loss = 0
	for (image, label) in tqdm(test_dataloader, total=len(test_dataloader), desc=f"Test - E{e + 1}"):
		with torch.no_grad():
			image, label = image.to(device), label.to(device)
			temp_loss = 0
			for layer in model:
				layer.zero_states()

			for t in range(num_timesteps):
				model_out = model(image)
				loss = nn.functional.mse_loss(model_out, label)
				temp_loss += loss.item()

		optimizer.step()
		optimizer.zero_grad()
		temp_loss /= num_timesteps
		train_loss += temp_loss

	train_loss /= len(train_dataloader)
	train_losses.append(train_loss)

	# Showing stuff
	plt.plot(train_losses, label="Train")
	plt.plot(test_losses, label="Test")
	plt.title("Loss")
	plt.legend()
	plt.show()

	plt.plot(garbate_losses, label="Train")
	plt.title("Train loss across time")
	plt.legend()
	plt.show()

	time.sleep(0.1)
	print(f"E{e + 1} - Train: {train_loss}, Test: {test_loss}")
	time.sleep(0.1)
