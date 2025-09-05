import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

from scale_invariant_image_refiner.module_test import guidance

# Constants
BATCH_SIZE = 100
NUM_CLASSES = 10


def global_embed(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
	"""
	Expand an embedding tensor [b, e] to [b, e, h, w] by repeating the embedding across spatial dimensions.

	Args:
		x (torch.Tensor): Input tensor of shape [b, e]
		h (int): Height of the spatial map
		w (int): Width of the spatial map

	Returns:
		torch.Tensor: Output tensor of shape [b, e, h, w]
	"""
	return x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)


def alpha_bar_cosine(t, s=0.008):
	"""Cosine alpha_bar schedule for continuous t in [0,1].
	t: tensor shape [B] or scalar in [0,1]. Returns alpha_bar in (0,1].
	"""
	# Make sure t is a float tensor
	t_tensor = torch.as_tensor(t, dtype=torch.float32)
	# formula: cos(((t + s)/(1+s)) * pi/2) ** 2
	return torch.cos((t_tensor + s) / (1.0 + s) * (math.pi / 2.0)).clamp(min=0.0, max=1.0) ** 2


def corrupt_image(x0, t, *, alpha_bar_fn=alpha_bar_cosine):
	"""
	Corrupt images according to a continuous diffusion schedule.

	Inputs
	- x0: float tensor [B, C, H, W], expected range [-1, 1] (common convention).
	- t: float tensor [B] with values in [0,1], where t=0 -> clean, t=1 -> nearly pure noise.
		 (Typically sampled with torch.rand(batch_size) during training; during inference
		 you walk t from ~1 down to ~0.)
	- alpha_bar_fn: function mapping t -> alpha_bar(t). Defaults to a cosine schedule.

	Returns (x_t, eps):
	- x_t: corrupted image [B, C, H, W]
	- eps: the Gaussian noise [B, C, H, W] that was mixed in (the model's training target)

	Notes:
	- Use the same alpha_bar_fn at training and inference time (samplers assume the same mapping).
	- This function uses the standard mixture:
		  x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * eps
	  where eps ~ N(0, I).
	- Assumes x0 is already normalized (e.g., [-1,1]). If your data is in [0,1], convert first.
	"""
	if x0.ndim != 4:
		raise ValueError("x0 must be [B,C,H,W]")
	B = x0.shape[0]
	# Ensure t is a float tensor on same device
	t = torch.as_tensor(t, dtype=torch.float32, device=x0.device)
	if t.ndim == 0:
		t = t.unsqueeze(0).expand(B)
	elif t.ndim == 1 and t.shape[0] != B:
		# allow passing single scalar or per-sample vector; expand if scalar
		if t.shape[0] == 1:
			t = t.expand(B)
		else:
			raise ValueError("t must be scalar or length B")
	# compute alpha_bar per sample
	alpha_bar = alpha_bar_fn(t)  # shape [B]
	# reshape to broadcast over images
	sqrt_ab = torch.sqrt(alpha_bar).view(B, 1, 1, 1)
	sqrt_1_ab = torch.sqrt(1.0 - alpha_bar).view(B, 1, 1, 1)
	# sample noise
	eps = torch.rand_like(x0)
	x_t = sqrt_ab * x0 + sqrt_1_ab * eps
	return x_t, eps


def relative_positional_conditioning(x: torch.Tensor):
	"""
	Create relative positional conditioning for an image batch.

	Inputs:
	  x: tensor [B, C, H, W] - only the spatial shape and batch size are used.

	Output:
	  pos: tensor [B, 2, H, W] where channel 0 = relative Y coordinate, channel 1 = relative X coordinate.
		- Coordinates range such that the LONGEST side spans [-1, 1].
		- The shorter side is centered and spans [-r, r] where r = short/long.
		- Example for H==W (square): top-left -> [-1, -1], top-right -> [-1, 1],
									  bottom-left -> [1, -1], bottom-right -> [1, 1].
	"""
	if x.ndim != 4:
		raise ValueError("Input x must have shape [B, C, H, W]")
	B, C, H, W = x.shape
	device = x.device
	dtype = torch.float32

	# Determine which axis is longest
	if H >= W:
		# Y spans -1..1, X spans -W/H .. W/H
		y_coords = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
		x_extent = (W / H)
		x_coords = torch.linspace(-x_extent, x_extent, steps=W, device=device, dtype=dtype)
	else:
		# X spans -1..1, Y spans -H/W .. H/W
		x_coords = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
		y_extent = (H / W)
		y_coords = torch.linspace(-y_extent, y_extent, steps=H, device=device, dtype=dtype)

	# Create meshgrid in (Y, X) order so indexing matches image layout
	yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # each [H, W]

	# Stack as [2, H, W] with channel 0 = Y, channel 1 = X
	pos_hw = torch.stack([yy, xx], dim=0)  # [2, H, W]

	# Expand to batch: [B, 2, H, W]
	pos = pos_hw.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

	return pos


def one_hot_encode(label, num_classes=NUM_CLASSES):
	return torch.nn.functional.one_hot(torch.tensor(label), num_classes=num_classes).float()


class OneHotMNIST(torch.utils.data.Dataset):
	def __init__(self, train=True):
		self.dataset = datasets.MNIST(
			root='data',
			train=train,
			download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),  # Converts to [C, H, W] in [0.0, 1.0]
			])
		)

	def __getitem__(self, index):
		image, label = self.dataset[index]
		one_hot_label = one_hot_encode(label)
		return image, one_hot_label

	def __len__(self):
		return len(self.dataset)


train_dataset = OneHotMNIST(train=True)
test_dataset = OneHotMNIST(train=False)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# images, labels = next(iter(train_dataloader))
# print("Image batch shape:", images.shape)  # [b, c, h, w]
# print("Label batch shape:", labels.shape)  # [b, 10]
# print(global_embed(labels, 28, 28).size())

from tqdm import tqdm
from _render_image import render_image
from _modules import ContinuousTimeEmbed, AxialAttentionBlock
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

time_embed_dim = 128
c_channels = 1
d_channels = 64
num_heads = 4


class SIIR(nn.Module):
	def __init__(self, c_channels, d_channels, time_embed_dim, num_heads):
		super().__init__()

		self.cte = ContinuousTimeEmbed(out_dim=time_embed_dim, n_frequencies=time_embed_dim // 2)
		self.enc = nn.Conv2d(in_channels=c_channels, out_channels=d_channels, kernel_size=1)
		self.pos_expander = nn.Conv2d(in_channels=2, out_channels=d_channels, kernel_size=1)
		self.text_expander = nn.Conv2d(in_channels=10, out_channels=d_channels, kernel_size=1)
		self.dec = nn.Sequential(
			nn.Conv2d(in_channels=d_channels, out_channels=c_channels, kernel_size=1),
			nn.Sigmoid()
		)

		self.aab1 = AxialAttentionBlock(dim=d_channels, num_heads=num_heads, time_emb_dim=time_embed_dim)
		self.aab2 = AxialAttentionBlock(dim=d_channels, num_heads=num_heads, time_emb_dim=time_embed_dim)
		self.aab3 = AxialAttentionBlock(dim=d_channels, num_heads=num_heads, time_emb_dim=time_embed_dim)
		self.aab4 = AxialAttentionBlock(dim=d_channels, num_heads=num_heads, time_emb_dim=time_embed_dim)
		self.aab5 = AxialAttentionBlock(dim=d_channels, num_heads=num_heads, time_emb_dim=time_embed_dim)

	def forward(self, noisy_image, text_cond, pos_cond, times):
		latent_image = self.enc(noisy_image)
		guidance = self.text_expander(text_cond) + self.pos_expander(pos_cond)
		time_vector = self.cte(times)

		latent_image = self.aab1(latent_image, guidance, time_vector)
		latent_image = self.aab2(latent_image, guidance, time_vector)
		latent_image = self.aab3(latent_image, guidance, time_vector)
		latent_image = self.aab4(latent_image, guidance, time_vector)
		latent_image = self.aab5(latent_image, guidance, time_vector)

		predicted_noise = self.dec(latent_image)

		return predicted_noise


model = SIIR(
	c_channels=c_channels,
	d_channels=d_channels,
	time_embed_dim=time_embed_dim,
	num_heads=num_heads
).to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-6)

num_epochs = 100

losses = []

for E in range(num_epochs):
	train_loss = 0
	for i, (images, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True, desc=f"E{E}"):
		images, labels = images.to(device), labels.to(device)
		b, c, h, w = images.shape
		with torch.no_grad():
			times = torch.rand(b).to(device)
			noisy_image, expected_output = corrupt_image(images, times)
			cfg_mask = torch.rand(b).round().view(b, 1, 1, 1).to(device)
			text_cond = global_embed(labels, h, w) * cfg_mask
			pos_cond = relative_positional_conditioning(images)

		optimizer.zero_grad()

		predicted_noise = model(noisy_image, text_cond, pos_cond, times)
		loss = nn.functional.mse_loss(predicted_noise, expected_output)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()

	train_loss /= len(train_dataloader)
	losses.append(train_loss)
	plt.plot(losses)
	plt.show()

	with torch.no_grad():
		test_loss = 0
		for i, (images, labels) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=True,
										desc=f"T{E}"):
			images, labels = images.to(device), labels.to(device)
			b, c, h, w = images.shape
			with torch.no_grad():
				times = torch.rand(b).to(device)
				noisy_image, expected_output = corrupt_image(images, times)
				cfg_mask = torch.rand(b).round().view(b, 1, 1, 1).to(device)
				text_cond = global_embed(labels, h, w) * cfg_mask
				pos_cond = relative_positional_conditioning(images)

			predicted_noise = model(noisy_image, text_cond, pos_cond, times)
			loss = nn.functional.mse_loss(predicted_noise, expected_output)
			test_loss += loss.item()

			if i == 0:
				render_image(noisy_image, title=f"E{E} - Noisy Image")
				render_image(predicted_noise, title=f"E{E} - Predicted")
				render_image(expected_output, title=f"E{E} - Expected")
				render_image((predicted_noise - expected_output) ** 2, title=f"E{E} - Squared Error")

		test_loss /= len(test_dataloader)
		print(f"Test Loss: {test_loss}")
