import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# === Constants ===
NUM_CLASSES = 10
IMAGE_SIZE = 28 * 28  # Flattened

# === Transform ===
transform = transforms.Compose([
	transforms.ToTensor(),  # Converts to [0,1], shape [1, 28, 28]
	transforms.Lambda(lambda x: x.view(-1))  # Flatten to [784]
])


# === One-hot encoding helper ===
def one_hot_encode(labels, num_classes=NUM_CLASSES):
	return nn.functional.one_hot(labels, num_classes=num_classes).float()


# === Custom collate_fn to return (flattened_image, one_hot_label) ===
def collate_fn(batch):
	images, labels = zip(*batch)
	images = torch.stack(images)  # [B, 784]
	labels = torch.tensor(labels)  # [B]
	labels = one_hot_encode(labels, NUM_CLASSES)  # [B, 10]
	return images, labels


# === Dataloaders ===
def get_mnist_dataloaders(batch_size: int):
	train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
	test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

	return train_loader, test_loader


# # === Per-example MSE loss ===
# def per_example_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
# 	"""
# 	predictions: Tensor of shape [A, B]
# 	targets: Tensor of shape [A, B]
# 	returns: Tensor of shape [A], where each element is the MSE for a single example
# 	"""
# 	return nn.functional.mse_loss(predictions, targets, reduction='sum').mean(dim=1)


# ======================================================================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"


class GloballyModulatedModel(nn.Module):
	def __init__(self, middle_size: int, temperature: float = 0.5, dirichlet_conc: float = 1.0):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(784, middle_size),
			nn.LeakyReLU(),
			nn.Linear(middle_size, middle_size),
			nn.LeakyReLU(),
			nn.Linear(middle_size, 10),  # logits
		)

		self.temperature = float(temperature)
		self.dirichlet_conc = float(dirichlet_conc)

		# temporaries for backward
		self._last_model_sample = None  # differentiable sample (requires_grad -> True)
		self._last_random = None  # non-diff Dirichlet sample (anchor)
		self._last_logits = None

	def forward(self, x: torch.Tensor, sample_model: bool = True, sample_random: bool = True):
		"""
		Run model and return (noise, sample).
		- x: [B, 784]
		- sample_model: if True => sample differentiable Gumbel-Softmax; else return softmax probs
		- sample_random: if True => produce a non-diff Dirichlet sample as 'noise'; else noise=None
		"""
		assert x.dim() == 2 and x.size(1) == 784, "input must be [B, 784]"
		device = x.device
		B = x.size(0)

		logits = self.net(x)  # [B, C]
		self._last_logits = logits

		# model sample (differentiable if sample_model=True)
		if sample_model:
			model_sample = nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
		else:
			model_sample = nn.functional.softmax(logits,
												 dim=-1)  # deterministic softmax (still differentiable but no sampling)
		# we keep the model_sample as the differentiable tensor used by backward
		self._last_model_sample = model_sample

		# random noise / anchor: non-differentiable Dirichlet sample per example
		if sample_random:
			conc = torch.ones(logits.size(-1), device=device) * float(self.dirichlet_conc)
			dirich = torch.distributions.Dirichlet(conc)
			noise = dirich.sample((B,)).to(device)  # shape [B, C]
			# ensure no grad attached
			noise = noise.detach()
			self._last_random = noise
		else:
			noise = None
			self._last_random = None

		return noise, model_sample

	def backward(self, global_loss: torch.Tensor) -> float:
		"""
		global_loss: Tensor [B] (per-example scalar, e.g. mse(noise, target))
		Computes:
			fake_loss = per-example mse(model_sample, noise)  -> [B]
			real_loss = mse(fake_loss, global_loss) -> scalar
		Calls real_loss.backward() to populate gradients.
		"""
		if self._last_model_sample is None or self._last_random is None:
			raise RuntimeError("forward(..., sample_random=True, sample_model=...) must be called before backward()")

		B = self._last_model_sample.size(0)
		if not (global_loss.dim() == 1 and global_loss.size(0) == B):
			raise ValueError(f"global_loss must be shape [B], got {tuple(global_loss.shape)}")

		# per-example MSE between model sample and noise
		fake_loss = ((self._last_model_sample - self._last_random) ** 2).mean(dim=1)  # [B]

		real_loss = nn.functional.mse_loss(fake_loss, global_loss)

		# Backpropagate to network parameters
		real_loss.backward()

		# cleanup
		self._last_model_sample = None
		self._last_random = None
		self._last_logits = None

		return float(real_loss.item())

	def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Deterministic evaluation: return softmax probabilities [B, C] detached."""
		logits = self.net(x)
		probs = nn.functional.softmax(logits, dim=-1)
		return probs.detach().clone()


model = GloballyModulatedModel(128).to(device)

BATCH_SIZE = 1
NUM_EPOCHS = 100
train_dloader, test_dloader = get_mnist_dataloaders(BATCH_SIZE)

optimizer = torch.optim.AdamW(model.parameters(), 1e-4)

train_losses, test_losses = [], []

for e in range(NUM_EPOCHS):
	model.train()
	train_loss = 0
	argmaxes = []
	for (image, label) in tqdm(train_dloader, total=len(train_dloader), desc=f"Train - E{e + 1}"):
		image, label = image.to(device), label.to(device)
		b, six_seven = image.shape
		model.zero_grad()
		noise, sample = model.forward(image, sample_model=True)
		with torch.no_grad():
			global_loss = ((noise - label) ** 2).mean(dim=1)
		loss = model.backward(global_loss)
		optimizer.step()

		with torch.no_grad():
			argmax_indices = torch.argmax(sample, dim=-1)
			argmaxes.extend(argmax_indices.tolist())
			printable_loss = nn.functional.mse_loss(sample, label)

		train_loss += printable_loss.item()

	train_loss /= len(train_dloader)
	train_losses.append(train_loss)

	model.eval()
	test_loss = 0
	for (image, label) in tqdm(test_dloader, total=len(test_dloader), desc=f"Test - E{e + 1}"):
		with torch.no_grad():
			image, label = image.to(device), label.to(device)
			noise, sample = model.forward(image, sample_model=False)
			printable_loss = nn.functional.mse_loss(sample, label)
			test_loss += printable_loss.item()
	test_loss /= len(test_dloader)
	test_losses.append(test_loss)

	plt.plot(train_losses, label="Train")
	plt.plot(test_losses, label="Test")
	plt.title("Loss")
	plt.legend()
	plt.show()

	from collections import Counter

	counts = Counter(argmaxes)
	all_keys = list(range(10))
	frequencies = [counts.get(k, 0) for k in all_keys]
	plt.bar(all_keys, frequencies)
	plt.xlabel('Integer')
	plt.ylabel('Frequency')
	plt.title('Argmax')
	plt.xticks(all_keys)
	plt.show()

	time.sleep(0.1)
	print(f"E{e + 1} - Train: {train_loss}, Test: {test_loss}")
	time.sleep(0.1)
