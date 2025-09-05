# --- Dummy test: construct block, make inputs, run forward ---
import torch
import torch.nn as nn
from _modules import AxialAttentionBlock, ContinuousTimeEmbed

# Instantiate helper & block
D = 64
heads = 4
time_dim = 128
block = AxialAttentionBlock(dim=D, num_heads=heads, time_emb_dim=time_dim, dropout=0.05)
time_encoder = ContinuousTimeEmbed(out_dim=time_dim, n_frequencies=64)

# Dummy batch
B, C, H, W = 128, 3, 28, 28
device = torch.device("cpu")
# Create a dummy image (in logits or normalized) and project to D using 1x1 conv
img = torch.randn(B, C, H, W, device=device)
img_proj = nn.Conv2d(C, D, kernel_size=1).to(device)
x = img_proj(img)  # [B, D, H, W]

# Create per-pixel conditioning [B, e, H, W] and project to D with 1x1 conv
e = 10
conditioning = torch.randn(B, e, H, W, device=device)
cond_proj = nn.Conv2d(e, D, kernel_size=1).to(device)
guidance = cond_proj(conditioning)  # [B, D, H, W]

# Sample continuous timesteps in [0,1)
times = torch.rand(B, device=device)  # e.g. torch.rand(batch_size)
t_emb = time_encoder(times)  # [B, time_dim]

# Forward pass
out = block(x, guidance=guidance, t_emb=t_emb)
print("x in:", x.shape, "guidance:", guidance.shape, "t_emb:", t_emb.shape, "out:", out.shape)
# out is [B, D, H, W] and can be passed to a final 1x1 conv to produce predicted noise in C channels.
