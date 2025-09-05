import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContinuousTimeEmbed(nn.Module):
	def __init__(self, out_dim=128, n_frequencies=64):
		super().__init__()
		self.n_frequencies = n_frequencies
		# MLP to map Fourier features -> final t_emb
		self.mlp = nn.Sequential(
			nn.Linear(n_frequencies * 2, out_dim),
			nn.GELU(),
			nn.Linear(out_dim, out_dim)
		)
		# Choose log-spaced frequencies if you like; here simple 1..n
		freqs = torch.arange(1.0, n_frequencies + 1.0)
		self.register_buffer("freqs", freqs * (2.0 * math.pi))  # [n_frequencies]

	def forward(self, t: torch.Tensor):
		"""
		t: [B] float in [0,1)
		returns: t_emb [B, out_dim]
		"""
		B = t.shape[0]
		# t: [B, 1] -> multiply by freqs -> [B, n_frequencies]
		tproj = t.unsqueeze(1) * self.freqs.unsqueeze(0)  # [B, n_frequencies]
		sin_feat = torch.sin(tproj)  # [B, n_frequencies]
		cos_feat = torch.cos(tproj)  # [B, n_frequencies]
		feat = torch.cat([sin_feat, cos_feat], dim=-1)  # [B, 2*n_frequencies]
		out = self.mlp(feat)
		return out


class AxialAttentionBlock(nn.Module):
	def __init__(self, dim, num_heads, time_emb_dim=128, dropout=0.0):
		"""
		dim: channel dimension D
		num_heads: number of attention heads (D must be divisible by num_heads)
		time_emb_dim: dimension of t-embedding input
		"""
		super().__init__()
		assert dim % num_heads == 0, "dim must be divisible by num_heads"
		self.dim = dim
		self.num_heads = num_heads
		self.time_emb_dim = time_emb_dim

		# Shared axial MHA used for row & column passes
		self.axial_mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)

		# Channel normalization: use GroupNorm(1, dim) (close to LayerNorm across channels for conv features)
		self.norm_chan_attn = nn.GroupNorm(1, dim)  # before axial attention & FiLM
		self.norm_chan_cross = nn.GroupNorm(1, dim)  # before cross fusion (kept for clarity)
		self.norm_chan_ffn = nn.GroupNorm(1, dim)  # before FFN & FiLM

		# Time -> FiLM params for attention (gamma_attn, beta_attn) and for FFN (gamma_ffn, beta_ffn)
		self.time_to_attn = nn.Sequential(nn.Linear(time_emb_dim, time_emb_dim), nn.GELU(),
										  nn.Linear(time_emb_dim, 2 * dim))
		self.time_to_ffn = nn.Sequential(nn.Linear(time_emb_dim, time_emb_dim), nn.GELU(),
										 nn.Linear(time_emb_dim, 2 * dim))

		# Guidance projection (1x1 conv) -> projects guidance [B, D, H, W] -> [B, D, H, W]
		self.guidance_proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)

		# FFN as conv1x1 (token-wise)
		self.ffn_conv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
		self.ffn_conv2 = nn.Conv2d(dim * 4, dim, kernel_size=1)

		self.dropout = nn.Dropout(dropout)

	def axial_self_attention(self, x):
		"""
		x: [B, D, H, W]
		returns: (attn_row_out, attn_col_out) each [B, D, H, W]
		"""
		B, D, H, W = x.shape

		# Row pass: treat each row as an independent sequence of length W
		# Reshape -> [B*H, W, D] (batch_first for MHA)
		x_row = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, D)  # [B*H, W, D]
		# Query/key/value all from x_row
		attn_row_out, _ = self.axial_mha(x_row, x_row, x_row, need_weights=False)
		attn_row_out = attn_row_out.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

		# Column pass: treat each column as independent sequence of length H
		x_col = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, D)  # [B*W, H, D]
		attn_col_out, _ = self.axial_mha(x_col, x_col, x_col, need_weights=False)
		attn_col_out = attn_col_out.view(B, W, H, D).permute(0, 3, 2, 1).contiguous()  # [B, D, H, W]

		return attn_row_out, attn_col_out

	def forward(self, x, guidance, t_emb):
		"""
		x: [B, D, H, W]  (image latent)
		guidance: [B, D, H, W] (per-pixel conditioning already projected to D)
		t_emb: [B, time_emb_dim] continuous timestep embedding
		returns: updated x [B, D, H, W]
		"""
		B, D, H, W = x.shape
		device = x.device

		# --- FiLM params from time embedding ---
		attn_out = self.time_to_attn(t_emb)  # [B, 2*D]
		gamma_attn, beta_attn = attn_out.chunk(2, dim=-1)  # each [B, D]
		gamma_attn = 1.0 + gamma_attn  # initialize around identity

		ffn_out = self.time_to_ffn(t_emb)
		gamma_ffn, beta_ffn = ffn_out.chunk(2, dim=-1)
		gamma_ffn = 1.0 + gamma_ffn

		# --- Self-attention stage ---
		# Normalize channels
		x_attn = self.norm_chan_attn(x)  # still [B,D,H,W]
		# Apply FiLM (broadcast gamma/beta over spatial dims)
		gamma_attn_exp = gamma_attn.view(B, D, 1, 1)
		beta_attn_exp = beta_attn.view(B, D, 1, 1)
		x_attn = x_attn * gamma_attn_exp + beta_attn_exp

		# Axial self-attention (row + column)
		attn_row_out, attn_col_out = self.axial_self_attention(x_attn)
		x = x + attn_row_out + attn_col_out
		x = self.dropout(x)

		# --- Guidance fusion (elementwise add) ---
		# guidance already projected to D channels upstream (avoid repeating)
		guid_proj = self.guidance_proj(guidance)  # [B, D, H, W]
		x = x + guid_proj

		# --- FFN stage (with FiLM) ---
		x_ffn_norm = self.norm_chan_ffn(x)
		gamma_ffn_exp = gamma_ffn.view(B, D, 1, 1)
		beta_ffn_exp = beta_ffn.view(B, D, 1, 1)
		x_ffn_norm = x_ffn_norm * gamma_ffn_exp + beta_ffn_exp

		# FFN as 1x1 convs
		h = self.ffn_conv1(x_ffn_norm)  # [B, 4D, H, W]
		h = F.gelu(h)
		h = self.ffn_conv2(h)  # [B, D, H, W]
		h = self.dropout(h)
		x = x + h

		return x
