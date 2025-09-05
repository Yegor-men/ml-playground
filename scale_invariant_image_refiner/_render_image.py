import matplotlib.pyplot as plt
import math
import torch

def render_image(tensor: torch.Tensor, title: str = None):
    """
    Render image(s) from a PyTorch tensor.

    Args:
        tensor (torch.Tensor): Image tensor of shape [B, C, H, W] or [C, H, W].
        title (str, optional): Title for the whole plot.

    Displays:
        Images using matplotlib in a square-esque grid.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # [C, H, W] → [1, C, H, W]
    elif tensor.ndim != 4:
        raise ValueError("Tensor must be of shape [C, H, W] or [B, C, H, W].")

    b, c, h, w = tensor.shape

    # Determine grid size: as square as possible
    cols = math.ceil(math.sqrt(b))
    rows = math.ceil(b / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if b > 1 else [axes]

    for i in range(rows * cols):
        ax = axes[i]
        if i < b:
            img = tensor[i]
            if c == 1:
                img = img.squeeze(0)
                ax.imshow(img.cpu(), cmap='gray')
            elif c == 3:
                img = img.permute(1, 2, 0)
                ax.imshow(img.cpu())
            else:
                raise ValueError(f"Unsupported number of channels: {c}")
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)  # Leave space for title

    plt.tight_layout()
    plt.show()
