from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn


@dataclass(frozen=True)
class LocalVarianceSummary:
    neuron_count: int
    mean: float
    minimum: float
    maximum: float


def _neuron_parameter_matrix(module: nn.Module) -> torch.Tensor | None:
    if isinstance(module, nn.Linear):
        matrix = module.weight.flatten(start_dim=1)
        if module.bias is not None:
            matrix = torch.cat([matrix, module.bias[:, None]], dim=1)
        return matrix

    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        matrix = module.weight.flatten(start_dim=1)
        if module.bias is not None:
            matrix = torch.cat([matrix, module.bias[:, None]], dim=1)
        return matrix

    return None


def iter_neuron_parameter_matrices(model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    for name, module in model.named_modules():
        matrix = _neuron_parameter_matrix(module)
        if matrix is not None and matrix.size(1) > 1:
            yield name, matrix


def local_weight_variance_loss(model: nn.Module) -> torch.Tensor:
    """
    Mean neuron-local variance for Linear and Conv layers.

    Each output neuron/filter contributes one variance. Its vector is the
    flattened incoming weights plus the bias scalar, matching the view that a
    bias is a weight on a constant input of 1.
    """
    variances = [
        matrix.var(dim=1, unbiased=False)
        for _, matrix in iter_neuron_parameter_matrices(model)
    ]
    if variances:
        return torch.cat(variances).mean()

    parameter = next(model.parameters(), None)
    if parameter is None:
        return torch.tensor(0.0)
    return parameter.new_zeros(())


@torch.no_grad()
def local_weight_variance_summary(model: nn.Module) -> LocalVarianceSummary:
    variances = [
        matrix.var(dim=1, unbiased=False).detach().flatten()
        for _, matrix in iter_neuron_parameter_matrices(model)
    ]
    if not variances:
        return LocalVarianceSummary(neuron_count=0, mean=0.0, minimum=0.0, maximum=0.0)

    values = torch.cat(variances)
    return LocalVarianceSummary(
        neuron_count=int(values.numel()),
        mean=float(values.mean().cpu()),
        minimum=float(values.min().cpu()),
        maximum=float(values.max().cpu()),
    )
