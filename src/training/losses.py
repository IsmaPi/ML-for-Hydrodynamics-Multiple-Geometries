"""Loss functions for hydrodynamic displacement prediction.

Implements the MSE loss from Equation (14) of the methodology:
  L_MSE = (1 / 3N) * ||DeltaX_hat - DeltaX||_2^2
"""

import torch
import torch.nn as nn


class DisplacementMSELoss(nn.Module):
    """Per-particle MSE loss normalized by (3N).

    Handles batched PyG data where total_nodes varies per batch.
    Averages the per-sample loss across the batch.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: (total_nodes, 3) predicted displacements.
            target: (total_nodes, 3) ground truth displacements.
            batch: (total_nodes,) batch assignment indices.
        Returns:
            Scalar loss.
        """
        # Per-particle squared error summed over 3D components
        sq_err = (pred - target).pow(2).sum(dim=-1)  # (total_nodes,)

        # Sum squared errors per sample, divide by 3N
        batch_size = batch.max().item() + 1
        sample_losses = torch.zeros(batch_size, device=pred.device)
        counts = torch.bincount(batch, minlength=batch_size).float()

        # Scatter-add squared errors per sample
        sample_losses.scatter_add_(0, batch, sq_err)
        sample_losses = sample_losses / (3.0 * counts)

        return sample_losses.mean()
