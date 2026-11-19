"""AEGIS main neural network only. No planner, data loader or file access."""
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class MultimodalEncoder(nn.Module):
    """Depth, range/mask and precomputed state encoding with GRU memory."""

    def __init__(self) -> None:
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Flatten(), nn.Linear(64 * 3 * 5, 128), nn.SiLU(),
        )
        self.lidar = nn.Sequential(
            nn.Conv2d(2, 8, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(8, 16, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
            nn.Flatten(), nn.Linear(32 * 2 * 45, 128), nn.SiLU(),
        )
        self.own = nn.Sequential(nn.Linear(16, 32), nn.SiLU())
        self.memory = nn.GRU(288, 128, batch_first=True)
        # Retained for encoder state-dict compatibility. The candidate network
        # below does not use this legacy projection in its forward pass.
        self.head = nn.Linear(128, 2)

    def encode(self, depth: Tensor, lidar: Tensor, state: Tensor) -> Tensor:
        safe_depth = torch.where(
            torch.isfinite(depth) & (depth > 0.04),
            depth,
            torch.full_like(depth, 0.1),
        )
        image = (safe_depth.clamp(0.1, 20).reciprocal() - 0.05) / 4
        ranges = (lidar[:, 0].clamp(0.1, 20).reciprocal() - 0.05) / 4
        lidar_input = torch.stack((ranges, lidar[:, 1]), dim=1)
        return torch.cat(
            (self.image(image[:, None]), self.lidar(lidar_input), self.own(state)),
            dim=-1,
        )


class PreferenceHead(nn.Module):
    """One shared MLP applied to each externally described candidate."""

    def __init__(self, feature_dim: int = 374) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128 + feature_dim, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1),
        )

    def forward(self, hidden: Tensor, features: Tensor) -> Tensor:
        repeated = hidden[:, None].expand(-1, features.shape[1], -1)
        return self.layers(torch.cat((repeated, features), dim=-1))[..., 0]


class AEGISNetwork(nn.Module):
    """Return candidate costs, normalized preferences and recurrent state.

    Inputs: depth [B,23,40], lidar [B,2,16,360], state [B,16],
    candidate features [B,N,feature_dim], optional hidden [1,B,128].
    The navigation configuration uses N=46 and feature_dim=374.
    Outputs are preferences, not calibrated collision probabilities.
    """

    def __init__(self, feature_dim: int = 374) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = feature_dim
        self.encoder = MultimodalEncoder()
        self.policy = PreferenceHead(feature_dim)

    def forward(
        self, depth: Tensor, lidar: Tensor, state: Tensor, features: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if depth.ndim != 3 or tuple(depth.shape[1:]) != (23, 40):
            raise ValueError("depth must have shape [B,23,40]")
        batch = depth.shape[0]
        if tuple(lidar.shape) != (batch, 2, 16, 360):
            raise ValueError("lidar must have shape [B,2,16,360]")
        if tuple(state.shape) != (batch, 16):
            raise ValueError("state must have shape [B,16]")
        if (features.ndim != 3 or features.shape[0] != batch
                or features.shape[1] < 1 or features.shape[2] != self.feature_dim):
            raise ValueError("features must have shape [B,N,feature_dim], N >= 1")
        if hidden is not None and tuple(hidden.shape) != (1, batch, 128):
            raise ValueError("hidden must have shape [1,B,128]")
        encoded = self.encoder.encode(depth, lidar, state)
        sequence, next_hidden = self.encoder.memory(encoded[:, None], hidden)
        scores = self.policy(sequence[:, 0], features)
        return -scores, scores.softmax(dim=-1), next_hidden
