"""Synthetic tensor example. No datasets, weights or simulator required."""
import torch

from model import AEGISNetwork


def main() -> None:
    torch.manual_seed(0)  # Synthetic example only, not an experiment seed.
    model = AEGISNetwork().eval()
    batch, candidates = 1, 46
    depth = 0.1 + 19.9 * torch.rand(batch, 23, 40)
    ranges = 0.1 + 19.9 * torch.rand(batch, 16, 360)
    lidar = torch.stack((ranges, torch.ones_like(ranges)), dim=1)
    state = torch.zeros(batch, 16)
    features = torch.zeros(batch, candidates, 374)
    with torch.inference_mode():
        costs, preferences, hidden = model(depth, lidar, state, features)
        next_costs, _, _ = model(depth, lidar, state, features, hidden)
    print("costs:", tuple(costs.shape))
    print("normalized preferences:", tuple(preferences.shape))
    print("recurrent state:", tuple(hidden.shape))
    print("next-step costs:", tuple(next_costs.shape))
    print("Random initialization only; this is not a navigation demonstration.")


if __name__ == "__main__":
    main()
