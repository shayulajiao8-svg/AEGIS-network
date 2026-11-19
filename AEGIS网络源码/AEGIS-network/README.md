# AEGIS: main network source

This minimal release contains only the main PyTorch network: depth and LiDAR
encoders, state projection, GRU, and candidate-conditioned preference head.
It is a partial source release, not a complete navigation implementation or
an end-to-end reproduction package.

## Quick start

With PyTorch installed, run:

```bash
python demo.py
```

The example uses random tensors and random initialization. No trained weights
or real sensor frames are bundled. It does not demonstrate navigation quality.
The extraction was checked using Python 3.8 and PyTorch 2.4.1.

## Tensor interface

| Input | Shape | Contract |
|---|---|---|
| `depth` | `[B, 23, 40]` | Already prepared metric depth; the network applies inverse-depth normalization. |
| `lidar` | `[B, 2, 16, 360]` | Already aligned range image in metres, followed by validity mask. |
| `state` | `[B, 16]` | Precomputed state/task features supplied by the caller. |
| `features` | `[B, N, 374]` | Precomputed candidate descriptors supplied by the caller; the navigation configuration uses `N=46`. |
| `hidden` | `[1, B, 128]` or `None` | Previous GRU state; `None` starts a new sequence. |

`forward` returns `costs [B,N]`, `preferences [B,N]`, and `next_hidden [1,B,128]`.
Costs are negative preference scores; preferences are a softmax over candidates.
These are not calibrated collision probabilities or a safety guarantee.
The legacy `encoder.head` projection is retained for state-dictionary
compatibility and is not used by the candidate forward pass.

The caller is responsible for producing valid, correctly aligned tensors.
Feature construction, candidate generation, selection/admission rules and
control are not provided. No training data, labels, real observations,
checkpoint, auxiliary models, training scripts, ROS interfaces, simulator,
planner, controller, experiment settings or logs are included.

## License

MIT; see `LICENSE`.
