"""Map normalised image coordinates <-> room coordinates.

Single-image depth is ambiguous, so this is a deliberately simple perspective
guess: image x maps to room width; image height maps to depth, with the bottom
edge near the camera (y ~ 0.08 L) and ~30 % from the top at the far wall
(y ~ 0.92 L). The UI exposes a manual distance that overrides this.
"""
from __future__ import annotations
import numpy as np

NEAR, SPAN, HORIZON = 0.08, 0.84, 0.70


def norm_to_room(nx: float, ny: float, dims, z: float) -> tuple[float, float, float]:
    W, L, H = dims
    depth = float(np.clip((1.0 - ny) / HORIZON, 0.0, 1.0))
    x = float(np.clip(nx, 0.03, 0.97)) * W
    return (x, (NEAR + SPAN * depth) * L, min(z, H - 0.2))


def room_to_norm(x: float, y: float, dims) -> dict:
    W, L, _ = dims
    depth = float(np.clip((y / L - NEAR) / SPAN, 0, 1))
    return {"x": float(np.clip(x / W, 0, 1)), "y": float(1.0 - HORIZON * depth)}
