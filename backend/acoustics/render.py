"""Apply a simulated room to a dry voice recording.

The original signal is only ever convolved (a linear time-invariant
operation): timing, pitch, pronunciation and delivery are untouched. Gain
staging never clips: the output is scaled, not limited, so the full
reverberation tail is preserved.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal

from .ir import build_impulse_response, ImpulseResponse
from .metrics import measure_ir
from .models import RoomParameters, Placement, MixControls


@dataclass
class RenderResult:
    audio: np.ndarray        # (channels, N), float, peak <= ceiling
    fs: int
    ir: ImpulseResponse
    gain_db: float           # gain applied after convolution
    measured: dict           # metrics measured from the simulated omni IR
    info: dict


def convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Full linear convolution (length len(x)+len(h)-1); keeps the whole tail."""
    return signal.oaconvolve(x, h, mode="full")


def trim_tail(y: np.ndarray, fs: int, floor_db: float = -90.0, margin_s: float = 0.05) -> np.ndarray:
    """Remove trailing samples below floor_db re. peak, keeping a margin and a short fade."""
    env = np.max(np.abs(y), axis=0)
    peak = env.max()
    if peak <= 0:
        return y
    above = np.where(env > peak * 10 ** (floor_db / 20))[0]
    end = min(y.shape[1], above[-1] + int(margin_s * fs)) if len(above) else y.shape[1]
    out = y[:, :end].copy()
    nf = min(int(0.02 * fs), out.shape[1])
    out[:, -nf:] *= np.linspace(1, 0, nf)[None, :] ** 2
    return out


def active_rms(x: np.ndarray, fs: int, gate_db: float = -35.0) -> float:
    """RMS over 20 ms frames within gate_db of the loudest frame (ignores silence)."""
    x = np.atleast_2d(x).mean(axis=0)
    n = max(int(0.02 * fs), 1)
    frames = x[: len(x) // n * n].reshape(-1, n)
    p = np.mean(frames ** 2, axis=1)
    if p.size == 0 or p.max() <= 0:
        return 0.0
    return float(np.sqrt(np.mean(p[p >= p.max() * 10 ** (gate_db / 10)])))


def render(dry: np.ndarray, fs: int, room: RoomParameters, placement: Placement,
           mix: MixControls | None = None, ir: ImpulseResponse | None = None,
           seed: int = 0) -> RenderResult:
    mix = mix or MixControls()
    x = np.array(dry, dtype=float, copy=True)          # never mutate the original
    if ir is None or ir.fs != fs:
        ir = build_impulse_response(room, placement, fs=fs, seed=seed)
    h = ir.combine(mix.wet_gain, mix.reflection_intensity)
    chans = [1, 2] if mix.stereo else [0]
    y = np.stack([convolve(x, h[c]) for c in chans])
    ceiling = 10 ** (mix.ceiling_dbfs / 20)
    peak = float(np.max(np.abs(y)))
    if mix.output_mode == "peak":
        g = ceiling / peak
    elif mix.output_mode == "match":
        g = min(active_rms(x, fs) / max(active_rms(y, fs), 1e-12), ceiling / peak)
    else:  # physical: preserve absolute level re. 1 m unless it would clip
        g = min(1.0, ceiling / peak)
    y = trim_tail(y * g, fs)
    measured = measure_ir(h[0], fs, ir.t_direct, direct=ir.direct[0])
    return RenderResult(y, fs, ir, float(20 * np.log10(g)), measured, ir.info)
