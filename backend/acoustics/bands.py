"""Octave-band definitions and the band -> continuous-frequency interpolation.

All frequency-dependent quantities in the engine (absorption, air absorption,
directivity, obstruction, head shadow, reverberation time) are specified per
octave band and turned into a continuous transfer function by interpolating
in log-frequency with triangular ("hat") weights. The weights sum to exactly
1 at every frequency, so a flat band profile gives a flat response.
"""
from __future__ import annotations
import numpy as np

BANDS_HZ = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
N_BANDS = len(BANDS_HZ)
SPEED_OF_SOUND = 343.0  # m/s at ~20 degC


def band_weights(freqs: np.ndarray) -> np.ndarray:
    """Return an (N_BANDS, len(freqs)) matrix of cos^2 partition weights in
    log2-frequency. Between centres b and b+1 at position x:
        W_b = cos^2(pi/2 (x-b)),  W_{b+1} = sin^2(pi/2 (x-b)).
    Columns sum to exactly 1, and sqrt(W) is smooth (used to shape noise).
    Below the lowest / above the highest centre the end band has weight 1.
    """
    f = np.maximum(np.asarray(freqs, dtype=float), 1e-6)
    x = np.clip(np.log2(f / BANDS_HZ[0]), 0.0, N_BANDS - 1.0)
    centres = np.arange(N_BANDS)[:, None]
    dist = np.abs(x[None, :] - centres)
    w = np.where(dist < 1.0, np.cos(0.5 * np.pi * dist) ** 2, 0.0)
    return w / w.sum(axis=0, keepdims=True)


def partition_weights(freqs: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """cos^2 partition of unity over arbitrary log-spaced centres (same idea as
    band_weights, but the grid spacing is taken from `centres`)."""
    f = np.maximum(np.asarray(freqs, dtype=float), 1e-6)
    lc = np.log2(np.asarray(centres, float))
    step = lc[1] - lc[0]
    x = np.clip((np.log2(f) - lc[0]) / step, 0.0, len(centres) - 1.0)
    dist = np.abs(x[None, :] - np.arange(len(centres))[:, None])
    w = np.where(dist < 1.0, np.cos(0.5 * np.pi * dist) ** 2, 0.0)
    return w / w.sum(axis=0, keepdims=True)


# 1/3-octave grid used to synthesise the late tail, so that T60 varies
# smoothly with frequency instead of mixing two octave-band exponentials.
TAIL_CENTRES_HZ = 125.0 * 2.0 ** (np.arange(-3, 22) / 3.0)   # ~62 Hz .. ~16 kHz


def interp_log(band_values: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of log(values) over log2(f) from the
    octave centres onto arbitrary frequencies (held constant beyond the ends)."""
    x = np.log2(np.maximum(np.asarray(freqs, float), 1e-6))
    return np.exp(np.interp(x, np.log2(BANDS_HZ), np.log(np.asarray(band_values, float))))


def interp_bands(band_values: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Interpolate per-band values (shape (..., N_BANDS)) onto `freqs`."""
    return np.asarray(band_values) @ band_weights(freqs)


def air_absorption_m() -> np.ndarray:
    """Energy attenuation coefficient m (1/m) per band.

    From ISO 9613-1 attenuation at 20 degC, 50 % RH (dB/km):
    125:0.44 250:1.31 500:2.73 1k:4.66 2k:9.86 4k:32.8 8k:117.
    Energy decays as exp(-m r); m = alpha_dB_per_m / (10 log10 e).
    """
    db_per_km = np.array([0.44, 1.31, 2.73, 4.66, 9.86, 32.8, 117.0])
    return db_per_km / 1000.0 / 4.343
