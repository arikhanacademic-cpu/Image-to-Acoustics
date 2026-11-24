"""Objective measurements on impulse responses and audio (ISO 3382-1 style).

These are *measured from the simulated IR*, distinct from the *predicted*
values of the statistical model; the UI should label them accordingly.
"""
from __future__ import annotations
import numpy as np
from scipy import signal

from .bands import BANDS_HZ


def octave_filter(x: np.ndarray, fs: int, fc: float, order: int = 4) -> np.ndarray:
    lo, hi = fc / np.sqrt(2), min(fc * np.sqrt(2), 0.45 * fs)
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfilt(sos, x)


def schroeder_db(h: np.ndarray) -> np.ndarray:
    e = np.cumsum((h ** 2)[::-1])[::-1]
    return 10 * np.log10(np.maximum(e / max(e[0], 1e-30), 1e-30))


def _decay_time(edc_db, fs, start_db, end_db):
    """Fit a line between start_db and end_db on the EDC, extrapolate to -60 dB."""
    idx = np.where((edc_db <= start_db) & (edc_db >= end_db))[0]
    if len(idx) < 10 or edc_db[-1] > end_db:
        return float("nan")
    t = idx / fs
    slope, _ = np.polyfit(t, edc_db[idx], 1)
    return float(-60.0 / slope) if slope < 0 else float("nan")


def measure_ir(h: np.ndarray, fs: int, t_direct: float, direct: np.ndarray | None = None) -> dict:
    """Per-band T20/T30/EDT/C50/C80/D50/DRR; headline C50/C80/D50/DRR = 500 Hz-1 kHz mean.
    DRR: if the separate `direct` component is given (simulation), DRR is exact per band:
    E(direct) / E(h - direct). Otherwise a 3.5 ms window is used, which is biased low in
    octave bands because the band filter smears the direct impulse beyond the window."""
    i0 = max(int(round(t_direct * fs)) - int(0.001 * fs), 0)
    h_full = np.asarray(h, float)
    h = h_full[i0:]
    out = {"bands_hz": BANDS_HZ.tolist(), "T20": [], "T30": [], "EDT": []}
    for fc in BANDS_HZ:
        if fc * np.sqrt(2) >= 0.5 * fs:
            for k in ("T20", "T30", "EDT"):
                out[k].append(float("nan"))
            continue
        edc = schroeder_db(octave_filter(h, fs, fc))
        out["T20"].append(_decay_time(edc, fs, -5, -25))
        out["T30"].append(_decay_time(edc, fs, -5, -35))
        out["EDT"].append(_decay_time(edc, fs, 0, -10))
    i50, i80, i_dir = int(0.050 * fs), int(0.080 * fs), int(0.0035 * fs)

    def clarity(x):
        e = x ** 2
        tot = e.sum() + 1e-30
        return (10 * np.log10(e[:i50].sum() / max(e[i50:].sum(), 1e-30)),
                10 * np.log10(e[:i80].sum() / max(e[i80:].sum(), 1e-30)),
                e[:i50].sum() / tot,
                10 * np.log10(e[:i_dir].sum() / max(e[i_dir:].sum(), 1e-30)))

    per_band = [clarity(octave_filter(h, fs, fc)) if fc * np.sqrt(2) < 0.5 * fs else (np.nan,) * 4
                for fc in BANDS_HZ]
    for j, key in enumerate(("C50_bands", "C80_bands", "D50_bands", "DRR_bands")):
        out[key] = [float(v[j]) for v in per_band]
    # Headline values: mean of the 500 Hz and 1 kHz octaves (ISO 3382 "mid" convention).
    for key in ("C50", "C80", "DRR"):
        out[f"{key}_dB"] = float(np.mean(out[f"{key}_bands"][2:4]))
    out["D50"] = float(np.mean(out["D50_bands"][2:4]))
    out["C50_broadband_dB"], _, _, out["DRR_broadband_dB"] = (float(v) for v in clarity(h))
    if direct is not None:
        d = np.asarray(direct, float)
        rest = h_full - d
        drr = []
        for fc in BANDS_HZ:
            if fc * np.sqrt(2) >= 0.5 * fs:
                drr.append(float("nan")); continue
            ed, er = np.sum(octave_filter(d, fs, fc) ** 2), np.sum(octave_filter(rest, fs, fc) ** 2)
            drr.append(float(10 * np.log10(ed / max(er, 1e-30))))
        out["DRR_bands"] = drr
        out["DRR_dB"] = float(np.mean(drr[2:4]))
        out["DRR_broadband_dB"] = float(10 * np.log10(np.sum(d ** 2) / max(np.sum(rest ** 2), 1e-30)))
        out["DRR_method"] = "exact_components"
    else:
        out["DRR_method"] = "window_3.5ms"
    return out


def spectral_centroid(x: np.ndarray, fs: int) -> float:
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return float((f * X).sum() / max(X.sum(), 1e-30))


def band_levels_db(x: np.ndarray, fs: int) -> list[float]:
    return [float(10 * np.log10(np.mean(octave_filter(x, fs, fc) ** 2) + 1e-20)) for fc in BANDS_HZ]
