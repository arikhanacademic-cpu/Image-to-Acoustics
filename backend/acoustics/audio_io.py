"""Audio loading / saving with validation. MP3 export via FFmpeg."""
from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
import numpy as np
import soundfile as sf

MAX_SECONDS = 600
MIN_SR, MAX_SR = 8000, 192000


class AudioError(ValueError):
    pass


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load any libsndfile-readable file (falls back to FFmpeg decoding).
    Returns mono float64 in [-1, 1] and the sample rate. Raises AudioError."""
    try:
        x, sr = sf.read(path, always_2d=True, dtype="float64")
    except Exception:
        x, sr = _ffmpeg_decode(path)
    if not (MIN_SR <= sr <= MAX_SR):
        raise AudioError(f"Unsupported sample rate {sr} Hz")
    if x.size == 0:
        raise AudioError("Audio file is empty")
    if x.shape[0] / sr > MAX_SECONDS:
        raise AudioError(f"Audio longer than {MAX_SECONDS} s")
    if not np.all(np.isfinite(x)):
        raise AudioError("Audio contains NaN/Inf samples (corrupted?)")
    mono = x.mean(axis=1)
    if np.max(np.abs(mono)) < 1e-6:
        raise AudioError("Audio is silent")
    return mono, int(sr)


def _ffmpeg_decode(path: str):
    if not shutil.which("ffmpeg"):
        raise AudioError("Could not decode audio and FFmpeg is not installed")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "decoded.wav")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-c:a", "pcm_f32le", out],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise AudioError("Could not decode audio (corrupted or unsupported format)")
        x, sr = sf.read(out, always_2d=True, dtype="float64")
    return x, sr


def save_audio(path: str, audio: np.ndarray, sr: int, fmt: str = "wav", bitrate: str = "256k"):
    """audio: (N,) or (C, N). WAV is 24-bit PCM; MP3 encoded with FFmpeg/libmp3lame."""
    a = np.asarray(audio)
    a = a.T if a.ndim == 2 else a
    if fmt == "wav":
        sf.write(path, a, sr, subtype="PCM_24")
    elif fmt == "mp3":
        if not shutil.which("ffmpeg"):
            raise AudioError("MP3 export requires FFmpeg")
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, "x.wav")
            sf.write(tmp, a, sr, subtype="PCM_24")
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-codec:a", "libmp3lame",
                                "-b:a", bitrate, path], capture_output=True, text=True)
            if r.returncode != 0:
                raise AudioError(f"MP3 encoding failed: {r.stderr.strip()[:200]}")
    else:
        raise AudioError(f"Unsupported format {fmt}")
