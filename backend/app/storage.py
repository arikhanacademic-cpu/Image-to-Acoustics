"""Content-addressed file storage for generated speech, images and renders."""
from __future__ import annotations
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

KINDS = {"speech", "image", "render", "analysis"}
ID_RE = re.compile(r"^[a-f0-9]{16,64}$")


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        for k in KINDS:
            (self.root / k).mkdir(parents=True, exist_ok=True)

    def path(self, kind: str, file_id: str, ext: str) -> Path:
        if kind not in KINDS or not ID_RE.match(file_id) or not re.match(r"^[a-z0-9]{2,5}$", ext):
            raise ValueError("invalid file reference")
        return self.root / kind / f"{file_id}.{ext}"

    def mp3_for(self, wav: Path, bitrate: str = "256k") -> Path:
        mp3 = wav.with_suffix(".mp3")
        if not mp3.exists():
            if not shutil.which("ffmpeg"):
                raise RuntimeError("FFmpeg is required for MP3 export")
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame",
                                "-b:a", bitrate, str(mp3)], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("MP3 encoding failed")
        return mp3


def digest(*parts: bytes | str, n: int = 32) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else p.encode())
        h.update(b"\x00")
    return h.hexdigest()[:n]
