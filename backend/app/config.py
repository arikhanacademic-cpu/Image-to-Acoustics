"""Server-side configuration. Credentials are read from the environment / .env
and never returned by any endpoint or written to logs."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "").strip()
    elevenlabs_model_id: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
    elevenlabs_output_format: str = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
    max_tts_chars: int = int(os.getenv("MAX_TTS_CHARS", "600"))
    vision_provider: str = os.getenv("VISION_PROVIDER", "demo").strip().lower()
    vision_api_key: str = os.getenv("VISION_API_KEY", "").strip()
    vision_model: str = os.getenv("VISION_MODEL", "").strip()
    data_dir: Path = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
    sample_voice: Path = ROOT / "samples" / "voice_dry.wav"

    @property
    def elevenlabs_enabled(self) -> bool:
        return bool(self.elevenlabs_api_key)

    @property
    def vision_enabled(self) -> bool:
        return self.vision_provider in ("anthropic", "openai") and bool(self.vision_api_key)


def get_settings() -> Settings:
    return Settings()
