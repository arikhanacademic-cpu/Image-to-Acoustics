"""ElevenLabs text-to-speech via the official Python SDK.

- Credentials come only from the server environment.
- Every generated clip is cached by (voice, model, format, text): repeating a
  request, or changing the environment, never spends credits again.
- Credit safeguards: a per-request character cap, a remaining-quota check,
  and one in-flight generation per cache key (double clicks are free).
- Demo mode (no key): returns the bundled sample voice so the app works offline.
"""
from __future__ import annotations
import io
import re
import threading
import time
from typing import Any, Callable

import numpy as np
import soundfile as sf

from acoustics.audio_io import load_audio
from .config import Settings
from .storage import Store, digest

VOICE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DEMO_VOICE_ID = "demo-narrator"


class TTSError(Exception):
    def __init__(self, status: int, message: str, code: str = "tts_error"):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


def _api_error_to_tts(e: Exception) -> TTSError:
    status = getattr(e, "status_code", None)
    body = getattr(e, "body", None)
    detail = body.get("detail") if isinstance(body, dict) else None
    code = detail.get("status") if isinstance(detail, dict) else None
    msg = detail.get("message") if isinstance(detail, dict) else (detail if isinstance(detail, str) else None)
    if code == "quota_exceeded":
        return TTSError(402, "Your ElevenLabs character quota is used up. Top up or wait for the monthly reset.", code)
    if status == 401:
        return TTSError(502, "ElevenLabs rejected the server's API key. Check ELEVENLABS_API_KEY in the server .env.",
                        code or "unauthorized")
    if status == 429:
        return TTSError(429, "ElevenLabs is rate limiting requests. Wait a few seconds and try again.", code or "rate_limited")
    if status in (400, 404, 422):
        return TTSError(400, f"ElevenLabs could not generate this request: {msg or 'invalid voice or text'}.",
                        code or "bad_request")
    if status is None:
        return TTSError(503, "Could not reach ElevenLabs. Check the server's internet connection.", "unreachable")
    return TTSError(502, f"ElevenLabs returned an error ({status}). Try again shortly.", code or "upstream_error")


class TTSService:
    def __init__(self, settings: Settings, store: Store, client_factory: Callable[[], Any] | None = None):
        self.s = settings
        self.store = store
        self._client = None
        self._client_factory = client_factory
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._voices_cache: tuple[float, list] | None = None
        self._credits_cache: tuple[float, dict] | None = None

    @property
    def demo(self) -> bool:
        return not self.s.elevenlabs_enabled and self._client_factory is None

    def client(self):
        if self._client is None:
            if self._client_factory:
                self._client = self._client_factory()
            else:
                from elevenlabs.client import ElevenLabs
                self._client = ElevenLabs(api_key=self.s.elevenlabs_api_key)
        return self._client

    # ------------------------------------------------------------------ voices
    def voices(self) -> list[dict]:
        if self.demo:
            return [{"id": DEMO_VOICE_ID, "name": "Demo narrator", "description": "Offline sample — set ELEVENLABS_API_KEY for real voices",
                     "labels": {}, "preview_url": None, "category": "demo"}]
        if self._voices_cache and time.time() - self._voices_cache[0] < 600:
            return self._voices_cache[1]
        try:
            resp = self.client().voices.search(page_size=100)
        except Exception as e:  # noqa: BLE001
            raise _api_error_to_tts(e)
        out = []
        for v in resp.voices:
            labels = dict(v.labels or {})
            out.append({"id": v.voice_id, "name": v.name, "category": v.category,
                        "description": v.description or "", "labels": labels, "preview_url": v.preview_url})
        out.sort(key=lambda v: (v["category"] != "premade", v["name"].lower()))
        self._voices_cache = (time.time(), out)
        return out

    # ----------------------------------------------------------------- credits
    def credits(self, fresh: bool = False) -> dict | None:
        if self.demo:
            return None
        if not fresh and self._credits_cache and time.time() - self._credits_cache[0] < 60:
            return self._credits_cache[1]
        try:
            sub = self.client().user.subscription.get()
        except Exception:  # noqa: BLE001 — credits are informative; never block on them
            return None
        c = {"used": sub.character_count, "limit": sub.character_limit,
             "remaining": max(sub.character_limit - sub.character_count, 0),
             "resets_at": getattr(sub, "next_character_count_reset_unix", None)}
        self._credits_cache = (time.time(), c)
        return c

    # --------------------------------------------------------------- synthesis
    def _lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def synthesize(self, voice_id: str, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise TTSError(400, "Enter some dialogue first.", "empty_text")
        if len(text) > self.s.max_tts_chars:
            raise TTSError(400, f"Dialogue is {len(text)} characters; the limit per clip is {self.s.max_tts_chars}.",
                           "too_long")
        if not VOICE_ID_RE.match(voice_id or ""):
            raise TTSError(400, "Choose a voice.", "bad_voice")
        model, fmt = self.s.elevenlabs_model_id, self.s.elevenlabs_output_format
        key = digest("demo" if self.demo else "el", voice_id, model, fmt, text)
        wav = self.store.path("speech", key, "wav")
        with self._lock(key):
            if wav.exists():
                info = sf.info(str(wav))
                return {"id": key, "cached": True, "characters": len(text), "duration": info.duration,
                        "sample_rate": info.samplerate, "demo": self.demo}
            if self.demo:
                x, sr = load_audio(str(self.s.sample_voice))
            else:
                credits = self.credits(fresh=True)
                if credits is not None and credits["remaining"] < len(text):
                    raise TTSError(402, f"This clip needs {len(text)} characters but only "
                                        f"{credits['remaining']} remain on the ElevenLabs account.", "insufficient_credits")
                try:
                    chunks = self.client().text_to_speech.convert(
                        voice_id=voice_id, text=text, model_id=model, output_format=fmt)
                    data = b"".join(chunks)
                except TTSError:
                    raise
                except Exception as e:  # noqa: BLE001
                    raise _api_error_to_tts(e)
                x, sr = self._decode(data, fmt)
                self._credits_cache = None
            sf.write(str(wav), x, sr, subtype="PCM_24")
            return {"id": key, "cached": False, "characters": len(text), "duration": len(x) / sr,
                    "sample_rate": sr, "demo": self.demo}

    def _decode(self, data: bytes, fmt: str) -> tuple[np.ndarray, int]:
        if not data:
            raise TTSError(502, "ElevenLabs returned empty audio.", "empty_audio")
        if fmt.startswith("pcm_"):
            sr = int(fmt.split("_")[1])
            x = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float64) / 32768.0
            return x, sr
        tmp = self.store.root / "speech" / f"tmp_{digest(data)}.{fmt.split('_')[0]}"
        tmp.write_bytes(data)
        try:
            return load_audio(str(tmp))
        except Exception:  # noqa: BLE001
            raise TTSError(502, "Could not decode the audio returned by ElevenLabs.", "decode_failed")
        finally:
            tmp.unlink(missing_ok=True)

    def import_upload(self, data: bytes, filename: str) -> dict:
        """Use an existing recording instead of ElevenLabs."""
        key = digest("upload", data)
        wav = self.store.path("speech", key, "wav")
        if not wav.exists():
            tmp = self.store.root / "speech" / f"tmp_{key}_{re.sub(r'[^a-zA-Z0-9.]', '', filename)[-20:] or 'a'}"
            tmp.write_bytes(data)
            try:
                x, sr = load_audio(str(tmp))
            finally:
                tmp.unlink(missing_ok=True)
            sf.write(str(wav), x, sr, subtype="PCM_24")
        info = sf.info(str(wav))
        return {"id": key, "cached": False, "characters": 0, "duration": info.duration,
                "sample_rate": info.samplerate, "demo": False}
