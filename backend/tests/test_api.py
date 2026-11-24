"""API integration tests (no network: ElevenLabs client and vision HTTP are mocked)."""
import io
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from fastapi.testclient import TestClient
from elevenlabs.core.api_error import ApiError

from app.config import Settings
from app.main import create_app
from app.storage import Store
from app.tts import TTSService

SECRET = "sk_test_SUPER_SECRET_123"


def png_bytes(size=(640, 480)):
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 80, 70)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def mp3_clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("mp3")
    t = np.arange(44100) / 44100
    sf.write(d / "a.wav", 0.3 * np.sin(2 * np.pi * 220 * t) * (t < 0.8), 44100)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "a.wav"), str(d / "a.mp3")], check=True)
    return (d / "a.mp3").read_bytes()


class FakeEleven:
    def __init__(self, audio: bytes, remaining=10_000, error=None):
        self.calls = 0
        self.error = error
        fake = self

        def convert(**kw):
            fake.calls += 1
            fake.last = kw
            if fake.error:
                raise fake.error
            return iter([audio[:1000], audio[1000:]])
        self.text_to_speech = SimpleNamespace(convert=convert)
        self.voices = SimpleNamespace(search=lambda **kw: SimpleNamespace(voices=[
            SimpleNamespace(voice_id="v2", name="Zed", category="cloned", description="", labels={}, preview_url=None),
            SimpleNamespace(voice_id="v1", name="Aria", category="premade", description="warm",
                            labels={"accent": "british"}, preview_url="https://example.com/p.mp3")]))
        self.user = SimpleNamespace(subscription=SimpleNamespace(get=lambda: SimpleNamespace(
            character_count=100_000 - remaining, character_limit=100_000, next_character_count_reset_unix=0)))


def make_client(tmp_path, fake=None, vision_handler=None, **over):
    s = replace(Settings(), elevenlabs_api_key=SECRET if fake else "", data_dir=tmp_path,
                vision_provider="anthropic" if vision_handler else "demo",
                vision_api_key="vk_secret" if vision_handler else "", **over)
    tts = TTSService(s, Store(tmp_path), client_factory=(lambda: fake) if fake else None)
    transport = httpx.MockTransport(vision_handler) if vision_handler else None
    return TestClient(create_app(s, tts, vision_transport=transport))


# ------------------------------------------------------------------ voice
def test_status_and_voices_never_leak_key(tmp_path, mp3_clip):
    c = make_client(tmp_path, FakeEleven(mp3_clip))
    for path in ("/api/status", "/api/voices"):
        r = c.get(path)
        assert r.status_code == 200 and SECRET not in r.text
    v = c.get("/api/voices").json()["voices"]
    assert [x["id"] for x in v] == ["v1", "v2"]  # premade first


def test_speech_is_generated_once_then_cached(tmp_path, mp3_clip):
    fake = FakeEleven(mp3_clip)
    c = make_client(tmp_path, fake)
    a = c.post("/api/speech", json={"voice_id": "v1", "text": "Hello hall"}).json()
    b = c.post("/api/speech", json={"voice_id": "v1", "text": "Hello hall"}).json()
    assert fake.calls == 1 and not a["cached"] and b["cached"] and a["id"] == b["id"]
    assert fake.last["model_id"] and fake.last["output_format"].startswith("mp3")
    wav = c.get(a["url"])
    x, sr = sf.read(io.BytesIO(wav.content))
    assert sr == 44100 and np.max(np.abs(x)) > 0.1


def test_pcm_output_format(tmp_path):
    pcm = (0.25 * 32767 * np.sin(np.arange(4410) / 10)).astype("<i2").tobytes()
    c = make_client(tmp_path, FakeEleven(pcm), elevenlabs_output_format="pcm_44100")
    r = c.post("/api/speech", json={"voice_id": "v1", "text": "x"}).json()
    assert abs(r["duration"] - 0.1) < 1e-3


@pytest.mark.parametrize("text,voice,status", [("", "v1", 400), ("a" * 601, "v1", 400), ("ok", "../etc", 400)])
def test_speech_input_validation(tmp_path, mp3_clip, text, voice, status):
    fake = FakeEleven(mp3_clip)
    r = make_client(tmp_path, fake).post("/api/speech", json={"voice_id": voice, "text": text})
    assert r.status_code == status and fake.calls == 0 and r.json()["error"]


def test_insufficient_credits_blocks_before_calling_api(tmp_path, mp3_clip):
    fake = FakeEleven(mp3_clip, remaining=5)
    r = make_client(tmp_path, fake).post("/api/speech", json={"voice_id": "v1", "text": "more than five"})
    assert r.status_code == 402 and fake.calls == 0


@pytest.mark.parametrize("err,status,needle", [
    (ApiError(status_code=401, body={"detail": {"status": "quota_exceeded", "message": "q"}}), 402, "quota"),
    (ApiError(status_code=401, body={"detail": {"status": "invalid_api_key"}}), 502, "ELEVENLABS_API_KEY"),
    (ApiError(status_code=429, body={}), 429, "rate"),
    (ApiError(status_code=422, body={"detail": "voice_not_found"}), 400, "voice_not_found"),
    (ApiError(status_code=500, body=None), 502, "500"),
    (httpx.ConnectError("boom"), 503, "reach"),
])
def test_elevenlabs_errors_are_mapped(tmp_path, mp3_clip, err, status, needle):
    c = make_client(tmp_path, FakeEleven(mp3_clip, error=err))
    r = c.post("/api/speech", json={"voice_id": "v1", "text": "hi"})
    assert r.status_code == status and needle in r.json()["error"] and SECRET not in r.text


def test_demo_mode_works_without_key(tmp_path):
    c = make_client(tmp_path)
    assert c.get("/api/voices").json()["demo"] is True
    r = c.post("/api/speech", json={"voice_id": "demo-narrator", "text": "anything"})
    assert r.status_code == 200 and r.json()["demo"]


def test_audio_upload_and_corrupt_upload(tmp_path):
    c = make_client(tmp_path)
    buf = io.BytesIO(); sf.write(buf, 0.2 * np.random.default_rng(0).standard_normal(8000), 16000, format="WAV")
    assert c.post("/api/speech/upload", files={"file": ("v.wav", buf.getvalue())}).status_code == 200
    r = c.post("/api/speech/upload", files={"file": ("v.wav", b"not audio at all" * 50)})
    assert r.status_code == 415


# ---------------------------------------------------------------- vision
GOOD_AI = {
    "scene": "Stone chapel with wooden pews", "style": "photo", "enclosure": "indoor",
    "dimensions_m": {"width": {"value": 9, "confidence": 0.5}, "length": {"value": 999, "confidence": 0.3},
                     "height": {"value": 7, "confidence": 0.4}},
    "surfaces": {"floor": {"materials": [{"material": "stone", "fraction": 1}], "confidence": 0.7},
                 "ceiling": {"materials": [{"material": "wood", "fraction": 1}]},
                 "left_wall": {"materials": [{"material": "stone_masonry", "fraction": 0.8},
                                             {"material": "velvet drapes", "fraction": 0.5}]},
                 "right_wall": {"materials": [{"material": "glass_window", "fraction": 1}]},
                 "near_wall": {"materials": [{"material": "unobtainium", "fraction": 1}]}},
    "objects": [{"kind": "pews", "count": 40}, {"kind": "candles", "count": 12}],
    "clutter_scattering": 0.3, "speaker": {"x": 0.5, "y": 0.5}, "listener": {"x": 0.5, "y": 0.95},
    "obstruction_between": 0, "assumptions": ["door height used for scale"], "overall_confidence": 0.6}


def anthropic_handler(payload_text, calls):
    def h(req: httpx.Request):
        calls.append(req)
        assert req.headers["x-api-key"] == "vk_secret"
        return httpx.Response(200, json={"content": [{"type": "text", "text": payload_text}]})
    return h


def test_ai_analysis_is_normalised_and_cached(tmp_path):
    calls = []
    c = make_client(tmp_path, vision_handler=anthropic_handler("```json\n" + json.dumps(GOOD_AI) + "\n```", calls))
    a = c.post("/api/analyze", files={"image": ("a.png", png_bytes())}).json()
    b = c.post("/api/analyze", files={"image": ("a.png", png_bytes())}).json()
    assert len(calls) == 1 and b["cached"] and a["id"] == b["id"]
    room = a["room"]
    assert room["length"]["value"] == 300 and room["length"]["provenance"] == "ai_estimate"
    assert room["surfaces"]["wall_x0"]["materials"][1]["material"] == "curtains_heavy"
    assert abs(sum(m["fraction"] for m in room["surfaces"]["wall_x0"]["materials"]) - 1) < 1e-6
    assert room["surfaces"]["wall_y0"]["materials"][0]["material"] == "plaster"   # unknown -> fallback
    assert room["surfaces"]["wall_y1"]["materials"][0]["material"] == "plaster"   # missing -> fallback
    assert [o["kind"] for o in room["objects"]] == ["pew_wooden"]
    assert len(a["corrections"]) >= 5 and a["predicted"]["rt60_eyring"]
    assert "vk_secret" not in json.dumps(a)


@pytest.mark.parametrize("status,code", [(401, 502), (429, 429), (500, 502)])
def test_vision_http_errors(tmp_path, status, code):
    c = make_client(tmp_path, vision_handler=lambda req: httpx.Response(status, json={}))
    r = c.post("/api/analyze", files={"image": ("a.png", png_bytes())})
    assert r.status_code == code and "vk_secret" not in r.text


def test_vision_garbage_response(tmp_path):
    c = make_client(tmp_path, vision_handler=anthropic_handler("I cannot see a room here.", []))
    assert c.post("/api/analyze", files={"image": ("a.png", png_bytes())}).status_code == 502


def test_image_validation(tmp_path):
    c = make_client(tmp_path)
    assert c.post("/api/analyze", files={"image": ("a.png", b"GIF89a nonsense")}, data={"preset": "cathedral"}).status_code == 415
    assert c.post("/api/analyze", files={"image": ("a.png", png_bytes((20, 20)))}, data={"preset": "cathedral"}).status_code == 415
    assert c.post("/api/analyze", files={"image": ("a.png", png_bytes())}).status_code == 400  # demo needs preset
    assert c.post("/api/analyze", files={"image": ("a.png", png_bytes())}, data={"preset": "nope"}).status_code == 400


# ---------------------------------------------------------------- render
def test_render_end_to_end_cached_and_downloadable(tmp_path):
    c = make_client(tmp_path)
    sp = c.post("/api/speech", json={"voice_id": "demo-narrator", "text": "x"}).json()
    an = c.post("/api/analyze", files={"image": ("a.png", png_bytes())}, data={"preset": "public_bathroom"}).json()
    body = {"speech_id": sp["id"], "analysis_id": an["id"], "speaker": an["speaker"], "listener": an["listener"]}
    r1 = c.post("/api/render", json=body).json()
    r2 = c.post("/api/render", json=body).json()
    assert not r1["cached"] and r2["cached"] and r1["measured"]["T30"][3] > 1.0
    wav = c.get(r1["url"]); x, sr = sf.read(io.BytesIO(wav.content))
    assert x.ndim == 2 and np.max(np.abs(x)) <= 10 ** (-1 / 20) + 1e-3 and len(x) / sr > sp["duration"]
    mp3 = c.get(r1["url"].replace(".wav", ".mp3") + "?download=bathroom take")
    assert mp3.status_code == 200 and "bathroom" in mp3.headers["content-disposition"]
    # edits change the result
    r3 = c.post("/api/render", json={**body, "facing": "away", "obstruction": 0.8, "rt60": 0.6}).json()
    assert r3["id"] != r1["id"] and r3["predicted"]["rt60_source"] == "user_override"


def test_render_requires_inputs_and_rejects_bad_ids(tmp_path):
    c = make_client(tmp_path)
    body = {"speech_id": "a" * 32, "analysis_id": "b" * 32, "speaker": {"x": .5, "y": .5}, "listener": {"x": .5, "y": .9}}
    assert c.post("/api/render", json=body).status_code == 404
    r = c.post("/api/render", json={**body, "analysis_id": "../../etc"})
    assert r.status_code == 422 and "analysis_id" in r.json()["error"]
    for bad in ("/api/files/speech/..%2F..%2Fsecret.wav", "/api/files/speech/abc.wav", "/api/files/other/" + "a" * 32 + ".wav",
                "/api/files/image/" + "a" * 32 + ".exe"):
        assert c.get(bad).status_code == 404
