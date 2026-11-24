"""Image-to-Acoustics API.

Flow: voice -> /api/speech (ElevenLabs, cached) -> /api/analyze (image -> room)
      -> /api/render (acoustic engine) -> /api/files/... (compare / download)
"""
from __future__ import annotations
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from acoustics.audio_io import AudioError, load_audio, save_audio
from acoustics.ir import build_impulse_response
from acoustics.models import Dimension, MixControls, Placement, RoomParameters
from acoustics.presets import PRESETS
from acoustics.render import render
from acoustics.theory import analyse_room

from .config import ROOT, Settings, get_settings
from .placement_map import norm_to_room
from .storage import Store, digest
from .tts import TTSError, TTSService
from .vision import VisionError, call_vision, normalise_analysis, prepare_image, preset_analysis

log = logging.getLogger("i2a")
MAX_AUDIO_UPLOAD = 50 * 1024 * 1024


class Point(BaseModel):
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)


class SpeechRequest(BaseModel):
    voice_id: str
    text: str


ID_PATTERN = r"^[a-f0-9]{16,64}$"


class RenderRequest(BaseModel):
    speech_id: str = Field(..., pattern=ID_PATTERN)
    analysis_id: str = Field(..., pattern=ID_PATTERN)
    width: Optional[float] = Field(None, ge=1, le=300)
    length: Optional[float] = Field(None, ge=1, le=300)
    height: Optional[float] = Field(None, ge=1.8, le=80)
    rt60: Optional[float] = Field(None, ge=0.05, le=20)
    absorption_scale: float = Field(1.0, ge=0.25, le=4)
    speaker: Point
    listener: Point
    facing: Literal["towards", "away"] = "towards"
    obstruction: float = Field(0.0, ge=0, le=1)
    distance: Optional[float] = Field(None, ge=0.3, le=150)
    wet: float = Field(1.0, ge=0, le=2)
    reflections: float = Field(1.0, ge=0, le=2)


def predicted(room: RoomParameters) -> dict:
    ac = analyse_room(room)
    return {"volume": ac.volume, "rt60_eyring": ac.rt60_eyring.tolist(), "rt60_sabine": ac.rt60_sabine.tolist(),
            "rt60_used": ac.rt60_used.tolist(), "rt60_source": ac.rt60_source,
            "mean_alpha": ac.mean_alpha.tolist(), "mixing_time": ac.mixing_time}


def create_app(settings: Settings | None = None, tts: TTSService | None = None, vision_transport=None) -> FastAPI:
    s = settings or get_settings()
    store = Store(s.data_dir)
    tts = tts or TTSService(s, store)
    ir_cache: OrderedDict = OrderedDict()
    app = FastAPI(title="Image-to-Acoustics", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(TTSError)
    async def _tts_err(_: Request, e: TTSError):
        return JSONResponse({"error": e.message, "code": e.code}, status_code=e.status)

    @app.exception_handler(VisionError)
    async def _vis_err(_: Request, e: VisionError):
        return JSONResponse({"error": e.message, "code": "vision"}, status_code=e.status)

    @app.exception_handler(AudioError)
    async def _audio_err(_: Request, e: AudioError):
        return JSONResponse({"error": str(e), "code": "audio"}, status_code=415)

    @app.exception_handler(RequestValidationError)
    async def _val_err(_: Request, e: RequestValidationError):
        first = e.errors()[0] if e.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", [])[1:])
        return JSONResponse({"error": f"Invalid request: {where} {first.get('msg', '')}".strip(), "code": "validation"},
                            status_code=422)

    @app.exception_handler(HTTPException)
    async def _http_err(_: Request, e: HTTPException):
        return JSONResponse({"error": e.detail, "code": "http"}, status_code=e.status_code)

    @app.exception_handler(Exception)
    async def _any_err(_: Request, e: Exception):
        log.exception("unhandled error: %s", type(e).__name__)
        return JSONResponse({"error": "Something went wrong on the server. Try again.", "code": "internal"},
                            status_code=500)

    # ------------------------------------------------------------- status
    @app.get("/api/status")
    def status():
        return {"elevenlabs": not tts.demo, "vision": s.vision_enabled,
                "vision_provider": s.vision_provider if s.vision_enabled else "demo",
                "max_chars": s.max_tts_chars, "credits": tts.credits(),
                "presets": [{"id": k, "name": r.name} for k, (r, _) in PRESETS.items()]}

    # -------------------------------------------------------------- voice
    @app.get("/api/voices")
    def voices():
        return {"voices": tts.voices(), "demo": tts.demo}

    @app.post("/api/speech")
    def speech(req: SpeechRequest):
        out = tts.synthesize(req.voice_id, req.text)
        out["url"] = f"/api/files/speech/{out['id']}.wav"
        out["credits"] = tts.credits()
        return out

    @app.post("/api/speech/upload")
    async def speech_upload(file: UploadFile = File(...)):
        data = await file.read()
        if len(data) > MAX_AUDIO_UPLOAD:
            raise HTTPException(413, "Audio file is larger than 50 MB.")
        out = tts.import_upload(data, file.filename or "upload")
        out["url"] = f"/api/files/speech/{out['id']}.wav"
        return out

    # ---------------------------------------------------------- analysis
    @app.post("/api/analyze")
    async def analyze(image: UploadFile = File(...), preset: str = Form("")):
        jpeg, size = prepare_image(await image.read())
        image_id = digest(jpeg)
        store.path("image", image_id, "jpg").write_bytes(jpeg)
        use_ai = s.vision_enabled and not preset
        if not use_ai and not preset:
            raise VisionError(400, "AI analysis isn't configured on this server. Choose the closest environment.")
        aid = digest(image_id, "ai" if use_ai else f"preset:{preset}", s.vision_provider, s.vision_model)
        cached_path = store.path("analysis", aid, "json")
        if cached_path.exists():
            out = json.loads(cached_path.read_text())
            out["cached"] = True
            return out
        t0 = time.time()
        if use_ai:
            result = normalise_analysis(call_vision(s.vision_provider, s.vision_api_key, s.vision_model, jpeg,
                                                    transport=vision_transport))
            result["source"] = "ai"
        else:
            result = preset_analysis(preset)
            result["source"] = "preset"
        room = RoomParameters.model_validate(result["room"])
        result.update(id=aid, image_id=image_id, image_url=f"/api/files/image/{image_id}.jpg",
                      image_size=list(size), predicted=predicted(room), seconds=round(time.time() - t0, 2),
                      provider=s.vision_provider if use_ai else "preset")
        cached_path.write_text(json.dumps(result))
        result["cached"] = False
        return result

    # ------------------------------------------------------------- render
    @app.post("/api/render")
    def do_render(req: RenderRequest):
        apath = store.path("analysis", req.analysis_id, "json")
        spath = store.path("speech", req.speech_id, "wav")
        if not apath.exists():
            raise HTTPException(404, "Analyse an image first.")
        if not spath.exists():
            raise HTTPException(404, "Generate or upload speech first.")
        analysis = json.loads(apath.read_text())
        room = RoomParameters.model_validate(analysis["room"])
        upd = {}
        for k in ("width", "length", "height"):
            v = getattr(req, k)
            if v is not None and abs(v - getattr(room, k).value) > 1e-6:
                upd[k] = Dimension(value=v, confidence=1.0, provenance="user")
        if req.rt60 is not None:
            upd["rt60_override"] = req.rt60
        if req.absorption_scale != 1.0:
            upd["absorption_scale"] = req.absorption_scale
        room = room.model_copy(update=upd)
        dims = room.dims
        placement = Placement(source=norm_to_room(req.speaker.x, req.speaker.y, dims, 1.65),
                              listener=norm_to_room(req.listener.x, req.listener.y, dims, 1.55),
                              source_facing=req.facing, obstruction=req.obstruction,
                              distance_override=req.distance)
        x, fs = load_audio(str(spath))
        mix = MixControls(wet_gain=req.wet, reflection_intensity=req.reflections, output_mode="match")
        ir_key = digest(room.model_dump_json(), placement.model_dump_json(), str(fs))
        rid = digest(req.speech_id, ir_key, mix.model_dump_json())
        out_wav = store.path("render", rid, "wav")
        meta_path = store.path("render", rid, "json")
        if out_wav.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["cached"] = True
            return meta
        t0 = time.time()
        ir = ir_cache.get(ir_key)
        if ir is None:
            ir = build_impulse_response(room, placement, fs=fs, seed=0)
            ir_cache[ir_key] = ir
            while len(ir_cache) > 6:
                ir_cache.popitem(last=False)
        else:
            ir_cache.move_to_end(ir_key)
        res = render(x, fs, room, placement, mix, ir=ir)
        save_audio(str(out_wav), res.audio, fs)
        m = res.measured
        nan = lambda v: None if v is None or v != v else round(float(v), 3)  # noqa: E731
        meta = {
            "id": rid, "url": f"/api/files/render/{rid}.wav", "duration": res.audio.shape[1] / fs,
            "seconds": round(time.time() - t0, 2), "distance": round(res.info["distance"], 2),
            "dims": dims.tolist(), "source": res.info["source"], "listener": res.info["listener"],
            "predicted": predicted(room),
            "measured": {"T30": [nan(v) for v in m["T30"]], "EDT": [nan(v) for v in m["EDT"]],
                         "C50": nan(m["C50_dB"]), "C80": nan(m["C80_dB"]), "DRR": nan(m["DRR_dB"])},
            "critical_distance": res.info["critical_distance"][3], "cached": False,
        }
        meta_path.write_text(json.dumps(meta))
        return meta

    # -------------------------------------------------------------- files
    @app.get("/api/files/{kind}/{name}")
    def files(kind: str, name: str, download: Optional[str] = None):
        if kind not in ("speech", "render", "image") or "." not in name:
            raise HTTPException(404, "Not found")
        fid, ext = name.rsplit(".", 1)
        try:
            if ext == "mp3" and kind in ("speech", "render"):
                p = store.mp3_for(store.path(kind, fid, "wav"))
            elif (kind == "image" and ext == "jpg") or (kind != "image" and ext == "wav"):
                p = store.path(kind, fid, ext)
            else:
                raise ValueError
        except ValueError:
            raise HTTPException(404, "Not found")
        except RuntimeError as e:
            raise HTTPException(500, str(e))
        if not p.exists():
            raise HTTPException(404, "Not found")
        media = {"wav": "audio/wav", "mp3": "audio/mpeg", "jpg": "image/jpeg"}[ext]
        safe = "".join(ch for ch in (download or "") if ch.isalnum() or ch in "-_")[:60]
        return FileResponse(p, media_type=media, filename=f"{safe}.{ext}" if safe else None,
                            headers={"Cache-Control": "public, max-age=31536000, immutable"})

    dist = ROOT / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
    return app


app = create_app()
