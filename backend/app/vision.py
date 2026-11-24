"""AI image analysis -> RoomParameters.

Providers: "anthropic" (Claude Messages API), "openai" (Chat Completions), or
"demo" (no AI; the user picks a reference environment). The model is asked for
physical properties with confidences, never for a reverb preset. Its answer
is validated and normalised by `normalise_analysis`, and every substitution or
clamp is recorded in `corrections`, so nothing is silently invented.
"""
from __future__ import annotations
import base64
import io
import json
import re
from typing import Any

import httpx
from PIL import Image

from acoustics.materials import MATERIALS, OBJECT_ABSORPTION
from acoustics.models import RoomParameters, Surface, MaterialShare, AbsorberObject, Dimension
from acoustics.presets import PRESETS
from .placement_map import room_to_norm

MAX_IMAGE_BYTES = 12 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class VisionError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


# ----------------------------------------------------------------- images
def prepare_image(data: bytes) -> tuple[bytes, tuple[int, int]]:
    """Validate an upload and return a JPEG (long edge <= 1568 px)."""
    if len(data) > MAX_IMAGE_BYTES:
        raise VisionError(413, "Image is larger than 12 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()
    except Exception:  # noqa: BLE001
        raise VisionError(415, "That file isn't a readable image. Use a JPEG, PNG or WebP.")
    if fmt not in ALLOWED_FORMATS:
        raise VisionError(415, f"{fmt or 'This'} images aren't supported. Use a JPEG, PNG or WebP.")
    if min(img.size) < 64:
        raise VisionError(415, "Image is too small to analyse (minimum 64 px).")
    img = img.convert("RGB")
    img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue(), img.size


# ------------------------------------------------------------------ prompt
SURFACE_KEYS = {"floor": "floor", "ceiling": "ceiling", "left_wall": "wall_x0", "right_wall": "wall_x1",
                "near_wall": "wall_y0", "far_wall": "wall_y1"}

PROMPT = f"""You are an architectural acoustician. Estimate the physical acoustic properties of the
environment in this image so a room-acoustics simulator can model it. The image may be a photograph
or a stylised / video-game scene; reason about the space it depicts.

Model the space as a box. Axes: width = left-right across the image, length = depth away from the
camera, height = floor to ceiling. "near_wall" is behind the camera, "far_wall" is the one facing it.
For outdoor scenes use the "open" material for the sky (ceiling) and for missing walls, and size the
box to the region that still returns reflections (trees, cliffs, buildings).

Use ONLY these surface materials: {", ".join(sorted(MATERIALS))}.
Use ONLY these object kinds (absorbing furnishings): {", ".join(sorted(OBJECT_ABSORPTION))}.
"tapestry" counts square metres; "pew_wooden" counts seat places; other kinds count items.

Estimate scale from cues (doors ~2.0 m, people ~1.7 m, chairs, tiles, windows). Do not pretend to
precision: give a confidence 0-1 for each estimate and list your assumptions.

Return ONLY a JSON object, no prose, with exactly this shape:
{{
  "scene": "one sentence describing the space",
  "style": "photo" | "stylised",
  "enclosure": "indoor" | "partially_enclosed" | "outdoor",
  "dimensions_m": {{"width": {{"value": n, "confidence": n}}, "length": {{...}}, "height": {{...}}}},
  "surfaces": {{
    "floor": {{"materials": [{{"material": "...", "fraction": n}}], "confidence": n}},
    "ceiling": {{...}}, "left_wall": {{...}}, "right_wall": {{...}}, "near_wall": {{...}}, "far_wall": {{...}}
  }},
  "objects": [{{"kind": "...", "count": n, "confidence": n}}],
  "clutter_scattering": n,      // 0 bare & flat .. 1 very irregular / cluttered (columns, rock, foliage)
  "openings": ["doors, windows, arches that leak sound"],
  "speaker": {{"x": n, "y": n}},  // most natural talker position, normalised image coords (0,0 = top-left)
  "listener": {{"x": n, "y": n}}, // natural listener / camera-near position, normalised image coords
  "obstruction_between": n,     // 0 clear line of sight .. 1 solid barrier between them
  "assumptions": ["..."],
  "overall_confidence": n
}}
Fractions on each surface must sum to 1."""


# ----------------------------------------------------------- normalisation
SYNONYMS = [
    (r"marble", "marble"), (r"granite|polished stone", "marble"), (r"tile", "ceramic_tile"),
    (r"porcelain|enamel", "ceramic_tile"), (r"cave|natural rock|^rock", "rough_rock"),
    (r"stone|masonry|cobble", "stone_masonry"), (r"brick", "brick"),
    (r"bare concrete|polished concrete|^concrete$", "concrete_smooth"), (r"concrete|cement|block", "concrete_painted"),
    (r"plaster|stucco|painted wall", "plaster"), (r"drywall|gypsum|plasterboard", "gypsum_board"),
    (r"parquet|hardwood|wood floor|floorboard|laminate", "wood_floor"), (r"wood|timber|panel", "wood_panelling"),
    (r"linoleum|vinyl|rubber", "linoleum"), (r"metal|steel|corrugated|iron|alumin", "metal_sheet"),
    (r"glass|window|mirror", "glass_window"), (r"rug|thin carpet|carpet tile", "carpet_thin"),
    (r"carpet", "carpet_thick"), (r"curtain|drape|velvet|fabric|textile", "curtains_heavy"),
    (r"acoustic tile|ceiling tile|suspended", "acoustic_tile"),
    (r"foam|acoustic|absorber|treatment|studio", "acoustic_treatment"),
    (r"grass|soil|dirt|earth|lawn|sand", "grass_soil"), (r"leaf|forest floor|moss|litter", "forest_floor"),
    (r"foliage|tree|bush|hedge|canopy|jungle", "foliage_boundary"), (r"water|pool|lake", "water"),
    (r"sky|open|none|air|nothing", "open"),
]
SURFACE_FALLBACK = {"floor": "concrete_painted", "ceiling": "plaster", "left_wall": "plaster",
                    "right_wall": "plaster", "near_wall": "plaster", "far_wall": "plaster"}
OBJECT_SYNONYMS = [(r"bed|mattress", "bed"), (r"sofa|couch", "sofa"), (r"armchair|upholster|cushion", "upholstered_chair"),
                   (r"chair|stool|bench", "wooden_chair"), (r"desk|table", "desk"), (r"book", "bookshelf_full"),
                   (r"wardrobe|cabinet|cupboard|closet", "wardrobe"), (r"person|people|human|crowd", "person"),
                   (r"pew", "pew_wooden"), (r"tapestr|banner|hanging", "tapestry"), (r"rack|shel|pallet", "pallet_rack")]


def _num(v, default, lo, hi, what, corrections):
    try:
        x = float(v)
        if x != x:
            raise ValueError
    except (TypeError, ValueError):
        corrections.append(f"{what}: missing or invalid, used {default}")
        return default
    if not lo <= x <= hi:
        y = min(max(x, lo), hi)
        corrections.append(f"{what}: {x:g} outside {lo:g}–{hi:g}, clamped to {y:g}")
        return y
    return x


def _material(name, surface_key, corrections) -> str:
    n = str(name or "").strip().lower().replace(" ", "_")
    if n in MATERIALS:
        return n
    raw = n.replace("_", " ")
    for pat, mat in SYNONYMS:
        if re.search(pat, raw):
            if surface_key == "floor" and mat == "wood_panelling":
                mat = "wood_floor"
            corrections.append(f"{surface_key}: '{name}' mapped to {mat}")
            return mat
    fb = SURFACE_FALLBACK[surface_key]
    corrections.append(f"{surface_key}: unknown material '{name}', assumed {fb}")
    return fb


def extract_json(text: str) -> dict:
    t = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise VisionError(502, "The vision model didn't return a JSON analysis. Try again or use another image.")
    t = re.sub(r"//[^\n]*", "", t[start:end + 1])
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        raise VisionError(502, "The vision model returned malformed JSON. Try again.")


def normalise_analysis(raw: dict) -> dict:
    """Validate a model response and convert it to engine parameters."""
    if not isinstance(raw, dict):
        raise VisionError(502, "Vision analysis has the wrong shape.")
    c: list[str] = []
    enclosure = raw.get("enclosure")
    if enclosure not in ("indoor", "partially_enclosed", "outdoor"):
        c.append(f"enclosure '{enclosure}' not recognised, assumed indoor")
        enclosure = "indoor"
    dims_raw = raw.get("dimensions_m") or {}
    limits = {"width": (1.0, 200.0, 6.0), "length": (1.0, 300.0, 8.0), "height": (1.8, 80.0, 3.0)}
    dims = {}
    for k, (lo, hi, d) in limits.items():
        v = dims_raw.get(k) or {}
        v = v if isinstance(v, dict) else {"value": v}
        dims[k] = Dimension(value=_num(v.get("value"), d, lo, hi, f"{k}", c),
                            confidence=_num(v.get("confidence", 0.4), 0.4, 0, 1, f"{k} confidence", []),
                            provenance="ai_estimate")
    surfaces = {}
    sraw = raw.get("surfaces") or {}
    for key, internal in SURFACE_KEYS.items():
        s = sraw.get(key) or {}
        mats = s.get("materials") if isinstance(s, dict) else None
        if not mats:
            fb = "open" if (enclosure == "outdoor" and key != "floor") else SURFACE_FALLBACK[key]
            c.append(f"{key}: no materials given, assumed {fb}")
            mats = [{"material": fb, "fraction": 1}]
        merged: dict[str, float] = {}
        for m in mats[:6]:
            if not isinstance(m, dict):
                continue
            mat = _material(m.get("material"), key, c)
            merged[mat] = merged.get(mat, 0.0) + max(_num(m.get("fraction", 1), 1, 0, 1, f"{key} fraction", c), 0.0)
        total = sum(merged.values())
        if total <= 0:
            merged, total = {SURFACE_FALLBACK[key]: 1.0}, 1.0
        if abs(total - 1) > 0.02:
            c.append(f"{key}: fractions summed to {total:.2f}, rescaled")
        shares = [MaterialShare(material=k, fraction=v / total) for k, v in merged.items() if v / total > 1e-3]
        tot2 = sum(sh.fraction for sh in shares)
        for sh in shares:
            sh.fraction /= tot2
        surfaces[internal] = Surface(materials=shares, provenance="ai_estimate",
                                     confidence=_num(s.get("confidence", 0.4) if isinstance(s, dict) else 0.4,
                                                     0.4, 0, 1, "", []))
    objects = []
    for o in (raw.get("objects") or [])[:20]:
        if not isinstance(o, dict):
            continue
        kind = str(o.get("kind", "")).lower().strip().replace(" ", "_")
        if kind not in OBJECT_ABSORPTION:
            for pat, k in OBJECT_SYNONYMS:
                if re.search(pat, kind.replace("_", " ")):
                    c.append(f"object '{kind}' mapped to {k}")
                    kind = k
                    break
            else:
                c.append(f"object '{kind}' ignored (no absorption data)")
                continue
        objects.append(AbsorberObject(kind=kind, count=_num(o.get("count", 1), 1, 0, 2000, f"{kind} count", c),
                                      confidence=_num(o.get("confidence", 0.4), 0.4, 0, 1, "", []),
                                      provenance="ai_estimate"))
    room = RoomParameters(
        name=str(raw.get("scene") or "Analysed environment")[:120], enclosure=enclosure,
        width=dims["width"], length=dims["length"], height=dims["height"], surfaces=surfaces, objects=objects,
        clutter_scattering=_num(raw.get("clutter_scattering", 0.2), 0.2, 0, 1, "clutter_scattering", c),
        assumptions=[str(a)[:200] for a in (raw.get("assumptions") or [])][:10])

    def pos(key, default):
        p = raw.get(key) or {}
        return {"x": _num(p.get("x") if isinstance(p, dict) else None, default[0], 0, 1, f"{key}.x", c),
                "y": _num(p.get("y") if isinstance(p, dict) else None, default[1], 0, 1, f"{key}.y", c)}

    return {
        "scene": room.name, "style": raw.get("style") if raw.get("style") in ("photo", "stylised") else "photo",
        "enclosure": enclosure, "room": room.model_dump(),
        "speaker": pos("speaker", (0.45, 0.62)), "listener": pos("listener", (0.55, 0.9)),
        "obstruction": _num(raw.get("obstruction_between", 0), 0, 0, 1, "obstruction_between", c),
        "openings": [str(o)[:120] for o in (raw.get("openings") or [])][:8],
        "overall_confidence": _num(raw.get("overall_confidence", 0.5), 0.5, 0, 1, "overall_confidence", c),
        "corrections": c,
    }


def preset_analysis(name: str) -> dict:
    if name not in PRESETS:
        raise VisionError(400, f"Unknown environment '{name}'.")
    room, pl = PRESETS[name]
    return {"scene": room.name, "style": "photo", "enclosure": room.enclosure, "room": room.model_dump(),
            "speaker": room_to_norm(pl.source[0], pl.source[1], room.dims),
            "listener": room_to_norm(pl.listener[0], pl.listener[1], room.dims),
            "obstruction": pl.obstruction, "openings": [], "overall_confidence": 1.0,
            "corrections": [], "preset": name}


# --------------------------------------------------------------- providers
def _http_error(provider: str, r: httpx.Response) -> VisionError:
    if r.status_code in (401, 403):
        return VisionError(502, f"The {provider} vision API rejected the server's key. Check VISION_API_KEY.")
    if r.status_code == 429:
        return VisionError(429, f"The {provider} vision API is rate limiting. Try again in a moment.")
    if r.status_code == 400:
        return VisionError(502, f"The {provider} vision API rejected the request (check VISION_MODEL).")
    return VisionError(502, f"The {provider} vision API returned an error ({r.status_code}).")


def call_vision(provider: str, api_key: str, model: str, jpeg: bytes, transport=None) -> dict:
    b64 = base64.b64encode(jpeg).decode()
    try:
        with httpx.Client(timeout=90, transport=transport) as http:
            if provider == "anthropic":
                r = http.post("https://api.anthropic.com/v1/messages",
                              headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                              json={"model": model or "claude-sonnet-5-5", "max_tokens": 2500,
                                    "messages": [{"role": "user", "content": [
                                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                                        {"type": "text", "text": PROMPT}]}]})
                if r.status_code != 200:
                    raise _http_error("Anthropic", r)
                text = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
            elif provider == "openai":
                r = http.post("https://api.openai.com/v1/chat/completions",
                              headers={"Authorization": f"Bearer {api_key}"},
                              json={"model": model or "gpt-4o", "response_format": {"type": "json_object"},
                                    "messages": [{"role": "user", "content": [
                                        {"type": "text", "text": PROMPT},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]})
                if r.status_code != 200:
                    raise _http_error("OpenAI", r)
                text = r.json()["choices"][0]["message"]["content"]
            else:
                raise VisionError(500, f"Unknown vision provider '{provider}'.")
    except httpx.TimeoutException:
        raise VisionError(504, "The vision API timed out. Try again.")
    except httpx.HTTPError:
        raise VisionError(503, "Could not reach the vision API. Check the server's internet connection.")
    return extract_json(text)
