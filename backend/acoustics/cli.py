"""Command-line renderer (Phase 1: manually supplied room parameters).

Examples
  python -m acoustics.cli list
  python -m acoustics.cli render voice.wav out.wav --preset cathedral
  python -m acoustics.cli render voice.wav out.mp3 --room my_room.json --placement pos.json \
        --wet 0.8 --obstruction 0.5 --facing away --distance 6
  python -m acoustics.cli dump-preset cathedral > my_room.json
"""
from __future__ import annotations
import argparse
import json
import sys
import numpy as np

from .presets import PRESETS
from .models import RoomParameters, Placement, MixControls
from .render import render
from .audio_io import load_audio, save_audio
from .theory import analyse_room


def main(argv=None):
    p = argparse.ArgumentParser(prog="acoustics")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    d = sub.add_parser("dump-preset"); d.add_argument("name")
    r = sub.add_parser("render")
    r.add_argument("input"); r.add_argument("output")
    r.add_argument("--preset"); r.add_argument("--room"); r.add_argument("--placement")
    r.add_argument("--wet", type=float, default=1.0)
    r.add_argument("--reflections", type=float, default=1.0)
    r.add_argument("--rt60", type=float, help="mid-frequency RT60 override (s)")
    r.add_argument("--absorption-scale", type=float)
    r.add_argument("--obstruction", type=float)
    r.add_argument("--facing", help="towards | away | yaw degrees")
    r.add_argument("--distance", type=float)
    r.add_argument("--mode", choices=["peak", "physical"], default="peak")
    r.add_argument("--mono", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    if a.cmd == "list":
        for k, (room, _) in PRESETS.items():
            ac = analyse_room(room)
            print(f"{k:20s} V={room.volume:8.0f} m3  RT60(500-1k, Eyring)={0.5*(ac.rt60_eyring[2]+ac.rt60_eyring[3]):5.2f} s")
        return 0
    if a.cmd == "dump-preset":
        room, pl = PRESETS[a.name]
        print(json.dumps({"room": room.model_dump(), "placement": pl.model_dump()}, indent=2))
        return 0

    if a.preset:
        room, pl = PRESETS[a.preset]
    elif a.room:
        data = json.load(open(a.room))
        room = RoomParameters.model_validate(data.get("room", data))
        pl = Placement.model_validate(data["placement"]) if "placement" in data else None
    else:
        p.error("give --preset or --room")
    if a.placement:
        pl = Placement.model_validate(json.load(open(a.placement)))
    if pl is None:
        W, L, H = room.dims
        pl = Placement(source=(W * .4, L * .3, 1.6), listener=(W * .6, L * .7, 1.6))
    upd = {}
    if a.rt60: room = room.model_copy(update={"rt60_override": a.rt60})
    if a.absorption_scale: room = room.model_copy(update={"absorption_scale": a.absorption_scale})
    if a.obstruction is not None: upd["obstruction"] = a.obstruction
    if a.facing: upd["source_facing"] = a.facing if a.facing in ("towards", "away") else float(a.facing)
    if a.distance: upd["distance_override"] = a.distance
    pl = pl.model_copy(update=upd)
    mix = MixControls(wet_gain=a.wet, reflection_intensity=a.reflections, output_mode=a.mode, stereo=not a.mono)

    x, sr = load_audio(a.input)
    res = render(x, sr, room, pl, mix, seed=a.seed)
    fmt = "mp3" if a.output.lower().endswith(".mp3") else "wav"
    save_audio(a.output, res.audio, sr, fmt=fmt)
    ac = res.info["acoustics"]
    print(json.dumps({
        "output": a.output, "seconds": res.audio.shape[1] / sr, "gain_db": round(res.gain_db, 2),
        "distance_m": round(res.info["distance"], 2),
        "predicted_rt60_eyring_s": [round(v, 2) for v in ac["rt60_eyring"]],
        "measured_T30_s": [round(v, 2) if np.isfinite(v) else None for v in res.measured["T30"]],
        "C50_dB": round(res.measured["C50_dB"], 1), "DRR_dB": round(res.measured["DRR_dB"], 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
