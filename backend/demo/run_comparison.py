"""Reproducible comparison of all reference environments.

    python demo/run_comparison.py [--voice ../samples/voice_dry.wav] [--out ../samples/comparison]

Writes one stereo WAV per environment, each environment's impulse response,
positioning variants for one room, metrics.json, REPORT.md and comparison.png.
Deterministic (fixed noise seed).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from acoustics import PRESETS, render, MixControls  # noqa: E402
from acoustics.audio_io import load_audio, save_audio  # noqa: E402
from acoustics.metrics import schroeder_db, octave_filter, spectral_centroid  # noqa: E402
from acoustics.bands import BANDS_HZ  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def fmt(v, n=2):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=os.path.join(HERE, "..", "..", "samples", "voice_dry.wav"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "samples", "comparison"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    x, fs = load_audio(a.voice)
    rows, irs = [], {}
    for name, (room, pl) in PRESETS.items():
        res = render(x, fs, room, pl, MixControls(), seed=0)
        save_audio(os.path.join(a.out, f"{name}.wav"), res.audio, fs)
        h = res.ir.combine()
        save_audio(os.path.join(a.out, f"ir_{name}.wav"), h[1:] / np.max(np.abs(h[1:])) * 0.9, fs)
        ac, m = res.info["acoustics"], res.measured
        rows.append(dict(name=name, label=room.name, volume=room.volume, distance=res.info["distance"],
                         rt_eyring=ac["rt60_eyring"], rt_sabine=ac["rt60_sabine"], T30=m["T30"], EDT=m["EDT"],
                         C50=m["C50_dB"], C80=m["C80_dB"], DRR=m["DRR_dB"],
                         crit_dist_1k=res.info["critical_distance"][3],
                         centroid_hz=spectral_centroid(res.audio.mean(0), fs),
                         n_images=res.info["n_image_sources"], seconds=res.audio.shape[1] / fs))
        irs[name] = (h[0], res.ir.t_direct)
        print(f"{name:20s} T30(1k)={fmt(m['T30'][3])} s  C50={m['C50_dB']:.1f} dB  DRR={m['DRR_dB']:.1f} dB")

    # positioning variants in the throne room
    room, pl = PRESETS["royal_throne_room"]
    variants = {"facing_away": pl.model_copy(update={"source_facing": "away"}),
                "obstructed": pl.model_copy(update={"obstruction": 1.0}),
                "close_2m": pl.model_copy(update={"distance_override": 2.0}),
                "far_16m": pl.model_copy(update={"distance_override": 16.0})}
    var_rows = []
    for vn, vpl in variants.items():
        res = render(x, fs, room, vpl, MixControls(), seed=0)
        save_audio(os.path.join(a.out, f"royal_throne_room__{vn}.wav"), res.audio, fs)
        var_rows.append((vn, res.info["distance"], res.measured["DRR_dB"], res.measured["C50_dB"],
                         spectral_centroid(res.audio.mean(0), fs)))
    save_audio(os.path.join(a.out, "dry_reference.wav"), np.stack([x, x]) * (0.89 / np.max(np.abs(x))), fs)

    # figure
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    for name, (h, t0) in irs.items():
        hb = octave_filter(h[int(t0 * fs):], fs, 1000)
        edc = schroeder_db(hb)
        t = (np.arange(len(edc)) + 1) / fs
        axs[0].plot(t, edc, label=name, lw=1.2)
    axs[0].set(xlim=(0.003, 12), ylim=(-65, 2), xlabel="time after direct sound (s, log)", ylabel="dB",
               title="Energy decay curves, 1 kHz octave", xscale="log")
    axs[0].legend(fontsize=7, loc="lower left")
    for r in rows:
        axs[1].plot(BANDS_HZ, r["T30"], marker="o", lw=1.2, label=r["name"])
    axs[1].set(xscale="log", yscale="log", xlabel="Hz", ylabel="T30 (s)", title="Measured T30 per octave band")
    from matplotlib.ticker import NullFormatter
    axs[1].xaxis.set_minor_formatter(NullFormatter())
    axs[1].set_xticks(BANDS_HZ, [f"{int(b) if b < 1000 else str(int(b/1000))+'k'}" for b in BANDS_HZ])
    names = [r["name"].replace("_", "\n") for r in rows]
    axs[2].bar(np.arange(len(rows)) - 0.2, [r["C50"] for r in rows], 0.4, label="C50, 500 Hz–1 kHz (dB)")
    axs[2].bar(np.arange(len(rows)) + 0.2, [r["DRR"] for r in rows], 0.4, label="DRR, 500 Hz–1 kHz (dB)")
    axs[2].set_xticks(np.arange(len(rows)), names, fontsize=6.5)
    axs[2].axhline(0, color="k", lw=0.5); axs[2].legend(loc="upper left"); axs[2].set_title("Clarity and direct-to-reverberant ratio")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "comparison.png"), dpi=130)

    json.dump({"environments": rows, "variants": var_rows}, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    L = ["# Environment comparison", "",
         f"Source: `{os.path.basename(a.voice)}` ({len(x)/fs:.1f} s, {fs} Hz). Seed 0. Regenerate with "
         "`python demo/run_comparison.py`.", "",
         "C50/DRR are 500 Hz–1 kHz means (ISO 3382 mid convention). *Predicted* = statistical model (Eyring / Sabine) from the room description. "
         "*Measured* = ISO 3382-style analysis of the simulated impulse response (omni channel).", "",
         "| Environment | V (m³) | r (m) | Predicted T60 Eyring 500/1k (s) | Sabine 1k | Measured T30 125 / 1k / 4k (s) | EDT 1k | C50 mid (dB) | DRR mid (dB) | r_crit 1k (m) | Centroid (Hz) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | {r['volume']:.0f} | {r['distance']:.1f} | {fmt(r['rt_eyring'][2])} / {fmt(r['rt_eyring'][3])} | "
                 f"{fmt(r['rt_sabine'][3])} | {fmt(r['T30'][0])} / {fmt(r['T30'][3])} / {fmt(r['T30'][5])} | {fmt(r['EDT'][3])} | "
                 f"{r['C50']:.1f} | {r['DRR']:.1f} | {r['crit_dist_1k']:.2f} | {r['centroid_hz']:.0f} |")
    L += ["", "## Positioning variants (Royal throne room)", "",
          "| Variant | r (m) | DRR mid (dB) | C50 mid (dB) | Centroid (Hz) |", "|---|---|---|---|---|"]
    for vn, d, drr, c50, cen in var_rows:
        L.append(f"| {vn} | {d:.1f} | {drr:.1f} | {c50:.1f} | {cen:.0f} |")
    L += ["", "![comparison](comparison.png)", ""]
    open(os.path.join(a.out, "REPORT.md"), "w").write("\n".join(L))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
