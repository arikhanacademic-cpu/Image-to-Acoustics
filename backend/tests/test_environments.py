"""Physical plausibility across the ten reference environments."""
import itertools
import numpy as np
import pytest
from acoustics import PRESETS, build_impulse_response, measure_ir, render
from acoustics.metrics import band_levels_db

FS = 48000


@pytest.fixture(scope="module")
def results():
    out = {}
    for name, (room, pl) in PRESETS.items():
        ir = build_impulse_response(room, pl, fs=FS)
        out[name] = (ir, measure_ir(ir.combine()[0], FS, ir.t_direct, direct=ir.direct[0]))
    return out


def t_mid(m):
    return 0.5 * (m["T30"][2] + m["T30"][3])


def test_reverberation_ordering(results):
    T = {k: t_mid(m) for k, (_, m) in results.items()}
    assert T["cathedral"] > T["royal_throne_room"] > T["school_classroom"] > T["furnished_bedroom"] > T["recording_studio"]
    assert T["empty_warehouse"] > T["school_classroom"]
    assert T["public_bathroom"] > 3 * T["furnished_bedroom"]
    assert T["marble_palace"] > T["royal_throne_room"]  # carpet + tapestries vs bare marble
    assert T["recording_studio"] < 0.25


def test_plausible_ranges(results):
    T = {k: t_mid(m) for k, (_, m) in results.items()}
    ranges = {"cathedral": (5, 12), "furnished_bedroom": (0.2, 0.6), "empty_warehouse": (2.5, 9),
              "marble_palace": (2.5, 9), "underground_cave": (2, 10), "school_classroom": (0.35, 0.9),
              "public_bathroom": (1.0, 3.0), "recording_studio": (0.05, 0.3), "royal_throne_room": (1.2, 3.5)}
    for k, (lo, hi) in ranges.items():
        assert lo <= T[k] <= hi, (k, T[k])


def test_forest_is_direct_dominated(results):
    drr = {k: m["DRR_dB"] for k, (_, m) in results.items()}
    c50 = {k: m["C50_dB"] for k, (_, m) in results.items()}
    assert drr["outdoor_forest"] > 8 and c50["outdoor_forest"] > 15
    assert drr["cathedral"] < 0 and c50["cathedral"] < 2


def test_frequency_dependence(results):
    # carpeted, furnished bedroom: highs die faster than lows
    _, m = results["furnished_bedroom"]
    assert m["T30"][0] > 1.5 * m["T30"][4]
    # large rooms: air absorption shortens 4-8 kHz substantially
    for k in ("cathedral", "empty_warehouse"):
        _, m = results[k]
        assert m["T30"][6] < 0.6 * m["T30"][3]


def test_no_two_environments_identical(results):
    feats = {k: np.array([np.log(t_mid(m)), m["C50_dB"] / 10, m["DRR_dB"] / 10,
                          np.log(m["T30"][0] / m["T30"][5])]) for k, (_, m) in results.items()}
    for a, b in itertools.combinations(feats, 2):
        assert np.linalg.norm(feats[a] - feats[b]) > 0.15, (a, b)


def test_processed_audio_differs_spectrally(voice):
    """Long, air-damped rooms tilt the rendered spectrum towards low frequencies;
    a treated studio leaves the dry spectrum almost unchanged."""
    from acoustics.metrics import band_levels_db
    dry = np.array(band_levels_db(voice, FS))
    tilt = {}
    for k in ("cathedral", "empty_warehouse", "recording_studio"):
        room, pl = PRESETS[k]
        r = render(voice, FS, room, pl)
        wet = np.array(band_levels_db(r.audio.mean(axis=0), FS)) - dry
        tilt[k] = (wet[1] - wet[3]) - (wet[5] - wet[3])   # (250 Hz vs 4 kHz) relative to 1 kHz
    assert abs(tilt["recording_studio"]) < 3
    assert tilt["cathedral"] > tilt["recording_studio"] + 2
    assert tilt["empty_warehouse"] > tilt["recording_studio"] + 2


def test_drr_agrees_with_critical_distance_theory():
    """In diffuse rooms the simulated 1 kHz DRR should follow 20 log10(r_c / r)."""
    from acoustics.metrics import measure_ir
    for name, (room, pl) in PRESETS.items():
        if room.enclosure != "indoor":
            continue  # outdoor: specular ground reflection dominates, theory does not apply
        ir = build_impulse_response(room, pl, fs=FS)
        m = measure_ir(ir.combine()[0], FS, ir.t_direct, direct=ir.direct[0])
        theory = 20 * np.log10(ir.info["critical_distance"][3] / ir.info["distance"])
        assert abs(m["DRR_bands"][3] - theory) < 3.0, (name, m["DRR_bands"][3], theory)
