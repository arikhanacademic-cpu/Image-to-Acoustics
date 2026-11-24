import pytest
import numpy as np
from acoustics.ir import build_impulse_response, voice_directivity_q
from acoustics.metrics import measure_ir, octave_filter
from acoustics.models import Placement
from acoustics.bands import SPEED_OF_SOUND as C
from conftest import make_room


def _peak(h):
    i = int(np.argmax(np.abs(h)))
    return i, h[i]


def test_direct_arrival_time_and_level(anechoic, fs):
    for r in (1.0, 2.0, 4.0):
        pl = Placement(source=(5, 3, 5), listener=(5, 3 + r, 5))
        ir = build_impulse_response(anechoic, pl, fs=fs)
        i, _ = _peak(ir.direct[0])
        assert abs(i - r / C * fs) <= 1.0
        # energy in the 1 kHz octave follows 1/r^2 (air absorption < 0.02 dB here).
        # Sample peaks are not used: a fractional delay's true peak lies between samples.
        e1k = np.sum(octave_filter(ir.direct[0], fs, 1000) ** 2)
        if r == 1.0:
            e_ref = e1k
        assert abs(10 * np.log10(e1k * r ** 2 / e_ref)) < 0.1
    # broadband energy at 1 m ~ 1 (unit on-axis direct sound at 1 m)
    ir = build_impulse_response(anechoic, Placement(source=(5, 3, 5), listener=(5, 4, 5)), fs=fs)
    assert abs(np.sum(ir.direct[0] ** 2) - 1.0) < 0.03


def test_inverse_square_law(anechoic, fs):
    lv = []
    for r in (1.0, 2.0, 4.0):
        pl = Placement(source=(5, 2, 5), listener=(5, 2 + r, 5))
        ir = build_impulse_response(anechoic, pl, fs=fs)
        lv.append(10 * np.log10(np.sum(ir.direct[0] ** 2)))
    assert abs((lv[0] - lv[1]) - 6.02) < 0.3
    assert abs((lv[1] - lv[2]) - 6.02) < 0.3


def test_floor_reflection_timing(fs):
    room = make_room(20, 20, 10, all="open", floor="marble")
    pl = Placement(source=(10, 8, 1.5), listener=(10, 12, 1.5))
    ir = build_impulse_response(room, pl, fs=fs)
    d_img = np.hypot(4.0, 3.0)  # image of source at z = -1.5
    i, _ = _peak(ir.early[0])
    assert abs(i - d_img / C * fs) <= 1.0
    # 1 kHz expectation: marble (alpha .01) -0.04 dB, talker directivity at 37 deg
    # below axis: -6 dB * (1 - 0.8) / 2 = -0.6 dB, air ~ -0.02 dB  => about -0.66 dB
    e_ref = np.sum(octave_filter(build_impulse_response(
        make_room(20, 20, 10, all="open"), pl, fs=fs).direct[0], fs, 1000) ** 2) * 4.0 ** 2
    e_ref_db = 10 * np.log10(e_ref)
    e_refl = np.sum(octave_filter(ir.early[0], fs, 1000) ** 2) * d_img ** 2
    assert abs(10 * np.log10(e_refl) - e_ref_db - (-0.66)) < 0.4


@pytest.mark.parametrize("override", [[0.5] * 7, [1.5] * 7, [4.0] * 7, [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8]])
def test_tail_calibration_matches_target_decay(fs, override):
    """Controlled targets isolate synthesis from octave-integration effects."""
    room = make_room(10, 14, 5, all="plaster").model_copy(update={"rt60_override": override})
    pl = Placement(source=(3, 4, 1.6), listener=(6, 10, 1.6))
    ir = build_impulse_response(room, pl, fs=fs)
    m = measure_ir(ir.combine()[0], fs, ir.t_direct)
    target = ir.info["acoustics"]["rt60_used"]
    for b in range(1, 7):
        assert abs(m["T30"][b] / target[b] - 1) < 0.08, (b, m["T30"][b], target[b])


def test_measured_rt_tracks_model_for_real_materials(fs):
    """With real materials T60 can change by 2x within one octave (e.g. wood
    panelling 500 Hz -> 1 kHz). Octave-band T30 then reflects the longer
    neighbouring decay, as in a real ISO 3382 measurement, so the tolerance is wider."""
    for mat, W, L, H in [("plaster", 6, 8, 3), ("brick", 12, 20, 6), ("wood_panelling", 5, 7, 3)]:
        room = make_room(W, L, H, all=mat)
        pl = Placement(source=(W * .3, L * .3, 1.6), listener=(W * .6, L * .7, 1.6))
        ir = build_impulse_response(room, pl, fs=fs)
        m = measure_ir(ir.combine()[0], fs, ir.t_direct)
        target = ir.info["acoustics"]["rt60_used"]
        for b in (2, 3, 4):
            assert abs(m["T30"][b] / target[b] - 1) < 0.30, (mat, b, m["T30"][b], target[b])


def test_obstruction_muffles_hf_more_than_lf(fs):
    room = make_room(all="plaster")
    base = Placement(source=(2, 2, 1.6), listener=(4, 6, 1.6))
    a = build_impulse_response(room, base, fs=fs).direct[0]
    b = build_impulse_response(room, base.model_copy(update={"obstruction": 1.0}), fs=fs).direct[0]
    lf = 10 * np.log10(np.sum(octave_filter(b, fs, 250) ** 2) / np.sum(octave_filter(a, fs, 250) ** 2))
    hf = 10 * np.log10(np.sum(octave_filter(b, fs, 4000) ** 2) / np.sum(octave_filter(a, fs, 4000) ** 2))
    assert lf < -2 and hf < lf - 8


def test_facing_away_reduces_hf_direct(fs):
    room = make_room(all="plaster")
    tw = Placement(source=(2, 2, 1.6), listener=(4, 6, 1.6), source_facing="towards")
    aw = tw.model_copy(update={"source_facing": "away"})
    a = build_impulse_response(room, tw, fs=fs).direct[0]
    b = build_impulse_response(room, aw, fs=fs).direct[0]
    hf = 10 * np.log10(np.sum(octave_filter(b, fs, 4000) ** 2) / np.sum(octave_filter(a, fs, 4000) ** 2))
    lf = 10 * np.log10(np.sum(octave_filter(b, fs, 250) ** 2) / np.sum(octave_filter(a, fs, 250) ** 2))
    assert hf < -8 and lf > -3


def test_stereo_lateral_source(anechoic, fs):
    # listener faces +y; source on the left (-x side)
    pl = Placement(source=(3, 5, 5), listener=(5, 5, 5), listener_facing=90.0)
    ir = build_impulse_response(anechoic, pl, fs=fs)
    iL, _ = _peak(ir.direct[1]); iR, _ = _peak(ir.direct[2])
    assert iL < iR  # left ear first
    assert 25 <= iR - iL <= 40  # Woodworth ITD ~0.66 ms at 90 deg
    eL = np.sum(octave_filter(ir.direct[1], fs, 4000) ** 2)
    eR = np.sum(octave_filter(ir.direct[2], fs, 4000) ** 2)
    assert 10 * np.log10(eL / eR) > 8


def test_tail_is_deterministic_with_seed(fs):
    room = make_room(all="plaster")
    pl = Placement(source=(2, 2, 1.6), listener=(4, 6, 1.6))
    a = build_impulse_response(room, pl, fs=fs, seed=3).late
    b = build_impulse_response(room, pl, fs=fs, seed=3).late
    assert np.array_equal(a, b)


def test_voice_q_reasonable():
    q = voice_directivity_q()
    assert np.all(np.diff(q) >= 0) and 1.0 <= q[0] < 1.2 and 2.0 < q[-1] < 6.0


def test_distance_override(fs):
    room = make_room(20, 20, 5, all="plaster")
    pl = Placement(source=(5, 5, 1.6), listener=(10, 10, 1.6), distance_override=3.0)
    ir = build_impulse_response(room, pl, fs=fs)
    assert abs(ir.info["distance"] - 3.0) < 1e-6
