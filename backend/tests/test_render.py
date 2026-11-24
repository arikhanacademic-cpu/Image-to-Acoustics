import numpy as np
from acoustics.render import render, convolve
from acoustics.models import Placement, MixControls
from conftest import make_room


def test_convolution_with_dirac_is_identity(voice):
    h = np.zeros(100); h[0] = 1.0
    y = convolve(voice, h)
    assert len(y) == len(voice) + 99
    assert np.allclose(y[:len(voice)], voice, atol=1e-9)


def test_convolution_matches_direct_definition():
    rng = np.random.default_rng(0)
    x, h = rng.standard_normal(500), rng.standard_normal(80)
    assert np.allclose(convolve(x, h), np.convolve(x, h), atol=1e-9)


def test_no_clipping_even_for_hot_input(fs):
    room = make_room(all="ceramic_tile")
    pl = Placement(source=(2, 2, 1.6), listener=(2.5, 2.5, 1.6))
    x = np.sign(np.sin(2 * np.pi * 200 * np.arange(fs) / fs))  # full-scale square wave
    for mode in ("peak", "physical"):
        r = render(x, fs, room, pl, MixControls(output_mode=mode, ceiling_dbfs=-1.0))
        assert np.max(np.abs(r.audio)) <= 10 ** (-1 / 20) + 1e-9


def test_original_untouched_and_tail_preserved(voice, fs):
    room = make_room(12, 20, 8, all="stone_masonry")
    pl = Placement(source=(4, 5, 1.6), listener=(8, 12, 1.6))
    before = voice.copy()
    r = render(voice, fs, room, pl)
    assert np.array_equal(voice, before)
    rt = max(r.info["acoustics"]["rt60_used"][2:4])
    assert r.audio.shape[1] - len(voice) > 0.8 * rt * fs  # tail kept (to ~-60 dB at least)


def test_physical_mode_distance_cue(voice, fs):
    room = make_room(20, 30, 6, all="plaster")
    near = render(voice, fs, room, Placement(source=(10, 10, 1.6), listener=(10, 11, 1.6)),
                  MixControls(output_mode="physical", ceiling_dbfs=0))
    far = render(voice, fs, room, Placement(source=(10, 10, 1.6), listener=(10, 22, 1.6)),
                 MixControls(output_mode="physical", ceiling_dbfs=0))
    rms = lambda a: np.sqrt(np.mean(a ** 2))
    assert 20 * np.log10(rms(far.audio) / rms(near.audio)) < -4


def test_wet_gain_scales_only_reflected_sound(voice, fs):
    room = make_room(all="plaster")
    pl = Placement(source=(2, 2, 1.6), listener=(4, 6, 1.6))
    ir = render(voice, fs, room, pl).ir
    full, half = ir.combine(1.0), ir.combine(0.5)
    assert np.allclose(full[0] - ir.direct[0], 2 * (half[0] - ir.direct[0]))
    e_ref = lambda h: np.sum((h[0] - ir.direct[0]) ** 2)
    assert abs(10 * np.log10(e_ref(full) / e_ref(half)) - 6.02) < 0.01


def test_match_mode_matches_loudness_without_clipping(voice, fs):
    from acoustics.render import active_rms
    room = make_room(10, 14, 5, all="plaster")
    pl = Placement(source=(3, 4, 1.6), listener=(6, 10, 1.6))
    r = render(voice * 0.3, fs, room, pl, MixControls(output_mode="match"))
    assert np.max(np.abs(r.audio)) <= 10 ** (-1 / 20) + 1e-9
    assert abs(20 * np.log10(active_rms(r.audio, fs) / active_rms(voice * 0.3, fs))) < 0.5
