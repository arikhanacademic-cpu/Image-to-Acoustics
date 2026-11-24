import os
import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError
from acoustics.models import Surface, RoomParameters, Dimension, Placement
from acoustics.audio_io import load_audio, save_audio, AudioError
from conftest import make_room


def test_material_fractions_must_sum_to_one():
    with pytest.raises(ValidationError):
        Surface.of(("plaster", 0.5), ("glass_window", 0.2))


def test_unknown_material_rejected():
    with pytest.raises(ValidationError):
        Surface.of("unobtainium")


def test_bad_dimensions_rejected():
    with pytest.raises(ValidationError):
        Dimension(value=-3)


def test_missing_surface_rejected():
    room = make_room()
    data = room.model_dump()
    del data["surfaces"]["ceiling"]
    with pytest.raises(ValidationError):
        RoomParameters(**data)


def test_rt60_override_validation():
    with pytest.raises(ValidationError):
        make_room().model_copy(update={}).__class__(**{**make_room().model_dump(), "rt60_override": [1, 2]})


def test_room_json_roundtrip():
    room = make_room()
    assert RoomParameters.model_validate_json(room.model_dump_json()) == room


def test_placement_clamped_inside_room():
    src, lst, *_ = Placement(source=(-5, 50, 1), listener=(2, 2, 99)).resolved(np.array([6, 8, 3]))
    assert np.all(src >= 0) and np.all(src <= [6, 8, 3]) and np.all(lst <= [6, 8, 3])


def test_corrupted_audio_rejected(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"RIFF\x00\x00garbage-not-audio" * 10)
    with pytest.raises(AudioError):
        load_audio(str(p))


def test_silent_audio_rejected(tmp_path):
    p = tmp_path / "silent.wav"
    sf.write(p, np.zeros(4800), 48000)
    with pytest.raises(AudioError):
        load_audio(str(p))


def test_stereo_input_mixed_to_mono_and_saved(tmp_path):
    p = tmp_path / "st.wav"
    sf.write(p, np.stack([np.ones(100) * .5, np.zeros(100)], 1), 48000)
    x, sr = load_audio(str(p))
    assert x.ndim == 1 and np.allclose(x, .25, atol=1e-3)
    save_audio(str(tmp_path / "o.wav"), np.stack([x, x]), sr)
    save_audio(str(tmp_path / "o.mp3"), np.stack([x, x]), sr, fmt="mp3")
    assert os.path.getsize(tmp_path / "o.mp3") > 0
