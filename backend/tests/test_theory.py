import numpy as np
from acoustics.theory import sabine_rt60, eyring_rt60, critical_distance, analyse_room, K_SABINE
from conftest import make_room


def test_sabine_constant():
    assert abs(K_SABINE - 0.161) < 0.001


def test_sabine_hand_calculation():
    # 10 x 8 x 3 room, every surface alpha = 0.1, no air: S = 268, V = 240
    V, S = 240.0, 2 * (80 + 30 + 24)
    assert np.isclose(sabine_rt60(V, S * 0.1), 0.1611 * 240 / 26.8, rtol=1e-3)


def test_eyring_below_sabine_and_converges():
    V, S = 240.0, 268.0
    for a in (0.05, 0.2, 0.5, 0.8):
        assert eyring_rt60(V, S, a) < sabine_rt60(V, S * a)
    # alpha -> 0: ratio -> 1
    a = 0.005
    assert abs(eyring_rt60(V, S, a) / sabine_rt60(V, S * a) - 1) < 0.01


def test_eyring_full_absorption_is_near_zero():
    assert eyring_rt60(240.0, 268.0, 1.0) < 0.02


def test_air_absorption_shortens_large_room_hf():
    room = make_room(40, 60, 20, all="marble")
    ac = analyse_room(room)
    assert ac.rt60_eyring[-1] < 0.5 * ac.rt60_eyring[3]


def test_critical_distance():
    assert np.isclose(critical_distance(1000.0, 1.0), 0.057 * np.sqrt(1000.0))


def test_absorption_scale_and_override():
    room = make_room(all="plaster")
    base = analyse_room(room).rt60_used
    room.absorption_scale = 2.0
    assert np.all(analyse_room(room).rt60_used < base)
    room.absorption_scale = 1.0
    room.rt60_override = 1.5
    ac = analyse_room(room)
    assert np.isclose(0.5 * (ac.rt60_used[2] + ac.rt60_used[3]), 1.5)
    assert ac.rt60_source == "user_override"
