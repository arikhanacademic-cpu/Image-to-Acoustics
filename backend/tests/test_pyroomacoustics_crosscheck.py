"""Cross-check against Pyroomacoustics (skipped if not installed)."""
import numpy as np
import pytest
from acoustics.ir import _image_sources
from conftest import make_room

pra = pytest.importorskip("pyroomacoustics")


def test_first_order_images_match_pyroomacoustics():
    room = make_room(6, 8, 3, all="plaster")
    src, lst = np.array([2.0, 3.0, 1.5]), np.array([4.0, 6.0, 1.2])
    p = pra.ShoeBox([6, 8, 3], fs=16000, max_order=1, materials=pra.Material(0.2))
    p.add_source(src.tolist()); p.add_microphone(lst.tolist())
    p.image_source_model()
    pra_imgs = p.sources[0].images.T
    pra_order = p.sources[0].orders
    ours = _image_sources(room, src, lst, max_path=100.0)
    ours1 = ours["pos"][ours["order"] == 1]
    ref = np.sort(np.round(pra_imgs[pra_order == 1], 6), axis=0)
    assert np.allclose(np.sort(np.round(ours1, 6), axis=0), ref)


@pytest.mark.parametrize("alpha", [0.15, 0.3])
def test_rt60_agrees_with_pyroomacoustics_ism_in_near_cubic_room(alpha):
    """Pure-specular ISM approaches the diffuse-field (Eyring) decay in
    near-cubic rooms. In flat rooms (e.g. 6x8x3 m) specular ISM decays
    30-40 % slower than Eyring (non-diffuse field) — see MODEL_NOTES.md."""
    from acoustics.theory import eyring_rt60
    dims = [7, 8, 6]
    p = pra.ShoeBox(dims, fs=16000, max_order=35, materials=pra.Material(alpha), air_absorption=False)
    p.add_source([2.0, 3.0, 1.5]); p.add_microphone([4.1, 4.2, 1.2])
    p.compute_rir()
    t_pra = pra.experimental.measure_rt60(p.rir[0][0], fs=16000, decay_db=30)
    V, S = 336.0, 2 * (56 + 42 + 48)
    t_ey = float(eyring_rt60(V, S, alpha))
    assert abs(t_pra / t_ey - 1) < 0.15, (t_pra, t_ey)
