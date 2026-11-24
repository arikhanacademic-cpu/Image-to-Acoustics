import sys, pathlib
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from acoustics.models import RoomParameters, Surface, Dimension, Placement  # noqa: E402

FS = 48000


def make_room(W=6, L=8, H=3, **surf):
    names = ("floor", "ceiling", "wall_x0", "wall_x1", "wall_y0", "wall_y1")
    default = surf.pop("all", "plaster")
    surfaces = {n: Surface.of(surf.get(n, default)) for n in names}
    return RoomParameters(name="test", width=Dimension(value=W), length=Dimension(value=L),
                          height=Dimension(value=H), surfaces=surfaces)


@pytest.fixture
def fs():
    return FS


@pytest.fixture
def anechoic():
    return make_room(10, 10, 10, all="open")


@pytest.fixture
def voice():
    """Speech-like test signal: glottal pulse train through formant resonators."""
    rng = np.random.default_rng(1)
    n = int(1.5 * FS)
    t = np.arange(n) / FS
    f0 = 120 + 20 * np.sin(2 * np.pi * 2 * t)
    pulses = (np.diff(np.floor(np.cumsum(f0) / FS), prepend=0) > 0).astype(float)
    from scipy import signal
    y = pulses + 0.02 * rng.standard_normal(n)
    for fc, bw in [(700, 110), (1200, 120), (2600, 160)]:
        b, a = signal.iirpeak(fc, fc / bw, fs=FS)
        y = signal.lfilter(b, a, y) + 0.3 * y
    y *= (np.sin(2 * np.pi * 3 * t) > -0.3)  # syllable gating
    return 0.8 * y / np.max(np.abs(y))
