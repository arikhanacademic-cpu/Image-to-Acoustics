"""Statistical room-acoustics formulas (all per octave band).

Sabine:  T = 0.161 V / (A + 4 m V),           A = sum S_i a_i + A_obj
Eyring:  T = 0.161 V / (-S ln(1 - a_mean) + A_obj + 4 m V)
           with a_mean = sum S_i a_i / S  (surfaces only)

Eyring converges to Sabine as a_mean -> 0 and stays finite-and-small as
a_mean -> 1 (Sabine wrongly predicts T > 0 for a fully absorbing room).
The engine uses Eyring by default for every band and reports Sabine
alongside it. 0.161 = 24 ln(10) / c with c = 343 m/s.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .bands import N_BANDS, SPEED_OF_SOUND, air_absorption_m
from .models import RoomParameters, SURFACE_NAMES

K_SABINE = 24.0 * np.log(10.0) / SPEED_OF_SOUND  # ~0.1611 s/m


def sabine_rt60(volume, surface_area_alpha_sum, object_area=0.0, air_m=0.0):
    A = np.asarray(surface_area_alpha_sum) + object_area + 4.0 * np.asarray(air_m) * volume
    return K_SABINE * volume / np.maximum(A, 1e-9)


def eyring_rt60(volume, total_surface, mean_alpha, object_area=0.0, air_m=0.0):
    a = np.clip(np.asarray(mean_alpha, dtype=float), 0.0, 0.9999)
    A = -total_surface * np.log1p(-a) + object_area + 4.0 * np.asarray(air_m) * volume
    return K_SABINE * volume / np.maximum(A, 1e-9)


def critical_distance(volume, rt60, directivity_q=1.0):
    """Distance where direct and diffuse energy are equal: r_c = 0.057 sqrt(Q V / T)."""
    return 0.057 * np.sqrt(directivity_q * volume / np.maximum(rt60, 1e-6))


def mean_free_path(volume, total_surface):
    return 4.0 * volume / total_surface


def mixing_time(volume) -> float:
    """Perceptual mixing time ~ sqrt(V) ms (Polack), clamped to 25..150 ms."""
    return float(np.clip(np.sqrt(volume) * 1e-3, 0.025, 0.150))


def schroeder_frequency(rt60_mid, volume):
    return 2000.0 * np.sqrt(rt60_mid / volume)


def barron_late_energy(rt60, volume, distance, t_after_direct):
    """Reflected energy arriving after `t_after_direct` seconds, relative to the
    direct sound energy at 1 m (Barron & Lee 1988 revised theory):

        E(t) = (31200 T / V) e^{-0.04 r / T} e^{-13.82 t / T}  /  100
    """
    T = np.maximum(np.asarray(rt60, float), 1e-4)
    return (312.0 * T / volume) * np.exp(-0.04 * distance / T) * np.exp(-13.82 * t_after_direct / T)


@dataclass
class RoomAcoustics:
    volume: float
    total_surface: float
    mean_alpha: np.ndarray      # surfaces only, per band
    object_area: np.ndarray     # m^2 Sabine, per band
    rt60_sabine: np.ndarray
    rt60_eyring: np.ndarray
    rt60_used: np.ndarray       # what drives the tail (Eyring or manual override)
    rt60_source: str            # "eyring" | "user_override"
    mean_scattering: float
    mixing_time: float
    mean_free_path: float

    def as_dict(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


def analyse_room(room: RoomParameters) -> RoomAcoustics:
    areas = room.surface_areas()
    S = sum(areas.values())
    SA = sum(areas[n] * room.surface_alpha(n) for n in SURFACE_NAMES)
    mean_alpha = SA / S
    A_obj = room.object_area()
    m = air_absorption_m()
    V = room.volume
    t_sab = sabine_rt60(V, SA, A_obj, m)
    t_ey = eyring_rt60(V, S, mean_alpha, A_obj, m)
    if room.rt60_override is not None:
        ov = np.broadcast_to(np.atleast_1d(np.asarray(room.rt60_override, float)), (N_BANDS,)).copy()
        if np.size(room.rt60_override) == 1:
            # scalar = mid-frequency (500/1k) target; keep the physical spectral shape
            mid = 0.5 * (t_ey[2] + t_ey[3])
            ov = t_ey * (float(np.atleast_1d(room.rt60_override)[0]) / mid)
        used, src = ov, "user_override"
    else:
        used, src = t_ey, "eyring"
    scat = sum(areas[n] * room.surfaces[n].scattering() for n in SURFACE_NAMES) / S
    scat = float(1.0 - (1.0 - scat) * (1.0 - room.clutter_scattering))
    return RoomAcoustics(V, S, mean_alpha, A_obj, t_sab, t_ey, used, src, scat,
                         mixing_time(V), mean_free_path(V, S))
