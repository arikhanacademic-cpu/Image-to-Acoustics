"""Room impulse response synthesis.

Hybrid model
------------
1. Direct sound: delay r/c, 1/r spreading, ISO 9613 air absorption, speech
   directivity (emission angle), obstruction filter.
2. Early specular reflections: image-source method for the shoebox
   (Allen & Berkley 1979), fully vectorised, per octave band:
      amplitude = prod_w beta_w^k_w * prod_w sqrt(1 - s_w)^k_w * e^{-m d / 2} / d
   with beta = sqrt(1 - alpha). Fractional delays are exact (built in the
   frequency domain). Arrivals are tapered out at the mixing time.
3. Late reverberation: Gaussian noise in 1/3-octave sub-bands, each with
   exponential decay exp(-6.91 t / T(f)), T(f) log-interpolated from the
   octave-band model so the decay rate varies smoothly with frequency. Its energy is calibrated to Barron's revised
   theory (relative to the direct sound at 1 m, corrected for source
   directivity Q). A build-up ramp follows the diffuse fraction
   1 - (1 - s)^(c t / mfp), forced to 1 at the mixing time.
4. Binaural-ish rendering: every arrival gets a Woodworth ITD and a
   per-band head-shadow ILD; the tail is decorrelated between ears with
   frequency-dependent interaural coherence.

Three channels are produced: omni (for measurement / mono export), left, right.
Components (direct, early, late) are kept separate so wet/dry and reflection
intensity can be changed without re-simulating.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy import fft as sfft

from .bands import (N_BANDS, SPEED_OF_SOUND as C, band_weights, air_absorption_m,
                    partition_weights, TAIL_CENTRES_HZ, interp_log)
from .models import RoomParameters, Placement
from .theory import analyse_room, barron_late_energy, critical_distance, RoomAcoustics

# Speech directivity: attenuation (dB) directly behind the talker, per band.
# Approximates published talker directivity (e.g. Monson et al. 2012).
VOICE_REAR_DB = np.array([0.5, 1.5, 3.0, 6.0, 9.0, 12.0, 15.0])
# Obstruction (thin barrier / partial occlusion) at obstruction = 1, per band (dB).
# Shape follows Maekawa-type diffraction loss growing ~3 dB/octave+.
OBSTRUCTION_DB = np.array([4.0, 6.0, 9.0, 13.0, 17.0, 21.0, 25.0])
OBSTRUCTION_REFLECTION_SHARE = 0.3   # fraction of the loss also applied to reflections
# Simplified spherical-head model.
HEAD_RADIUS = 0.0875
HEAD_SHADOW_DB = np.array([0.5, 1.0, 2.5, 5.0, 8.0, 11.0, 14.0])   # far ear at 90 deg
NEAR_EAR_BOOST_DB = np.array([0.0, 0.0, 0.5, 1.0, 2.0, 3.0, 3.0])
TAIL_COHERENCE = np.array([0.9, 0.7, 0.4, 0.2, 0.1, 0.05, 0.05])   # interaural, diffuse field
PAD_S = 0.010  # negative-time guard for zero-phase filtering

# axis -> (surface at 0, surface at L)
AXIS_SURFACES = (("wall_x0", "wall_x1"), ("wall_y0", "wall_y1"), ("floor", "ceiling"))


def voice_directivity_db(cos_theta: np.ndarray) -> np.ndarray:
    """(K,) cosine of emission angle -> (K, N_BANDS) gain in dB (<= 0)."""
    return -VOICE_REAR_DB[None, :] * (1.0 - np.asarray(cos_theta)[:, None]) / 2.0


def voice_directivity_q() -> np.ndarray:
    """Directivity factor Q per band implied by `voice_directivity_db`."""
    th = np.linspace(0, np.pi, 2001)
    g2 = 10 ** (voice_directivity_db(np.cos(th)) / 10.0)          # (T, B)
    integral = 2 * np.pi * np.trapezoid(g2 * np.sin(th)[:, None], th, axis=0)
    return 4 * np.pi / integral


@dataclass
class ImpulseResponse:
    fs: int
    direct: np.ndarray   # (3, N)  omni, left, right
    early: np.ndarray    # (3, N)
    late: np.ndarray     # (3, N)
    t_direct: float
    info: dict = field(default_factory=dict)

    def combine(self, wet_gain: float = 1.0, reflection_intensity: float = 1.0) -> np.ndarray:
        return self.direct + wet_gain * (reflection_intensity * self.early + self.late)

    @property
    def n(self) -> int:
        return self.direct.shape[1]


def _image_sources(room: RoomParameters, src, lst, max_path: float):
    """Enumerate shoebox image sources within max_path.

    Returns dict with positions (K,3), distances (K,), log-amplitude per band
    from wall absorption + specular share (K,B), reflection order (K,),
    and the axis flip pattern (K,3) used to recover the emission direction.
    """
    dims = room.dims
    per_axis = []
    for i in range(3):
        L, s = dims[i], src[i]
        N = int(np.ceil(max_path / (2 * L))) + 1
        n = np.arange(-N, N + 1)
        nn, qq = np.meshgrid(n, np.array([0, 1]), indexing="ij")
        nn, qq = nn.ravel(), qq.ravel()
        flip = 1 - 2 * qq
        pos = flip * s + 2 * nn * L
        k_lo, k_hi = np.abs(nn - qq), np.abs(nn)
        w_lo, w_hi = AXIS_SURFACES[i]
        a_lo, a_hi = room.surface_alpha(w_lo), room.surface_alpha(w_hi)
        s_lo = room.surfaces[w_lo].scattering()
        s_hi = room.surfaces[w_hi].scattering()
        with np.errstate(divide="ignore"):
            lb_lo = 0.5 * np.log(np.maximum(1 - a_lo, 0)) + 0.5 * np.log(1 - min(s_lo, 0.999))
            lb_hi = 0.5 * np.log(np.maximum(1 - a_hi, 0)) + 0.5 * np.log(1 - min(s_hi, 0.999))
        # 0 ** -inf issue: order 0 must give exactly 0 log-amp even for open surfaces
        with np.errstate(invalid="ignore"):
            la = (np.where(k_lo[:, None] > 0, k_lo[:, None] * lb_lo[None, :], 0.0)
                  + np.where(k_hi[:, None] > 0, k_hi[:, None] * lb_hi[None, :], 0.0))
        per_axis.append((pos, la, k_lo + k_hi, flip))

    (px, lx, kx, fx), (py, ly, ky, fy), (pz, lz, kz, fz) = per_axis
    ix, iy, iz = np.meshgrid(np.arange(len(px)), np.arange(len(py)), np.arange(len(pz)), indexing="ij")
    ix, iy, iz = ix.ravel(), iy.ravel(), iz.ravel()
    pos = np.stack([px[ix], py[iy], pz[iz]], axis=1)
    d = np.linalg.norm(pos - lst[None, :], axis=1)
    keep = d <= max_path
    ix, iy, iz, pos, d = ix[keep], iy[keep], iz[keep], pos[keep], d[keep]
    logamp = lx[ix] + ly[iy] + lz[iz]
    order = kx[ix] + ky[iy] + kz[iz]
    flip = np.stack([fx[ix], fy[iy], fz[iz]], axis=1)
    alive = np.isfinite(logamp).all(axis=1) | (order == 0)
    alive &= np.max(logamp, axis=1) > np.log(1e-5)
    return dict(pos=pos[alive], d=d[alive], logamp=logamp[alive], order=order[alive], flip=flip[alive])


def _synth_arrivals(fs, nfft, delays, log_gains, chunk=256):
    """Sum of arrivals with exact fractional delays and per-band gains.
    log_gains: (K, B) natural-log amplitude. Returns time signal length nfft
    (time origin shifted by PAD_S)."""
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    W = band_weights(freqs)                       # (B, F)
    H = np.zeros(len(freqs), dtype=complex)
    lg = np.maximum(log_gains, -60.0)
    for k0 in range(0, len(delays), chunk):
        g = np.exp(lg[k0:k0 + chunk] @ W)          # log-domain interpolation, (k, F)
        ph = np.exp(-2j * np.pi * np.outer(delays[k0:k0 + chunk] + PAD_S, freqs))
        H += np.sum(g * ph, axis=0)
    return np.fft.irfft(H, nfft)


def build_impulse_response(room: RoomParameters, placement: Placement, fs: int = 48000,
                           seed: int = 0, max_seconds: float = 15.0) -> ImpulseResponse:
    rng = np.random.default_rng(seed)
    ac: RoomAcoustics = analyse_room(room)
    src, lst, sdir, lyaw = placement.resolved(room.dims)
    r0 = float(np.linalg.norm(src - lst))
    t0 = r0 / C
    t_mix = ac.mixing_time
    T = ac.rt60_used
    m_air = air_absorption_m()

    # ------------------------------------------------------------- length
    t_end = t0 + t_mix + float(np.max(T)) * (80.0 / 60.0)
    n_total = int(np.ceil(min(max(t_end, 0.3), max_seconds) * fs))
    pad = int(round(PAD_S * fs))

    # ------------------------------------------------- image sources (+ direct)
    t_early_end = t0 + t_mix
    imgs = _image_sources(room, src, lst, max_path=C * t_early_end)
    d = np.maximum(imgs["d"], 0.05)
    K = len(d)
    tau_after = d / C - t0

    # emission direction in the real source's frame -> speech directivity
    v = (lst[None, :] - imgs["pos"]) / d[:, None] * imgs["flip"]
    cos_em = v @ sdir
    log_g = (imgs["logamp"]
             - np.log(d)[:, None]
             - 0.5 * m_air[None, :] * d[:, None]
             + voice_directivity_db(cos_em) * np.log(10) / 20)
    is_direct = imgs["order"] == 0
    obs_db = -placement.obstruction * OBSTRUCTION_DB
    log_g += np.where(is_direct[:, None], 1.0, OBSTRUCTION_REFLECTION_SHARE) * obs_db[None, :] * np.log(10) / 20

    # taper specular arrivals out over the last 30 % of the mixing window
    x = np.clip((tau_after - 0.7 * t_mix) / (0.3 * t_mix), 0.0, 1.0)
    taper = np.cos(0.5 * np.pi * x) ** 2
    with np.errstate(divide="ignore"):
        log_g += np.where(is_direct, 0.0, np.log(np.maximum(taper, 1e-30)))[:, None]

    # head model
    u = (imgs["pos"] - lst[None, :]) / d[:, None]
    left_axis = np.array([-np.sin(lyaw), np.cos(lyaw), 0.0])
    lat = np.clip(u @ left_axis, -1, 1)
    phi = np.arcsin(np.abs(lat))
    itd = HEAD_RADIUS / C * (phi + np.sin(phi))
    shadow = -HEAD_SHADOW_DB[None, :] * np.abs(lat)[:, None] * np.log(10) / 20
    boost = NEAR_EAR_BOOST_DB[None, :] * np.abs(lat)[:, None] * np.log(10) / 20
    left_near = (lat >= 0)[:, None]
    log_L = log_g + np.where(left_near, boost, shadow)
    log_R = log_g + np.where(left_near, shadow, boost)
    del_L = d / C + np.where(lat >= 0, -itd / 2, itd / 2)
    del_R = d / C + np.where(lat >= 0, itd / 2, -itd / 2)

    n_early = int(np.ceil((t_early_end + 0.02) * fs)) + 2 * pad
    nfft_e = int(2 ** np.ceil(np.log2(n_early + int(0.05 * fs))))

    def comp(mask):
        out = np.zeros((3, n_total))
        if not np.any(mask):
            return out
        for ch, (dl, lg) in enumerate([(d / C, log_g), (del_L, log_L), (del_R, log_R)]):
            y = _synth_arrivals(fs, nfft_e, dl[mask], lg[mask])[pad:]
            n = min(len(y), n_total)
            out[ch, :n] = y[:n]
        return out

    direct = comp(is_direct)
    early = comp(~is_direct)

    # ------------------------------------------------------- stochastic tail
    Q = voice_directivity_q()
    t = np.arange(n_total) / fs
    tau = t - t0
    pos = tau > 0
    diff_frac = 1.0 - (1.0 - min(ac.mean_scattering, 0.999)) ** np.maximum(C * tau / ac.mean_free_path, 0)
    sm = np.clip(tau / t_mix, 0, 1)
    smooth = sm * sm * (3 - 2 * sm)
    ramp = np.where(pos, diff_frac + (1 - diff_frac) * smooth, 0.0)

    nfft_l = int(2 ** np.ceil(np.log2(n_total + 2 * pad + int(0.1 * fs))))
    freqs = np.fft.rfftfreq(nfft_l, 1 / fs)
    centres = TAIL_CENTRES_HZ[TAIL_CENTRES_HZ < 0.45 * fs]
    sqrtW = np.sqrt(partition_weights(freqs, centres))
    T_sub = interp_log(T, centres)
    Q_sub = interp_log(Q, centres)
    coh_sub = interp_log(TAIL_COHERENCE, centres)
    spec = np.zeros((3, nfft_l // 2 + 1), dtype=complex)   # filter is linear: sum spectra, one inverse FFT
    i_mix = int(np.ceil((t0 + t_mix) * fs))
    tau_pos = np.maximum(tau, 0)
    sqrt_ramp = np.sqrt(ramp)
    buf = np.zeros((3, nfft_l))
    for b in range(len(centres)):
        env = np.where(pos, np.exp(-6.91 * tau_pos / T_sub[b]) * sqrt_ramp, 0.0)
        # Barron energy relative to the on-axis direct sound at 1 m, / Q for talker directivity
        target = barron_late_energy(T_sub[b], ac.volume, r0, t_mix) / Q_sub[b]
        env_e = np.sum(env[i_mix:] ** 2)
        A = np.sqrt(target / env_e) if env_e > 0 else 0.0
        c = coh_sub[b]
        shared = rng.standard_normal(n_total)
        buf[0, pad:pad + n_total] = A * env * rng.standard_normal(n_total)
        for ch in (1, 2):
            buf[ch, pad:pad + n_total] = A * env * (np.sqrt(c) * shared + np.sqrt(1 - c) * rng.standard_normal(n_total))
        spec += sfft.rfft(buf, axis=1, workers=-1) * sqrtW[b][None, :]
    late = sfft.irfft(spec, nfft_l, axis=1, workers=-1)[:, pad:pad + n_total]

    info = dict(
        acoustics=ac.as_dict(),
        source=src.tolist(), listener=lst.tolist(), source_dir=sdir.tolist(),
        listener_yaw_deg=float(np.rad2deg(lyaw)),
        distance=r0, t_direct=t0, n_image_sources=int(K - is_direct.sum()),
        voice_q=Q.tolist(),
        critical_distance=critical_distance(ac.volume, ac.rt60_used, Q).tolist(),
    )
    return ImpulseResponse(fs, direct, early, late, t0, info)
