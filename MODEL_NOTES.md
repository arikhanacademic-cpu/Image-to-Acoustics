# Acoustic model: equations, assumptions, and review targets

This document is written for an independent review of the acoustic
mathematics. Every constant lives in code under the name used here. Section 8
lists known simplifications, ordered by how much I expect them to matter for
realism.

Conventions: c = 343 m/s. Octave bands b = 125, 250, 500, 1k, 2k, 4k, 8k Hz.
Energy quantities are referenced to the **on-axis direct sound at 1 m = 1**.
In the IR this is a unit impulse with flat spectrum (Σh² = 1).

---

## 1. Statistical room model (`theory.py`)

Surface absorption: A_s(b) = Σ_i S_i α_i(b). Mixed surfaces use
area-weighted α, and `absorption_scale` multiplies every α (clipped to 1).
Objects add equivalent absorption areas A_obj(b) (m² Sabine per item,
`materials.OBJECT_ABSORPTION`).

Air absorption: m(b) = α_ISO(b) / (1000 · 10 log₁₀ e) in 1/m (energy), with
ISO 9613-1 values at 20 °C, 50 % RH of 0.44, 1.31, 2.73, 4.66, 9.86, 32.8 and
117 dB/km.

- **Sabine:** T = 0.1611 V / (A_s + A_obj + 4mV)
- **Eyring:** T = 0.1611 V / (−S ln(1 − ᾱ) + A_obj + 4mV), with ᾱ = A_s/S
  (surfaces only, so objects are treated Sabine-style)
- 0.1611 = 24 ln 10 / c.

Eyring is used for every band, because it converges to Sabine as ᾱ → 0 and
behaves correctly as ᾱ → 1 (the forest, the studio, open boundaries). Sabine
is reported alongside. A manual `rt60_override` can be a scalar, which
rescales the Eyring curve so the 500 Hz–1 kHz mean hits the target while
keeping its spectral shape, or 7 explicit values.

Derived quantities:
- Mixing time t_mix = √V ms, clamped to 25–150 ms
- Mean free path 4V/S
- Critical distance r_c = 0.057 √(Q V / T)

## 2. Direct sound and image sources (`ir.py`)

The room is a shoebox [0,W]×[0,L]×[0,H]. Image positions follow Allen &
Berkley: per axis, x = (1−2q)·x_s + 2nL with q ∈ {0,1}. The image hits the
lower wall |n−q| times and the upper wall |n| times. All images with path
length d ≤ c(t_direct + t_mix) are enumerated in one vectorised pass.

Per-band amplitude of an arrival:

    a(b) = Π_w [ √(1−α_w(b)) · √(1−s_w) ]^{k_w}      wall reflection × specular share
         · exp(−m(b) d / 2)                           air
         · 10^{D_b(θ)/20}                             talker directivity
         · 10^{−o·O_b·κ/20}                           obstruction (κ = 1 direct, 0.3 reflections)
         / d                                          spherical spreading

**Emission angle.** The image's emission vector (receiver − image)/d is
mirrored back into the real source's frame by flipping each axis where q = 1.
That gives the true departure direction from the talker for every reflection,
so talking towards a wall really does brighten that wall's reflection.

**Talker directivity.** D_b(θ) = −R_b (1 − cos θ)/2, with rear attenuation
R = 0.5, 1.5, 3, 6, 9, 12, 15 dB. The implied directivity factor Q_b is
computed numerically (1.0 at 125 Hz rising to ~3–4 at 8 kHz) and is used
again in the tail calibration and r_c.

**Synthesis.** Each arrival is built in the frequency domain:
H(f) = Σ_k exp(Σ_b ln a_k(b) · W_b(f)) · e^{−j2πf τ_k}. Fractional delays are
exact. W_b is a cos² partition of unity in log₂ f, so the gain interpolation
is log-linear in dB between band centres. The filter is zero-phase: a 10 ms
guard absorbs the (very small) acausal spread and is then discarded.

**Window.** Specular arrivals are tapered out with cos² over the last 30 % of
t_mix after the direct sound.

## 3. Late reverberation

Energy target (Barron & Lee 1988 revised theory, converted to the 1 m
reference and divided by talker Q to convert from omni power):

    E_late(t_after_direct ≥ t_mix, b) = (312 T_b / V) · e^{−0.04 r / T_b} · e^{−13.82 t_mix / T_b} / Q_b

Synthesis runs on a **1/3-octave grid**, 25 sub-bands from about 62 Hz to
16 kHz. T, Q and coherence are log-interpolated from the octave values, so the
decay rate varies smoothly with frequency. An earlier version used 7 octave
sub-bands, and between band centres it produced a *sum of two exponentials*.
That biased measured T30 by up to 25 %, and the current tests catch it.

Each sub-band is Gaussian noise · e^{−6.91 τ / T} · √ramp(τ), filtered by the
square root of its cos² weight, and scaled so that its energy after t_mix
equals E_late.

**Build-up ramp.** ramp(τ) = d(τ) + (1 − d(τ))·smoothstep(τ/t_mix), where
d(τ) = 1 − (1 − s̄)^{cτ/ℓ} is the fraction of energy scattered after cτ/ℓ
reflections. Highly scattering rooms (cave, forest) therefore get diffuse
energy early, and specular rooms (bathroom, marble) stay "specular" until
t_mix.

**Ears.** noise_L,R = √c_b·shared + √(1−c_b)·independent, with interaural
coherence c = 0.9, 0.7, 0.4, 0.2, 0.1, 0.05, 0.05. The omni channel is an
independent realisation used for measurement and mono export.

## 4. Head model

For each arrival, the lateral component is ℓ = u·(left axis) and φ = asin|ℓ|.

- ITD (Woodworth, a = 8.75 cm): (a/c)(φ + sin φ)
- Far-ear shadow: −H_b|ℓ| with H = 0.5, 1, 2.5, 5, 8, 11, 14 dB
- Near-ear boost: +B_b|ℓ| with B = 0, 0, 0.5, 1, 2, 3, 3 dB

There are no pinna or elevation cues.

## 5. Obstruction

Direct loss is o·O_b, with o ∈ [0,1] and O = 4, 6, 9, 13, 17, 21, 25 dB (a
Maekawa-like slope). Reflections get 30 % of that loss in dB, and the tail
gets none.

## 6. Rendering and gain

y = (x ∗ h_direct) + wet·(refl·(x ∗ h_early) + x ∗ h_late), computed as a full
linear convolution (`oaconvolve`).

- `peak` mode normalises to the ceiling (default −1 dBFS).
- `physical` mode keeps the 1 m reference gain, so distance changes loudness,
  and only turns the level down when it would clip.

There is no limiter or saturation, so the signal path is LTI. The tail is
trimmed at −90 dB re. peak, plus 50 ms and a 20 ms fade.

## 7. Verification (all automated, `tests/`)

- **Direct path:** arrival within 1 sample of r/c. The 1 kHz energy follows
  1/r² within 0.1 dB, and broadband energy at 1 m is 1 ± 3 %.
- **Floor reflection:** correct delay, and the 1 kHz level is within 0.4 dB of
  the analytic value (marble + directivity at 37° + air = −0.66 dB).
- **Tail calibration:** measured T30 is within 8 % of target for flat 0.5 /
  1.5 / 4 s targets and a sloped target. With real materials it is within
  30 % at 500 Hz–2 kHz, because steep T(f) steps inside an octave pull the
  octave T30 towards the longer neighbour, as happens in real measurements.
- **DRR vs theory:** the 1 kHz DRR, computed exactly from the separated
  direct/reflected components, is within 3 dB of 20 log₁₀(r_c/r) in every
  indoor preset (typically within 1.5 dB).
- **Pyroomacoustics cross-checks:**
  - First-order image positions are identical.
  - Pyroomacoustics' pure-specular ISM RT60 matches our Eyring value within
    15 % in a near-cubic room (7×8×6 m).
  - In a flat 6×8×3 m room, Pyroomacoustics decays **30–40 % slower** than
    Eyring. That is the non-diffuse-field effect our statistical tail does not
    capture (see 8.1).
- **Behaviour:**
  - Obstruction removes ≥ 8 dB more at 4 kHz than at 250 Hz.
  - Facing away costs > 8 dB at 4 kHz and < 3 dB at 250 Hz.
  - A source at 90° arrives 25–40 samples (≈ 0.66 ms) earlier at the near ear,
    with > 8 dB ILD at 4 kHz.
- **Environments:** RT ordering and ranges are plausible, the forest is
  direct-dominated, the bedroom's highs decay faster than its lows, air
  absorption shortens HF in large rooms, no two presets coincide, and the
  rendered spectral tilt differs between live and treated rooms.

The headline numbers are in `samples/comparison/REPORT.md`.

## 8. Known simplifications and suggested review targets

1. **Diffuse-field assumption / absorption distribution.** Eyring assumes
   uniformly distributed absorption. Rooms with absorption concentrated on one
   pair of surfaces (carpeted floor, absorbent ceiling, low flat rooms) decay
   slower than Eyring in reality. The Pyroomacoustics flat-room check shows
   +30–40 %. Candidates: Fitzroy, Arau-Puchades, Kuttruff's correction, or
   per-axis decay (Nilsson). *I expect this to be the largest realism gain.*
2. **Shoebox geometry.** Vaults, domes, L-shapes and caves are approximated as
   an equal-volume shoebox plus extra scattering. Pyroomacoustics polygonal
   rooms with ray tracing are a natural upgrade for early reflections.
3. **No room modes.** Below the Schroeder frequency (≈ 180 Hz in the bedroom)
   the response should be modal, not statistical. Small rooms would benefit
   from a modal synthesis layer (e.g. 20–200 Hz from the shoebox mode
   frequencies with per-mode decay).
4. **Energy bookkeeping before t_mix.** The ramp is a heuristic. Specular
   energy removed by scattering is not explicitly conserved, because the tail
   is calibrated only after t_mix. A Lambert-diffuse early component (diffuse
   rain / radiosity) would be more principled.
5. **Reflection coefficients.** These are real-valued, angle-independent
   random-incidence √(1−α), with no phase. Grazing incidence on porous ground
   (the forest) should use a locally reacting impedance model (Delany-Bazley +
   spherical-wave reflection coefficient), which would also produce the
   ground-effect dip.
6. **Obstruction.** This is a heuristic curve rather than geometry. It could
   compute the Fresnel number N = 2δ/λ from a barrier height and use
   Maekawa / Kurze-Anderson per band.
7. **Head model.** ITD/ILD only, with no HRTF or pinna cues, so front/back and
   elevation are ambiguous. A SOFA HRTF per arrival (early) plus a diffuse-field
   EQ for the tail is a straightforward swap.
8. **Talker directivity.** Axisymmetric (1−cos θ)/2 in dB. Measured data
   (e.g. Monson et al. 2012, Pollow et al.) shows up/down asymmetry and lobing.
9. **Barron's theory** assumes a diffuse field and Sabine-like decay beginning
   at the direct arrival. It may over-predict late energy close to the source
   in strongly non-diffuse rooms.
10. **Constant atmospheric conditions** (20 °C, 50 % RH), fixed c, and some
    material values that are estimates (marked `(est.)` in `materials.py`).
11. **Zero-phase interpolation filters.** The acausal spread is ≲ 1 ms and
    inaudible, but minimum-phase would be strictly causal.

## 9. Questions I would value an opinion on

- Which non-uniform-absorption formula would you trust for rooms estimated
  from a single image, given that the material distribution itself is
  uncertain?
- Is Barron's revised theory the right calibration for the tail, or would you
  calibrate from the image-source energy continuation at t_mix instead?
- Should the obstruction share applied to reflections (currently 30 %) depend
  on whether the obstacle is near the source, near the listener, or in
  between?
