"""Frequency-dependent absorption coefficients (random incidence).

Values for 125 Hz .. 8 kHz octave bands. Most come from standard published
tables (e.g. Vorländer 'Auralization' Appendix, Everest 'Master Handbook of
Acoustics'); the 8 kHz column is extrapolated from 4 kHz where tables stop.
Entries marked (est.) are engineering estimates and should be reviewed.

`scattering` is a nominal broadband scattering coefficient used to split
reflected energy between specular (image-source) and diffuse (tail) parts.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    alpha: tuple  # 7 bands
    scattering: float = 0.1
    note: str = ""

    @property
    def alpha_arr(self) -> np.ndarray:
        return np.array(self.alpha, dtype=float)


_M = [
    Material("marble",            (0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03), 0.05),
    Material("ceramic_tile",      (0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03), 0.05),
    Material("concrete_smooth",   (0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04), 0.05),
    Material("concrete_painted",  (0.10, 0.05, 0.06, 0.07, 0.09, 0.08, 0.08), 0.05),
    Material("stone_masonry",     (0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06), 0.3),
    Material("brick",             (0.03, 0.03, 0.03, 0.04, 0.05, 0.07, 0.08), 0.2),
    Material("rough_rock",        (0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12), 0.6, "(est.) natural cave rock"),
    Material("plaster",           (0.013, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06), 0.05),
    Material("gypsum_board",      (0.29, 0.10, 0.05, 0.04, 0.07, 0.09, 0.09), 0.05),
    Material("wood_floor",        (0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07), 0.1),
    Material("wood_panelling",    (0.28, 0.22, 0.17, 0.09, 0.10, 0.11, 0.11), 0.1),
    Material("linoleum",          (0.02, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02), 0.05),
    Material("metal_sheet",       (0.12, 0.12, 0.10, 0.09, 0.08, 0.08, 0.08), 0.15, "(est.) uninsulated corrugated steel incl. panel losses"),
    Material("glass_window",      (0.35, 0.25, 0.18, 0.12, 0.07, 0.04, 0.03), 0.05),
    Material("carpet_thin",       (0.02, 0.06, 0.14, 0.37, 0.60, 0.65, 0.65), 0.1),
    Material("carpet_thick",      (0.08, 0.24, 0.57, 0.69, 0.71, 0.73, 0.73), 0.1),
    Material("curtains_heavy",    (0.14, 0.35, 0.55, 0.72, 0.70, 0.65, 0.65), 0.2),
    Material("acoustic_tile",     (0.50, 0.70, 0.60, 0.70, 0.70, 0.50, 0.50), 0.1),
    Material("acoustic_treatment",(0.30, 0.65, 0.90, 0.95, 0.95, 0.95, 0.95), 0.3, "(est.) 50-100 mm porous panels"),
    Material("grass_soil",        (0.11, 0.26, 0.60, 0.69, 0.92, 0.99, 0.99), 0.5),
    Material("forest_floor",      (0.20, 0.40, 0.55, 0.60, 0.65, 0.70, 0.75), 0.6, "(est.) leaf litter on soil"),
    Material("foliage_boundary",  (0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.93), 0.8, "(est.) mostly open, sparse trunks scatter"),
    Material("water",             (0.01, 0.01, 0.01, 0.015, 0.02, 0.02, 0.02), 0.05),
    Material("open",              (1.0,) * 7, 0.0, "opening / sky: nothing reflected"),
]
MATERIALS = {m.name: m for m in _M}


def get_material(name: str) -> Material:
    try:
        return MATERIALS[name]
    except KeyError:
        raise ValueError(f"Unknown material '{name}'. Known: {sorted(MATERIALS)}")


# Objects expressed as equivalent absorption area (m^2 Sabine) per band, per item.
OBJECT_ABSORPTION = {
    "bed":            (0.6, 1.2, 2.0, 2.4, 2.5, 2.5, 2.5),   # (est.) double bed + duvet
    "sofa":           (0.5, 0.9, 1.3, 1.6, 1.7, 1.7, 1.7),   # (est.)
    "upholstered_chair": (0.15, 0.25, 0.35, 0.40, 0.40, 0.40, 0.40),
    "wooden_chair":   (0.02, 0.02, 0.03, 0.04, 0.04, 0.04, 0.04),
    "desk":           (0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.06),
    "bookshelf_full": (0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5),   # (est.)
    "wardrobe":       (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    "person":         (0.15, 0.25, 0.45, 0.55, 0.60, 0.60, 0.60),
    "pew_wooden":     (0.10, 0.12, 0.15, 0.15, 0.15, 0.15, 0.15),  # per seat-place (est.)
    "tapestry":       (0.05, 0.15, 0.35, 0.50, 0.55, 0.55, 0.55),  # per m^2 (est.)
    "pallet_rack":    (0.1, 0.1, 0.15, 0.2, 0.2, 0.2, 0.2),        # (est.) partially stocked
}
