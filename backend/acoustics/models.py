"""Parameter schema for the acoustic engine.

This is the contract between the (future) vision model, the user's manual
corrections, and the simulation. Every physical quantity can carry
provenance ("ai_estimate", "user", "measured", "preset", "default") and a
confidence, so the UI can separate measured values from AI estimates.

Coordinate frame: shoebox with one corner at the origin.
x in [0, width], y in [0, length], z in [0, height] (z up).
Yaw angles are in degrees, counter-clockwise from +x in the x-y plane.
"""
from __future__ import annotations
from typing import Literal, Optional, Union
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from .bands import N_BANDS
from .materials import get_material, MATERIALS, OBJECT_ABSORPTION

Provenance = Literal["ai_estimate", "user", "measured", "preset", "default"]
SURFACE_NAMES = ("floor", "ceiling", "wall_x0", "wall_x1", "wall_y0", "wall_y1")


class MaterialShare(BaseModel):
    material: str
    fraction: float = Field(1.0, gt=0.0, le=1.0)

    @field_validator("material")
    @classmethod
    def _known(cls, v):
        get_material(v)
        return v


class Surface(BaseModel):
    """One boundary of the shoebox, possibly made of several materials."""
    materials: list[MaterialShare]
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    provenance: Provenance = "default"

    @model_validator(mode="after")
    def _fractions(self):
        total = sum(m.fraction for m in self.materials)
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"material fractions must sum to 1 (got {total:.3f})")
        return self

    @classmethod
    def of(cls, *pairs, **kw) -> "Surface":
        """Surface.of("plaster") or Surface.of(("plaster", .7), ("glass_window", .3))."""
        if len(pairs) == 1 and isinstance(pairs[0], str):
            pairs = ((pairs[0], 1.0),)
        return cls(materials=[MaterialShare(material=m, fraction=f) for m, f in pairs], **kw)

    def alpha(self) -> np.ndarray:
        return sum(m.fraction * get_material(m.material).alpha_arr for m in self.materials)

    def scattering(self) -> float:
        return float(sum(m.fraction * get_material(m.material).scattering for m in self.materials))


class AbsorberObject(BaseModel):
    kind: str
    count: float = Field(1.0, ge=0.0)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    provenance: Provenance = "default"

    @field_validator("kind")
    @classmethod
    def _known(cls, v):
        if v not in OBJECT_ABSORPTION:
            raise ValueError(f"Unknown object '{v}'. Known: {sorted(OBJECT_ABSORPTION)}")
        return v

    def area(self) -> np.ndarray:
        return self.count * np.array(OBJECT_ABSORPTION[self.kind], dtype=float)


class Dimension(BaseModel):
    value: float = Field(..., gt=0.2, le=500.0)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    provenance: Provenance = "default"


class RoomParameters(BaseModel):
    name: str = "room"
    enclosure: Literal["indoor", "partially_enclosed", "outdoor"] = "indoor"
    width: Dimension
    length: Dimension
    height: Dimension
    surfaces: dict[str, Surface]
    objects: list[AbsorberObject] = []
    # Extra diffusion from clutter / irregular geometry the shoebox can't model (0..1).
    clutter_scattering: float = Field(0.0, ge=0.0, le=1.0)
    # Manual overrides (advanced controls). None = derive from physics.
    absorption_scale: float = Field(1.0, gt=0.0, le=5.0)
    rt60_override: Optional[Union[float, list[float]]] = None
    assumptions: list[str] = []

    @field_validator("surfaces")
    @classmethod
    def _all_surfaces(cls, v):
        missing = set(SURFACE_NAMES) - set(v)
        extra = set(v) - set(SURFACE_NAMES)
        if missing or extra:
            raise ValueError(f"surfaces must be exactly {SURFACE_NAMES}; missing={missing} extra={extra}")
        return v

    @field_validator("rt60_override")
    @classmethod
    def _rt(cls, v):
        if v is None:
            return v
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        if arr.size not in (1, N_BANDS) or np.any(arr <= 0.01) or np.any(arr > 30):
            raise ValueError("rt60_override must be a scalar or 7 values in (0.01, 30] s")
        return v

    @property
    def dims(self) -> np.ndarray:
        return np.array([self.width.value, self.length.value, self.height.value])

    @property
    def volume(self) -> float:
        return float(np.prod(self.dims))

    def surface_areas(self) -> dict[str, float]:
        W, L, H = self.dims
        return {"floor": W * L, "ceiling": W * L, "wall_x0": L * H,
                "wall_x1": L * H, "wall_y0": W * H, "wall_y1": W * H}

    def surface_alpha(self, name: str) -> np.ndarray:
        return np.clip(self.surfaces[name].alpha() * self.absorption_scale, 0.0, 1.0)

    def object_area(self) -> np.ndarray:
        total = np.zeros(N_BANDS)
        for o in self.objects:
            total += o.area()
        return total * self.absorption_scale


class Placement(BaseModel):
    source: tuple[float, float, float]
    listener: tuple[float, float, float]
    # "towards" = face the listener, "away" = face directly away, or yaw in degrees.
    source_facing: Union[Literal["towards", "away"], float] = "towards"
    # "towards_source" or yaw in degrees.
    listener_facing: Union[Literal["towards_source"], float] = "towards_source"
    obstruction: float = Field(0.0, ge=0.0, le=1.0)  # 0 none .. 1 heavy wall/door
    distance_override: Optional[float] = Field(None, gt=0.1, le=200.0)

    def resolved(self, dims: np.ndarray, margin: float = 0.1):
        """Return (src, lst, src_dir_unit, listener_yaw_rad) clipped inside the room."""
        lo, hi = np.full(3, margin), dims - margin
        lst = np.clip(np.array(self.listener, float), lo, hi)
        src = np.clip(np.array(self.source, float), lo, hi)
        if self.distance_override is not None:
            d = src - lst
            n = np.linalg.norm(d)
            u = d / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])
            src = np.clip(lst + u * self.distance_override, lo, hi)
        to_lst = lst - src
        to_lst_h = np.array([to_lst[0], to_lst[1], 0.0])
        if np.linalg.norm(to_lst_h) < 1e-6:
            to_lst_h = np.array([0.0, 1.0, 0.0])
        if self.source_facing == "towards":
            sdir = to_lst_h / np.linalg.norm(to_lst_h)
        elif self.source_facing == "away":
            sdir = -to_lst_h / np.linalg.norm(to_lst_h)
        else:
            a = np.deg2rad(float(self.source_facing))
            sdir = np.array([np.cos(a), np.sin(a), 0.0])
        if self.listener_facing == "towards_source":
            lyaw = float(np.arctan2(-to_lst_h[1], -to_lst_h[0]))
        else:
            lyaw = np.deg2rad(float(self.listener_facing))
        return src, lst, sdir, lyaw


class MixControls(BaseModel):
    # 1.0 = physically derived balance. Scales all reflected sound (early + late).
    wet_gain: float = Field(1.0, ge=0.0, le=4.0)
    # Additional scale on specular early reflections only.
    reflection_intensity: float = Field(1.0, ge=0.0, le=4.0)
    # "peak": normalise so output peak = ceiling.
    # "physical": keep the 1 m reference gain (distance loudness cue); only turn
    #             down if the result would exceed the ceiling.
    # "match": same speech-gated loudness as the dry input (fair A/B comparison),
    #          reduced only if the peak would exceed the ceiling.
    output_mode: Literal["peak", "physical", "match"] = "peak"
    ceiling_dbfs: float = Field(-1.0, ge=-30.0, le=0.0)
    stereo: bool = True
