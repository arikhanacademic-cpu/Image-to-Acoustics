"""Image-to-Acoustics — Phase 1 acoustic engine.

Physically motivated room acoustic simulation driven by explicit room
parameters (later supplied by the vision model and user corrections).

Public entry points:
    RoomParameters, Placement, MixControls   (models)
    build_impulse_response                    (ir)
    render                                    (render)
    measure_ir                                (metrics)
    PRESETS                                   (presets)
"""
from .models import RoomParameters, Surface, AbsorberObject, Placement, MixControls
from .ir import build_impulse_response, ImpulseResponse
from .render import render, RenderResult
from .metrics import measure_ir
from .presets import PRESETS

__all__ = [
    "RoomParameters", "Surface", "AbsorberObject", "Placement", "MixControls",
    "build_impulse_response", "ImpulseResponse", "render", "RenderResult",
    "measure_ir", "PRESETS",
]
