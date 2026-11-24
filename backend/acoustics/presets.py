"""Reference environments (stand-ins for vision-model output in Phase 1).

Each preset is a full RoomParameters + default Placement. Dimensions and
materials are typical values for the named space, flagged provenance
"preset". They are deliberately *physical descriptions*, not reverb
presets: RT60, reflection pattern and spectrum all fall out of the model.
"""
from __future__ import annotations
from .models import RoomParameters, Surface, AbsorberObject, Placement, Dimension


def _dim(v):
    return Dimension(value=v, confidence=1.0, provenance="preset")


def _s(*pairs):
    return Surface.of(*pairs, confidence=1.0, provenance="preset")


def _o(kind, n):
    return AbsorberObject(kind=kind, count=n, confidence=1.0, provenance="preset")


def _room(name, enclosure, W, L, H, floor, ceiling, x0, x1, y0, y1, objects=(), clutter=0.0, notes=()):
    return RoomParameters(
        name=name, enclosure=enclosure, width=_dim(W), length=_dim(L), height=_dim(H),
        surfaces=dict(floor=floor, ceiling=ceiling, wall_x0=x0, wall_x1=x1, wall_y0=y0, wall_y1=y1),
        objects=list(objects), clutter_scattering=clutter, assumptions=list(notes))


PRESETS: dict[str, tuple[RoomParameters, Placement]] = {
    "cathedral": (
        _room("Cathedral", "indoor", 25, 65, 24,
              _s(("stone_masonry", 0.85), ("carpet_thin", 0.05), ("wood_floor", 0.10)), _s("stone_masonry"),
              _s(("stone_masonry", 0.7), ("glass_window", 0.3)), _s(("stone_masonry", 0.7), ("glass_window", 0.3)),
              _s(("stone_masonry", 0.8), ("wood_panelling", 0.2)), _s(("stone_masonry", 0.6), ("wood_panelling", 0.3), ("glass_window", 0.1)),
              [_o("pew_wooden", 700), _o("upholstered_chair", 40), _o("tapestry", 40)], clutter=0.35,
              notes=["Vaulted nave approximated as a shoebox of equal volume",
                     "Columns, side chapels and vaulting modelled as extra scattering",
                     "Large stained-glass windows, choir stalls and wooden doors on end walls"]),
        Placement(source=(12.5, 12, 1.7), listener=(12.5, 30, 1.2))),
    "furnished_bedroom": (
        _room("Furnished bedroom", "indoor", 3.6, 4.2, 2.5,
              _s("carpet_thick"), _s("plaster"),
              _s("plaster"), _s(("plaster", 0.6), ("curtains_heavy", 0.4)),
              _s("gypsum_board"), _s("plaster"),
              [_o("bed", 1), _o("wardrobe", 1), _o("upholstered_chair", 1)], clutter=0.3),
        Placement(source=(1.2, 1.4, 1.5), listener=(2.6, 3.0, 1.2))),
    "empty_warehouse": (
        _room("Empty warehouse", "indoor", 30, 50, 10,
              _s("concrete_smooth"), _s("metal_sheet"),
              _s(("metal_sheet", 0.8), ("concrete_painted", 0.2)), _s(("metal_sheet", 0.8), ("concrete_painted", 0.2)),
              _s("concrete_painted"), _s(("metal_sheet", 0.7), ("open", 0.05), ("concrete_painted", 0.25)),
              [], clutter=0.05, notes=["Roller door slightly open (5 % of end wall)"]),
        Placement(source=(12, 20, 1.7), listener=(16, 26, 1.7))),
    "marble_palace": (
        _room("Marble palace hall", "indoor", 15, 32, 11,
              _s("marble"), _s(("plaster", 0.8), ("wood_panelling", 0.2)),
              _s(("marble", 0.55), ("glass_window", 0.30), ("curtains_heavy", 0.15)), _s(("marble", 0.85), ("wood_panelling", 0.15)),
              _s(("marble", 0.8), ("wood_panelling", 0.2)), _s(("marble", 0.8), ("wood_panelling", 0.2)),
              [_o("upholstered_chair", 24), _o("sofa", 4), _o("person", 4)], clutter=0.15,
              notes=["Tall windows with partly drawn curtains on one side", "Gilded wooden doors on end walls"]),
        Placement(source=(7, 8, 1.7), listener=(8, 16, 1.7))),
    "underground_cave": (
        _room("Underground cave", "indoor", 18, 40, 12,
              _s(("rough_rock", 0.8), ("water", 0.2)), _s("rough_rock"),
              _s("rough_rock"), _s("rough_rock"), _s("rough_rock"), _s(("rough_rock", 0.9), ("open", 0.1)),
              [], clutter=0.6, notes=["Irregular cave approximated as shoebox with very high scattering",
                                      "Tunnel opening at one end"]),
        Placement(source=(8, 10, 1.7), listener=(10, 16, 1.7))),
    "school_classroom": (
        _room("School classroom", "indoor", 7, 9, 3.0,
              _s("linoleum"), _s("acoustic_tile"),
              _s(("plaster", 0.6), ("glass_window", 0.4)), _s("plaster"),
              _s(("plaster", 0.7), ("wood_panelling", 0.3)), _s(("plaster", 0.8), ("gypsum_board", 0.2)),
              [_o("desk", 15), _o("wooden_chair", 30), _o("bookshelf_full", 2)], clutter=0.3),
        Placement(source=(3.5, 1.5, 1.7), listener=(3.5, 6.5, 1.2))),
    "outdoor_forest": (
        _room("Outdoor forest", "outdoor", 40, 40, 15,
              _s("forest_floor"), _s("open"),
              _s("foliage_boundary"), _s("foliage_boundary"), _s("foliage_boundary"), _s("foliage_boundary"),
              [], clutter=0.6, notes=["Outdoors: shoebox is a virtual domain; sky is fully absorbing",
                                      "Tree trunks / foliage modelled as a lossy, highly scattering boundary"]),
        Placement(source=(18, 18, 1.7), listener=(21, 21, 1.7))),
    "public_bathroom": (
        _room("Public bathroom", "indoor", 4, 7, 2.8,
              _s("ceramic_tile"), _s(("concrete_painted", 0.85), ("acoustic_tile", 0.15)),
              _s("ceramic_tile"), _s(("ceramic_tile", 0.6), ("glass_window", 0.1), ("metal_sheet", 0.3)),
              _s("ceramic_tile"), _s("ceramic_tile"),
              [], clutter=0.2, notes=["Cubicle partitions as metal on one wall", "Painted ceiling with a few vent grilles / ceiling tiles"]),
        Placement(source=(1.2, 2.0, 1.7), listener=(2.8, 5.0, 1.7))),
    "recording_studio": (
        _room("Recording studio", "indoor", 5, 6, 3.0,
              _s("carpet_thin"), _s("acoustic_treatment"),
              _s(("acoustic_treatment", 0.7), ("gypsum_board", 0.3)), _s(("acoustic_treatment", 0.7), ("gypsum_board", 0.3)),
              _s(("acoustic_treatment", 0.6), ("glass_window", 0.4)), _s("acoustic_treatment"),
              [_o("upholstered_chair", 1)], clutter=0.3),
        Placement(source=(2.5, 2.5, 1.7), listener=(2.5, 3.3, 1.7))),
    "royal_throne_room": (
        _room("Royal throne room", "indoor", 14, 30, 10,
              _s(("marble", 0.6), ("carpet_thick", 0.4)), _s("wood_panelling"),
              _s(("stone_masonry", 0.7), ("curtains_heavy", 0.3)), _s(("stone_masonry", 0.7), ("glass_window", 0.3)),
              _s("stone_masonry"), _s(("stone_masonry", 0.8), ("wood_panelling", 0.2)),
              [_o("tapestry", 60), _o("upholstered_chair", 4), _o("person", 6)], clutter=0.25),
        Placement(source=(7, 24, 1.8), listener=(7, 12, 1.7))),
}
