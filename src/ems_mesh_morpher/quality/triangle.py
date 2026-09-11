from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import meshio
import numpy as np


@dataclass(frozen=True)
class QualityReport:
    element_count: int
    inverted_count: int
    min_signed_area: float | None
    min_abs_area: float | None
    min_angle_deg: float | None
    max_aspect_ratio: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _polygon_signed_area(coords: np.ndarray) -> float:
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _corner_angles_deg(coords: np.ndarray) -> list[float]:
    angles = []
    count = len(coords)
    for i in range(count):
        prev_point = coords[(i - 1) % count]
        point = coords[i]
        next_point = coords[(i + 1) % count]
        a = prev_point[:2] - point[:2]
        b = next_point[:2] - point[:2]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            angles.append(0.0)
            continue
        cos_theta = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
        angles.append(math.degrees(math.acos(cos_theta)))
    return angles


def _aspect_ratio(coords: np.ndarray) -> float:
    edges = np.linalg.norm(np.roll(coords[:, :2], -1, axis=0) - coords[:, :2], axis=1)
    min_edge = float(np.min(edges))
    if min_edge == 0.0:
        return math.inf
    return float(np.max(edges) / min_edge)


def evaluate_2d_quality(mesh: meshio.Mesh) -> QualityReport:
    signed_areas: list[float] = []
    min_angles: list[float] = []
    aspect_ratios: list[float] = []

    for block in mesh.cells:
        if block.type not in {"triangle", "quad"}:
            continue
        for element in block.data:
            coords = mesh.points[np.asarray(element)]
            signed_areas.append(_polygon_signed_area(coords))
            min_angles.append(min(_corner_angles_deg(coords)))
            aspect_ratios.append(_aspect_ratio(coords))

    if not signed_areas:
        return QualityReport(
            element_count=0,
            inverted_count=0,
            min_signed_area=None,
            min_abs_area=None,
            min_angle_deg=None,
            max_aspect_ratio=None,
        )

    areas = np.asarray(signed_areas, dtype=float)
    return QualityReport(
        element_count=int(len(areas)),
        inverted_count=int(np.count_nonzero(areas <= 0.0)),
        min_signed_area=float(np.min(areas)),
        min_abs_area=float(np.min(np.abs(areas))),
        min_angle_deg=float(np.min(min_angles)),
        max_aspect_ratio=float(np.max(aspect_ratios)),
    )


def _signed_area_array(mesh: meshio.Mesh) -> np.ndarray:
    signed_areas: list[float] = []
    for block in mesh.cells:
        if block.type not in {"triangle", "quad"}:
            continue
        for element in block.data:
            signed_areas.append(_polygon_signed_area(mesh.points[np.asarray(element)]))
    return np.asarray(signed_areas, dtype=float)


def count_orientation_flips(before: meshio.Mesh, after: meshio.Mesh) -> int:
    before_areas = _signed_area_array(before)
    after_areas = _signed_area_array(after)
    if before_areas.shape != after_areas.shape:
        raise ValueError("before and after meshes have different 2D element counts")

    before_sign = np.sign(before_areas)
    after_sign = np.sign(after_areas)
    nonzero_before = before_sign != 0.0
    return int(np.count_nonzero(nonzero_before & (before_sign != after_sign)))
