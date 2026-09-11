from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Mapping

import meshio
import numpy as np


_HEX_NATURAL_COORDINATES = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ]
)


@dataclass(frozen=True)
class ElementQuality:
    signed_measure: float
    minimum_scaled_jacobian: float


@dataclass(frozen=True)
class VolumeQualityReport:
    element_count: int
    inverted_count: int
    degenerate_count: int
    minimum_signed_measure: float | None
    minimum_scaled_jacobian: float | None
    total_signed_measure: float
    counts_by_type: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _scaled_jacobian(matrix: np.ndarray) -> float:
    denominator = float(np.prod(np.linalg.norm(matrix, axis=0)))
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.linalg.det(matrix) / denominator)


def _tetra_quality(coordinates: np.ndarray) -> ElementQuality:
    jacobian = np.column_stack(
        (
            coordinates[1] - coordinates[0],
            coordinates[2] - coordinates[0],
            coordinates[3] - coordinates[0],
        )
    )
    determinant = float(np.linalg.det(jacobian))
    return ElementQuality(determinant / 6.0, _scaled_jacobian(jacobian))


def _hex_jacobian(coordinates: np.ndarray, natural: np.ndarray) -> np.ndarray:
    xi, eta, zeta = natural
    sx = _HEX_NATURAL_COORDINATES[:, 0]
    sy = _HEX_NATURAL_COORDINATES[:, 1]
    sz = _HEX_NATURAL_COORDINATES[:, 2]
    derivatives = np.column_stack(
        (
            sx * (1.0 + sy * eta) * (1.0 + sz * zeta) / 8.0,
            sy * (1.0 + sx * xi) * (1.0 + sz * zeta) / 8.0,
            sz * (1.0 + sx * xi) * (1.0 + sy * eta) / 8.0,
        )
    )
    return coordinates.T @ derivatives


def _hexahedron_quality(coordinates: np.ndarray) -> ElementQuality:
    scaled = [
        _scaled_jacobian(_hex_jacobian(coordinates, natural))
        for natural in _HEX_NATURAL_COORDINATES
    ]
    gauss = 1.0 / sqrt(3.0)
    signed_measure = 0.0
    for xi in (-gauss, gauss):
        for eta in (-gauss, gauss):
            for zeta in (-gauss, gauss):
                signed_measure += float(
                    np.linalg.det(_hex_jacobian(coordinates, np.array([xi, eta, zeta])))
                )
    return ElementQuality(signed_measure, min(scaled))


def _wedge_jacobian(coordinates: np.ndarray, natural: np.ndarray) -> np.ndarray:
    r, s, t = natural
    lower = (1.0 - t) / 2.0
    upper = (1.0 + t) / 2.0
    derivatives = np.array(
        [
            [-lower, -lower, -(1.0 - r - s) / 2.0],
            [lower, 0.0, -r / 2.0],
            [0.0, lower, -s / 2.0],
            [-upper, -upper, (1.0 - r - s) / 2.0],
            [upper, 0.0, r / 2.0],
            [0.0, upper, s / 2.0],
        ]
    )
    return coordinates.T @ derivatives


def _wedge_quality(coordinates: np.ndarray) -> ElementQuality:
    corners = (
        (0.0, 0.0, -1.0),
        (1.0, 0.0, -1.0),
        (0.0, 1.0, -1.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
    )
    scaled = [
        _scaled_jacobian(_wedge_jacobian(coordinates, np.asarray(natural)))
        for natural in corners
    ]
    triangle_points = ((1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0))
    gauss = 1.0 / sqrt(3.0)
    signed_measure = 0.0
    for r, s in triangle_points:
        for t in (-gauss, gauss):
            signed_measure += float(
                np.linalg.det(_wedge_jacobian(coordinates, np.array([r, s, t])))
            ) / 6.0
    return ElementQuality(signed_measure, min(scaled))


def evaluate_element_quality(cell_type: str, coordinates: np.ndarray) -> ElementQuality:
    values = np.asarray(coordinates, dtype=float)
    if cell_type == "tetra" and values.shape == (4, 3):
        return _tetra_quality(values)
    if cell_type == "hexahedron" and values.shape == (8, 3):
        return _hexahedron_quality(values)
    if cell_type == "wedge" and values.shape == (6, 3):
        return _wedge_quality(values)
    raise ValueError(f"unsupported 3D quality cell type or shape: {cell_type} {values.shape}")


def _element_indices(count: int, selection: np.ndarray | None) -> np.ndarray:
    if selection is None:
        return np.arange(count, dtype=int)
    values = np.asarray(selection)
    if values.dtype == bool:
        if values.shape != (count,):
            raise ValueError("element selection mask has an invalid shape")
        return np.flatnonzero(values)
    return values.astype(int, copy=False).reshape(-1)


def element_qualities(
    mesh: meshio.Mesh,
    selections: Mapping[int, np.ndarray] | None = None,
) -> dict[tuple[int, int], ElementQuality]:
    result: dict[tuple[int, int], ElementQuality] = {}
    for block_index, block in enumerate(mesh.cells):
        if block.type not in {"tetra", "hexahedron", "wedge"}:
            continue
        if selections is not None and block_index not in selections:
            continue
        selected = None if selections is None else selections[block_index]
        for element_index in _element_indices(len(block.data), selected):
            coordinates = np.asarray(mesh.points, dtype=float)[block.data[element_index], :3]
            result[(block_index, int(element_index))] = evaluate_element_quality(
                block.type, coordinates
            )
    return result


def evaluate_3d_quality(
    mesh: meshio.Mesh,
    selections: Mapping[int, np.ndarray] | None = None,
    *,
    zero_tolerance: float = 1.0e-15,
) -> VolumeQualityReport:
    qualities = element_qualities(mesh, selections)
    counts: dict[str, int] = {}
    for block_index, _ in qualities:
        cell_type = mesh.cells[block_index].type
        counts[cell_type] = counts.get(cell_type, 0) + 1
    if not qualities:
        return VolumeQualityReport(0, 0, 0, None, None, 0.0, counts)
    measures = np.array([quality.signed_measure for quality in qualities.values()])
    scaled = np.array([quality.minimum_scaled_jacobian for quality in qualities.values()])
    return VolumeQualityReport(
        element_count=len(qualities),
        inverted_count=int(np.count_nonzero(scaled < -zero_tolerance)),
        degenerate_count=int(np.count_nonzero(np.abs(scaled) <= zero_tolerance)),
        minimum_signed_measure=float(np.min(measures)),
        minimum_scaled_jacobian=float(np.min(scaled)),
        total_signed_measure=float(np.sum(measures)),
        counts_by_type=counts,
    )


def minimum_measure_ratio(
    before: meshio.Mesh,
    after: meshio.Mesh,
    selections: Mapping[int, np.ndarray],
) -> float:
    before_values = element_qualities(before, selections)
    after_values = element_qualities(after, selections)
    if before_values.keys() != after_values.keys():
        raise ValueError("before and after element selections do not match")
    ratios = []
    for key, quality_before in before_values.items():
        denominator = abs(quality_before.signed_measure)
        if denominator <= np.finfo(float).eps:
            ratios.append(0.0)
        else:
            ratios.append(abs(after_values[key].signed_measure) / denominator)
    return min(ratios, default=1.0)
