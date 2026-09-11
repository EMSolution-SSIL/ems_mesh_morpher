from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import meshio
import numpy as np

from ems_mesh_morpher.skin_layer.quality import evaluate_element_quality


_SUPPORTED_CELL_TYPES = {
    "triangle",
    "quad",
    "tetra",
    "pyramid",
    "wedge",
    "hexahedron",
}


@dataclass(frozen=True)
class MeshQualityReport:
    element_count: int
    orientation_flip_count: int
    degenerate_count: int
    minimum_measure_ratio: float | None
    minimum_scaled_jacobian: float | None
    counts_by_type: dict[str, int]
    unsupported_counts_by_type: dict[str, int]
    worst_element: tuple[int, int] | None

    @property
    def inverted_count(self) -> int:
        return self.orientation_flip_count

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QualityGate:
    reject_orientation_flips: bool = True
    reject_degenerate_elements: bool = True
    minimum_measure_ratio: float = 0.0
    minimum_scaled_jacobian: float | None = None
    absolute_tolerance: float = 1.0e-15

    def __post_init__(self) -> None:
        if self.minimum_measure_ratio < 0.0 or not np.isfinite(
            self.minimum_measure_ratio
        ):
            raise ValueError("minimum_measure_ratio must be non-negative and finite")
        if self.minimum_scaled_jacobian is not None and not np.isfinite(
            self.minimum_scaled_jacobian
        ):
            raise ValueError("minimum_scaled_jacobian must be finite or None")
        if self.absolute_tolerance < 0.0 or not np.isfinite(self.absolute_tolerance):
            raise ValueError("absolute_tolerance must be non-negative and finite")

    def rejection_reason(self, report: MeshQualityReport) -> str | None:
        if self.reject_orientation_flips and report.orientation_flip_count:
            return f"{report.orientation_flip_count} element orientation flips"
        if self.reject_degenerate_elements and report.degenerate_count:
            return f"{report.degenerate_count} degenerate elements"
        if (
            report.minimum_measure_ratio is not None
            and report.minimum_measure_ratio < self.minimum_measure_ratio
        ):
            return (
                f"minimum measure ratio {report.minimum_measure_ratio:.6g} is below "
                f"{self.minimum_measure_ratio:.6g}"
            )
        if (
            self.minimum_scaled_jacobian is not None
            and report.minimum_scaled_jacobian is not None
            and report.minimum_scaled_jacobian < self.minimum_scaled_jacobian
        ):
            return (
                f"minimum scaled Jacobian {report.minimum_scaled_jacobian:.6g} "
                f"is below {self.minimum_scaled_jacobian:.6g}"
            )
        return None

    def accepts(self, report: MeshQualityReport) -> bool:
        return self.rejection_reason(report) is None


@dataclass(frozen=True)
class QualityLimitedDisplacement:
    displacement: np.ndarray
    scale: float
    report: MeshQualityReport
    iterations: int


def _polygon_signed_area(coordinates: np.ndarray) -> float:
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_scaled_jacobian(coordinates: np.ndarray) -> float:
    values: list[float] = []
    xy = coordinates[:, :2]
    for index in range(len(xy)):
        outgoing = xy[(index + 1) % len(xy)] - xy[index]
        incoming = xy[(index - 1) % len(xy)] - xy[index]
        denominator = float(np.linalg.norm(outgoing) * np.linalg.norm(incoming))
        if denominator <= np.finfo(float).eps:
            values.append(0.0)
        else:
            values.append(float(np.linalg.det(np.vstack((outgoing, incoming)))) / denominator)
    return min(values)


def _tetra_signed_measure(coordinates: np.ndarray) -> float:
    return float(
        np.linalg.det(
            np.column_stack(
                (
                    coordinates[1] - coordinates[0],
                    coordinates[2] - coordinates[0],
                    coordinates[3] - coordinates[0],
                )
            )
        )
        / 6.0
    )


def _element_metrics(cell_type: str, coordinates: np.ndarray) -> tuple[float, float | None]:
    if cell_type in {"triangle", "quad"}:
        return _polygon_signed_area(coordinates), _polygon_scaled_jacobian(coordinates)
    if cell_type in {"tetra", "wedge", "hexahedron"}:
        quality = evaluate_element_quality(cell_type, coordinates[:, :3])
        return quality.signed_measure, quality.minimum_scaled_jacobian
    if cell_type == "pyramid":
        first = coordinates[[0, 1, 2, 4], :3]
        second = coordinates[[0, 2, 3, 4], :3]
        measures = (_tetra_signed_measure(first), _tetra_signed_measure(second))
        scaled = (
            evaluate_element_quality("tetra", first).minimum_scaled_jacobian,
            evaluate_element_quality("tetra", second).minimum_scaled_jacobian,
        )
        return sum(measures), min(scaled)
    raise ValueError(f"unsupported quality cell type {cell_type!r}")


def evaluate_mesh_quality(
    reference_mesh: meshio.Mesh,
    target: meshio.Mesh | np.ndarray,
    *,
    absolute_tolerance: float = 1.0e-15,
    cell_types: Iterable[str] | None = None,
) -> MeshQualityReport:
    reference_points = np.asarray(reference_mesh.points, dtype=float)
    target_points = np.asarray(target.points if isinstance(target, meshio.Mesh) else target, dtype=float)
    if target_points.shape != reference_points.shape:
        raise ValueError("target point shape does not match the reference mesh")
    if not np.all(np.isfinite(target_points)):
        raise ValueError("target points must contain only finite values")
    selected_types = None if cell_types is None else set(cell_types)

    ratios: list[float] = []
    scaled_jacobians: list[float] = []
    counts: dict[str, int] = {}
    unsupported: dict[str, int] = {}
    flips = 0
    degenerates = 0
    worst: tuple[int, int] | None = None
    worst_ratio = np.inf

    for block_index, block in enumerate(reference_mesh.cells):
        if selected_types is not None and block.type not in selected_types:
            continue
        if block.type not in _SUPPORTED_CELL_TYPES:
            unsupported[block.type] = unsupported.get(block.type, 0) + len(block.data)
            continue
        counts[block.type] = counts.get(block.type, 0) + len(block.data)
        for element_index, connectivity in enumerate(np.asarray(block.data, dtype=int)):
            before, _ = _element_metrics(block.type, reference_points[connectivity])
            after, scaled = _element_metrics(block.type, target_points[connectivity])
            if abs(before) <= absolute_tolerance:
                raise ValueError(
                    f"reference element ({block_index}, {element_index}) is degenerate"
                )
            ratio = abs(after) / abs(before)
            ratios.append(ratio)
            if ratio < worst_ratio:
                worst_ratio = ratio
                worst = (block_index, element_index)
            if before * after < 0.0:
                flips += 1
            if abs(after) <= absolute_tolerance:
                degenerates += 1
            if scaled is not None:
                scaled_jacobians.append(float(scaled))

    return MeshQualityReport(
        element_count=len(ratios),
        orientation_flip_count=flips,
        degenerate_count=degenerates,
        minimum_measure_ratio=min(ratios, default=None),
        minimum_scaled_jacobian=min(scaled_jacobians, default=None),
        counts_by_type=counts,
        unsupported_counts_by_type=unsupported,
        worst_element=worst,
    )


def limit_displacement_by_quality(
    reference_mesh: meshio.Mesh,
    displacement: np.ndarray,
    gate: QualityGate,
    *,
    minimum_scale: float = 0.0,
    max_iterations: int = 32,
    scale_tolerance: float = 1.0e-6,
) -> QualityLimitedDisplacement:
    field = np.asarray(displacement, dtype=float)
    points = np.asarray(reference_mesh.points, dtype=float)
    if field.shape != points.shape or not np.all(np.isfinite(field)):
        raise ValueError("displacement must be a finite array matching mesh.points")
    if not 0.0 <= minimum_scale <= 1.0:
        raise ValueError("minimum_scale must be between zero and one")
    if max_iterations <= 0 or scale_tolerance <= 0.0:
        raise ValueError("max_iterations and scale_tolerance must be positive")

    full_report = evaluate_mesh_quality(
        reference_mesh,
        points + field,
        absolute_tolerance=gate.absolute_tolerance,
    )
    if gate.accepts(full_report):
        return QualityLimitedDisplacement(field.copy(), 1.0, full_report, 0)

    zero_report = evaluate_mesh_quality(
        reference_mesh,
        points,
        absolute_tolerance=gate.absolute_tolerance,
    )
    if not gate.accepts(zero_report):
        reason = gate.rejection_reason(zero_report)
        raise ValueError(f"reference mesh does not satisfy the quality gate: {reason}")

    lower = 0.0
    upper = 1.0
    accepted_report = zero_report
    iterations = 0
    while iterations < max_iterations and upper - lower > scale_tolerance:
        candidate = 0.5 * (lower + upper)
        report = evaluate_mesh_quality(
            reference_mesh,
            points + candidate * field,
            absolute_tolerance=gate.absolute_tolerance,
        )
        if gate.accepts(report):
            lower = candidate
            accepted_report = report
        else:
            upper = candidate
        iterations += 1

    if lower < minimum_scale:
        raise ValueError(
            f"quality-limited displacement scale {lower:.6g} is below "
            f"minimum_scale {minimum_scale:.6g}"
        )
    return QualityLimitedDisplacement(
        displacement=lower * field,
        scale=lower,
        report=accepted_report,
        iterations=iterations,
    )
