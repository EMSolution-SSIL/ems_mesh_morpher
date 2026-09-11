from __future__ import annotations

from dataclasses import dataclass

import meshio
import numpy as np

from ems_mesh_morpher.quality import MeshQualityReport, evaluate_mesh_quality


@dataclass(frozen=True)
class MorphingQualityConfig:
    enabled: bool = True
    reject_orientation_flips: bool = True
    reject_degenerate_elements: bool = True
    minimum_measure_ratio: float = 1.0e-6
    absolute_tolerance: float = 1.0e-15

    def __post_init__(self) -> None:
        if self.minimum_measure_ratio < 0.0 or not np.isfinite(
            self.minimum_measure_ratio
        ):
            raise ValueError("minimum_measure_ratio must be non-negative and finite")
        if self.absolute_tolerance < 0.0 or not np.isfinite(self.absolute_tolerance):
            raise ValueError("absolute_tolerance must be non-negative and finite")


MorphingQualityReport = MeshQualityReport


class MorphingQualityError(ValueError):
    def __init__(self, deform_id: int, report: MorphingQualityReport, reason: str):
        self.deform_id = deform_id
        self.report = report
        super().__init__(f"DEFORM {deform_id} mesh quality rejected: {reason}")


def evaluate_relative_quality(
    reference_mesh: meshio.Mesh,
    target_points: np.ndarray,
    *,
    absolute_tolerance: float = 1.0e-15,
) -> MorphingQualityReport:
    return evaluate_mesh_quality(
        reference_mesh,
        target_points,
        absolute_tolerance=absolute_tolerance,
    )


def enforce_quality(
    deform_id: int,
    reference_mesh: meshio.Mesh,
    target_points: np.ndarray,
    config: MorphingQualityConfig,
) -> MorphingQualityReport | None:
    if not config.enabled:
        return None
    report = evaluate_relative_quality(
        reference_mesh,
        target_points,
        absolute_tolerance=config.absolute_tolerance,
    )
    if config.reject_orientation_flips and report.orientation_flip_count:
        raise MorphingQualityError(
            deform_id, report, f"{report.orientation_flip_count} orientation flips"
        )
    if config.reject_degenerate_elements and report.degenerate_count:
        raise MorphingQualityError(
            deform_id, report, f"{report.degenerate_count} degenerate elements"
        )
    if (
        report.minimum_measure_ratio is not None
        and report.minimum_measure_ratio < config.minimum_measure_ratio
    ):
        raise MorphingQualityError(
            deform_id,
            report,
            f"minimum measure ratio {report.minimum_measure_ratio:.6g} is below "
            f"{config.minimum_measure_ratio:.6g}",
        )
    return report
