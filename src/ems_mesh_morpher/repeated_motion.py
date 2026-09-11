from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence

import meshio
import numpy as np

from .config import MorphConfig, MotionConfig
from .mesh_motion import MeshMotionResult, apply_mesh_motion
from .morph import LaplaceConfig, PreparedLaplaceInterpolator, prepare_laplace_interpolator
from .selection import classify_nodes_from_regions


@dataclass(frozen=True)
class MotionSchedule:
    times: tuple[float, ...]
    motions: tuple[MotionConfig, ...]

    def __post_init__(self) -> None:
        if not self.times:
            raise ValueError("motion schedule must contain at least one frame")
        if len(self.times) != len(self.motions):
            raise ValueError("motion schedule times and motions must have equal length")
        values = np.asarray(self.times, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("motion schedule times must be finite")
        if np.any(np.diff(values) < 0.0):
            raise ValueError("motion schedule times must be non-decreasing")

    @classmethod
    def translations(
        cls,
        times: Iterable[float],
        displacements: Iterable[Sequence[float]],
    ) -> MotionSchedule:
        return cls(
            times=tuple(float(value) for value in times),
            motions=tuple(
                MotionConfig(
                    type="translation",
                    displacement=tuple(float(value) for value in displacement),
                )
                for displacement in displacements
            ),
        )

    @classmethod
    def rotations(
        cls,
        times: Iterable[float],
        angles_degrees: Iterable[float],
        *,
        center: Sequence[float],
        axis: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> MotionSchedule:
        return cls(
            times=tuple(float(value) for value in times),
            motions=tuple(
                MotionConfig(
                    type="rotation",
                    center=tuple(float(value) for value in center),
                    axis=tuple(float(value) for value in axis),
                    angle_degrees=float(angle),
                )
                for angle in angles_degrees
            ),
        )


@dataclass(frozen=True)
class RepeatedMotionFrame:
    index: int
    time: float
    motion: MotionConfig
    result: MeshMotionResult
    elapsed_seconds: float

    def summary(self) -> dict[str, object]:
        return {
            "index": self.index,
            "time": self.time,
            "motion_type": self.motion.type,
            "elapsed_seconds": self.elapsed_seconds,
            **self.result.summary(),
        }


@dataclass(frozen=True)
class RepeatedMotionResult:
    frames: tuple[RepeatedMotionFrame, ...]
    method: str
    reused_laplace_factorization: bool
    preparation_seconds: float
    total_seconds: float

    def summary(self) -> dict[str, object]:
        quality_values = [
            frame.result.relative_quality.minimum_measure_ratio
            for frame in self.frames
            if frame.result.relative_quality is not None
            and frame.result.relative_quality.minimum_measure_ratio is not None
        ]
        return {
            "method": self.method,
            "frame_count": len(self.frames),
            "reused_laplace_factorization": self.reused_laplace_factorization,
            "preparation_seconds": self.preparation_seconds,
            "total_seconds": self.total_seconds,
            "minimum_measure_ratio": min(quality_values, default=None),
            "minimum_applied_displacement_scale": min(
                (frame.result.applied_displacement_scale for frame in self.frames),
                default=1.0,
            ),
            "frames": [frame.summary() for frame in self.frames],
        }


class RepeatedMotionRunner:
    def __init__(self, reference_mesh: meshio.Mesh, config: MorphConfig):
        self.reference_mesh = copy.deepcopy(reference_mesh)
        self.config = config
        start = perf_counter()
        self._prepared_laplace = self._prepare_laplace()
        self.preparation_seconds = perf_counter() - start

    def _prepare_laplace(self) -> PreparedLaplaceInterpolator | None:
        if self.config.morphing.type not in {"laplace", "weighted_laplace"}:
            return None
        if self.config.motion.type == "prescribed":
            return None
        classification = classify_nodes_from_regions(
            self.reference_mesh, self.config.regions
        )
        controls = classification.moving | classification.fixed
        queries = classification.deformable
        if not np.any(queries) or not np.any(controls):
            return None
        morphing = self.config.morphing
        return prepare_laplace_interpolator(
            self.reference_mesh,
            np.flatnonzero(queries),
            np.flatnonzero(controls),
            LaplaceConfig(
                weighting=morphing.weighting,
                distance_power=morphing.distance_power,
                minimum_distance=morphing.minimum_distance,
                element_size_power=morphing.element_size_power,
                element_aspect_power=morphing.element_aspect_power,
            ),
        )

    def run(
        self,
        schedule: MotionSchedule,
        *,
        frame_callback: Callable[[RepeatedMotionFrame], None] | None = None,
    ) -> RepeatedMotionResult:
        start = perf_counter()
        frames: list[RepeatedMotionFrame] = []
        for index, (time, motion) in enumerate(
            zip(schedule.times, schedule.motions, strict=True)
        ):
            frame_start = perf_counter()
            result = apply_mesh_motion(
                self.reference_mesh,
                replace(self.config, motion=motion),
                prepared_laplace=self._prepared_laplace,
            )
            frame = RepeatedMotionFrame(
                index=index,
                time=time,
                motion=motion,
                result=result,
                elapsed_seconds=perf_counter() - frame_start,
            )
            frames.append(frame)
            if frame_callback is not None:
                frame_callback(frame)
        return RepeatedMotionResult(
            frames=tuple(frames),
            method=self.config.morphing.type,
            reused_laplace_factorization=self._prepared_laplace is not None,
            preparation_seconds=self.preparation_seconds,
            total_seconds=perf_counter() - start,
        )


def benchmark_repeated_motion(
    reference_mesh: meshio.Mesh,
    schedule: MotionSchedule,
    configurations: Mapping[str, MorphConfig],
) -> dict[str, dict[str, object]]:
    if not configurations:
        raise ValueError("benchmark requires at least one configuration")
    return {
        name: RepeatedMotionRunner(reference_mesh, config).run(schedule).summary()
        for name, config in configurations.items()
    }
