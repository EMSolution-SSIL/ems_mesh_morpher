from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ems_mesh_morpher.morph import (
    IDWConfig,
    LaplaceConfig,
    RBFConfig,
    build_idw_displacement_field,
    build_laplace_displacement_field,
    build_rbf_displacement_field,
)

from .quality import MorphingQualityConfig, MorphingQualityReport, enforce_quality
from .schema import (
    DeformationDescription,
    DeformationRegionDescription,
    DeformationTopology,
    DeformationTopologyRegion,
    parse_deformation_description,
    parse_deformation_topology,
    validate_deformation_payload,
)


_METHODS = frozenset({"linear", "idw", "rbf", "weighted_laplace"})


@dataclass(frozen=True)
class PyEMSolMorphingResult:
    payload: dict[str, object]
    quality_reports: Mapping[int, MorphingQualityReport | None]


@dataclass(frozen=True)
class PyEMSolConstraintConfig:
    infer_zero_displacement_anchors: bool = True
    zero_displacement_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.zero_displacement_tolerance < 0.0 or not np.isfinite(
            self.zero_displacement_tolerance
        ):
            raise ValueError(
                "zero_displacement_tolerance must be non-negative and finite"
            )


@dataclass(frozen=True)
class PyEMSolConstraintSummary:
    rigid_count: int
    explicit_fixed_count: int
    inferred_fixed_count: int
    deformable_query_count: int
    inferred_fixed_node_ids: tuple[int, ...]


def _mesh_revision(value: object) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str, np.integer)):
        raise ValueError("mesh_revision must be an int or str")
    return int(value) if isinstance(value, np.integer) else value


def _positions_by_region(
    positions: float | Mapping[int, float] | None,
    description: DeformationDescription,
    overridden_regions: set[int],
) -> dict[int, float]:
    def finite_position(value: object) -> float:
        if isinstance(value, bool):
            raise ValueError("positions must contain finite numbers, not booleans")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("positions must contain only finite numbers") from exc
        if not np.isfinite(result):
            raise ValueError("positions must contain only finite numbers")
        return result

    if positions is None:
        result: dict[int, float] = {}
    elif isinstance(positions, Mapping):
        result = {int(key): finite_position(value) for key, value in positions.items()}
    elif len(description.regions) == 1:
        result = {
            description.regions[0].deform_id: finite_position(positions)
        }
    else:
        raise ValueError("multiple DEFORM regions require a position mapping")
    known = {region.deform_id for region in description.regions}
    if set(result) - known:
        raise ValueError("positions contains an unknown deform_id")
    missing = known - set(result) - overridden_regions
    if missing:
        raise ValueError(f"positions is missing deform_id values {sorted(missing)}")
    return result


class PyEMSolMorphingAdapter:
    """Prepare reference-based mesh morphing payloads for pyemsol.

    The adapter has no runtime dependency on pyemsol.  It consumes the dictionaries
    returned by ``describe_deformation`` and ``describe_deformation_mesh`` and emits
    a dictionary accepted by ``CoupledSession.step(deformation=...)``.
    """

    def __init__(
        self,
        description: DeformationDescription,
        topology: DeformationTopology | None,
        *,
        method: str,
        idw_config: IDWConfig,
        rbf_config: RBFConfig,
        laplace_config: LaplaceConfig,
        quality_config: MorphingQualityConfig,
        constraint_config: PyEMSolConstraintConfig,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {sorted(_METHODS)}")
        if method != "linear" and topology is None:
            raise ValueError(f"{method} morphing requires schema version 2 topology")
        self.description = description
        self.topology = topology
        self.method = method
        self.idw_config = idw_config
        self.rbf_config = rbf_config
        self.laplace_config = laplace_config
        self.quality_config = quality_config
        self.constraint_config = constraint_config
        self._endpoint_displacements: dict[int, np.ndarray] = {}
        self._constraint_indices_by_region: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self.constraint_summaries: dict[int, PyEMSolConstraintSummary] = {}

    @classmethod
    def prepare(
        cls,
        description: Mapping[str, object] | DeformationDescription,
        *,
        topology: Mapping[str, object] | DeformationTopology | None = None,
        method: str = "linear",
        idw_config: IDWConfig | None = None,
        rbf_config: RBFConfig | None = None,
        laplace_config: LaplaceConfig | None = None,
        quality_config: MorphingQualityConfig | None = None,
        constraint_config: PyEMSolConstraintConfig | None = None,
        coordinate_tolerance: float = 1.0e-12,
    ) -> "PyEMSolMorphingAdapter":
        parsed_description = (
            parse_deformation_description(description)
            if isinstance(description, Mapping)
            else description
        )
        parsed_topology = (
            parse_deformation_topology(
                topology,
                parsed_description,
                coordinate_tolerance=coordinate_tolerance,
            )
            if isinstance(topology, Mapping)
            else topology
        )
        return cls(
            parsed_description,
            parsed_topology,
            method=method,
            idw_config=idw_config or IDWConfig(),
            rbf_config=rbf_config or RBFConfig(),
            laplace_config=laplace_config or LaplaceConfig(),
            quality_config=quality_config or MorphingQualityConfig(),
            constraint_config=constraint_config or PyEMSolConstraintConfig(),
        )

    def _constraint_indices(
        self,
        description_region: DeformationRegionDescription,
        topology_region: DeformationTopologyRegion,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._constraint_indices_by_region.get(description_region.deform_id)
        if cached is not None:
            return cached
        rigid = topology_region.indices_for_role("rigid")
        explicit_fixed = topology_region.indices_for_role("fixed")
        deformable = topology_region.indices_for_role("deformable")
        inferred_fixed = np.empty(0, dtype=int)
        if self.constraint_config.infer_zero_displacement_anchors:
            description_index = {
                int(node_id): index
                for index, node_id in enumerate(description_region.node_ids)
            }
            candidates: list[int] = []
            for topology_index in deformable:
                node_id = int(topology_region.node_ids[topology_index])
                description_index_value = description_index.get(node_id)
                if description_index_value is None:
                    continue
                displacement = (
                    description_region.target_coordinates[description_index_value]
                    - description_region.reference_coordinates[description_index_value]
                )
                if (
                    np.linalg.norm(displacement)
                    <= self.constraint_config.zero_displacement_tolerance
                ):
                    candidates.append(int(topology_index))
            inferred_fixed = np.asarray(candidates, dtype=int)
        fixed = np.unique(np.concatenate((explicit_fixed, inferred_fixed)))
        queries = np.setdiff1d(deformable, inferred_fixed, assume_unique=True)
        result = (rigid, fixed, queries)
        self._constraint_indices_by_region[description_region.deform_id] = result
        inferred_ids = tuple(
            int(topology_region.node_ids[index]) for index in inferred_fixed
        )
        self.constraint_summaries[description_region.deform_id] = (
            PyEMSolConstraintSummary(
                rigid_count=len(rigid),
                explicit_fixed_count=len(explicit_fixed),
                inferred_fixed_count=len(inferred_fixed),
                deformable_query_count=len(queries),
                inferred_fixed_node_ids=inferred_ids,
            )
        )
        return result

    def _description_targets_for_ids(
        self,
        description_region: DeformationRegionDescription,
        node_ids: np.ndarray,
    ) -> np.ndarray:
        index_by_id = {
            int(node_id): index
            for index, node_id in enumerate(description_region.node_ids)
        }
        try:
            indices = [index_by_id[int(node_id)] for node_id in node_ids]
        except KeyError as exc:
            raise ValueError("a rigid node is absent from the schema v1 description") from exc
        return description_region.target_coordinates[np.asarray(indices, dtype=int)]

    def _solve_target_points(
        self,
        description_region: DeformationRegionDescription,
        topology_region: DeformationTopologyRegion,
        control_targets: np.ndarray,
    ) -> np.ndarray:
        rigid, fixed, deformable = self._constraint_indices(
            description_region, topology_region
        )
        if not len(rigid):
            raise ValueError(
                f"DEFORM {topology_region.deform_id} has no rigid control nodes"
            )
        if control_targets.shape != (len(rigid), 3):
            raise ValueError("rigid target coordinates have an invalid shape")
        kwargs = dict(
            control_target_points=control_targets,
            fixed_indices=fixed,
            query_indices=deformable,
        )
        if self.method == "idw":
            field = build_idw_displacement_field(
                topology_region.mesh.points,
                rigid,
                config=self.idw_config,
                **kwargs,
            )
        elif self.method == "rbf":
            field = build_rbf_displacement_field(
                topology_region.mesh.points,
                rigid,
                config=self.rbf_config,
                **kwargs,
            )
        elif self.method == "weighted_laplace":
            field = build_laplace_displacement_field(
                topology_region.mesh,
                rigid,
                config=self.laplace_config,
                **kwargs,
            )
        else:
            raise RuntimeError("nonlinear target solve called for the linear method")
        return field.target_points

    def _endpoint_displacement(
        self,
        description_region: DeformationRegionDescription,
        topology_region: DeformationTopologyRegion,
    ) -> np.ndarray:
        cached = self._endpoint_displacements.get(description_region.deform_id)
        if cached is not None:
            return cached
        rigid = topology_region.indices_for_role("rigid")
        rigid_ids = topology_region.node_ids[rigid]
        rigid_targets = self._description_targets_for_ids(
            description_region, rigid_ids
        )
        endpoint = self._solve_target_points(
            description_region, topology_region, rigid_targets
        )
        displacement = endpoint - np.asarray(topology_region.mesh.points, dtype=float)
        displacement.setflags(write=False)
        self._endpoint_displacements[description_region.deform_id] = displacement
        return displacement

    def _custom_rigid_targets(
        self,
        topology_region: DeformationTopologyRegion,
        targets_by_id: Mapping[int, Sequence[float]],
    ) -> np.ndarray:
        rigid_indices = topology_region.indices_for_role("rigid")
        rigid_ids = [int(topology_region.node_ids[index]) for index in rigid_indices]
        normalized = {int(node_id): value for node_id, value in targets_by_id.items()}
        if set(normalized) != set(rigid_ids):
            missing = sorted(set(rigid_ids) - set(normalized))
            extra = sorted(set(normalized) - set(rigid_ids))
            raise ValueError(
                f"rigid_targets for DEFORM {topology_region.deform_id} must contain "
                f"every rigid node exactly once (missing={missing[:5]}, extra={extra[:5]})"
            )
        result = np.asarray([normalized[node_id] for node_id in rigid_ids], dtype=float)
        if result.shape != (len(rigid_ids), 3) or not np.all(np.isfinite(result)):
            raise ValueError("rigid_targets must contain finite XYZ coordinates")
        return result

    def _translated_rigid_targets(
        self,
        topology_region: DeformationTopologyRegion,
        translation: Sequence[float],
    ) -> np.ndarray:
        value = np.asarray(translation, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("rigid_translations must contain finite XYZ vectors")
        rigid_indices = topology_region.indices_for_role("rigid")
        return np.asarray(topology_region.mesh.points, dtype=float)[rigid_indices] + value

    def evaluate_result(
        self,
        positions: float | Mapping[int, float] | None = None,
        *,
        rigid_targets: Mapping[int, Mapping[int, Sequence[float]]] | None = None,
        rigid_translations: Mapping[int, Sequence[float]] | None = None,
        mesh_revision: int | str | None = None,
    ) -> PyEMSolMorphingResult:
        custom_targets = dict(rigid_targets or {})
        translations = dict(rigid_translations or {})
        overlap = set(custom_targets).intersection(translations)
        if overlap:
            raise ValueError(
                f"rigid_targets and rigid_translations overlap for DEFORM {sorted(overlap)}"
            )
        if self.method == "linear" and (custom_targets or translations):
            raise ValueError(
                "rigid_targets and rigid_translations require idw, rbf, or "
                "weighted_laplace"
            )
        positions_map = _positions_by_region(
            positions, self.description, set(custom_targets).union(translations)
        )
        payload_regions: list[dict[str, object]] = []
        reports: dict[int, MorphingQualityReport | None] = {}

        for description_region in self.description.regions:
            deform_id = description_region.deform_id
            topology_region = (
                self.topology.region(deform_id) if self.topology is not None else None
            )
            if self.method == "linear":
                coordinates = description_region.coordinates_at(positions_map[deform_id])
                if topology_region is not None:
                    target_points = np.asarray(
                        topology_region.mesh.points, dtype=float
                    ).copy()
                    target_points[topology_region.return_indices] = coordinates
                    reports[deform_id] = enforce_quality(
                        deform_id,
                        topology_region.mesh,
                        target_points,
                        self.quality_config,
                    )
            else:
                assert topology_region is not None
                if deform_id in custom_targets:
                    control_targets = self._custom_rigid_targets(
                        topology_region, custom_targets[deform_id]
                    )
                    target_points = self._solve_target_points(
                        description_region, topology_region, control_targets
                    )
                elif deform_id in translations:
                    control_targets = self._translated_rigid_targets(
                        topology_region, translations[deform_id]
                    )
                    target_points = self._solve_target_points(
                        description_region, topology_region, control_targets
                    )
                else:
                    ratio = description_region.interpolation_ratio(
                        positions_map[deform_id]
                    )
                    target_points = np.asarray(
                        topology_region.mesh.points, dtype=float
                    ) + ratio * self._endpoint_displacement(
                        description_region, topology_region
                    )
                reports[deform_id] = enforce_quality(
                    deform_id,
                    topology_region.mesh,
                    target_points,
                    self.quality_config,
                )
                coordinates = target_points[topology_region.return_indices]

            payload_regions.append(
                {
                    "deform_id": int(deform_id),
                    "node_ids": [int(value) for value in description_region.node_ids],
                    "coordinates": [
                        [float(component) for component in row] for row in coordinates
                    ],
                }
            )

        payload: dict[str, object] = {
            "mode": "absolute",
            "unit": "m",
            "coordinate_frame": "global_after_rigid",
            "regions": payload_regions,
        }
        if mesh_revision is not None:
            payload["mesh_revision"] = _mesh_revision(mesh_revision)
        validate_deformation_payload(payload, self.description)
        return PyEMSolMorphingResult(payload=payload, quality_reports=reports)

    def evaluate(
        self,
        positions: float | Mapping[int, float] | None = None,
        *,
        rigid_targets: Mapping[int, Mapping[int, Sequence[float]]] | None = None,
        rigid_translations: Mapping[int, Sequence[float]] | None = None,
        mesh_revision: int | str | None = None,
    ) -> dict[str, object]:
        return self.evaluate_result(
            positions,
            rigid_targets=rigid_targets,
            rigid_translations=rigid_translations,
            mesh_revision=mesh_revision,
        ).payload


def build_linear_deformation_payload(
    description: Mapping[str, object] | DeformationDescription,
    positions: float | Mapping[int, float],
    *,
    mesh_revision: int | str | None = None,
) -> dict[str, object]:
    return PyEMSolMorphingAdapter.prepare(description).evaluate(
        positions, mesh_revision=mesh_revision
    )
