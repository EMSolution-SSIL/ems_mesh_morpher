from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import meshio
import numpy as np


_NODE_ROLES = frozenset({"fixed", "rigid", "deformable"})
_ELEMENT_ROLES = frozenset({"rigid", "deformable"})
_ROLE_CODES = {"fixed": 0, "rigid": 1, "deformable": 2}

_ELEMENT_TYPE_ALIASES = {
    "TR1": ("triangle", 3),
    "TRI3": ("triangle", 3),
    "TRIA3": ("triangle", 3),
    "TRIANGLE": ("triangle", 3),
    "TRIANGLE_N3E3": ("triangle", 3),
    "Q1": ("quad", 4),
    "QUAD4": ("quad", 4),
    "QUADRILATERAL": ("quad", 4),
    "QUAD_N4E4": ("quad", 4),
    "T1": ("tetra", 4),
    "TET4": ("tetra", 4),
    "TETRA": ("tetra", 4),
    "TETRA_N4E6": ("tetra", 4),
    "PY1": ("pyramid", 5),
    "PYRAMID5": ("pyramid", 5),
    "PYRAMID_N5E8": ("pyramid", 5),
    "PR1": ("wedge", 6),
    "PRISM6": ("wedge", 6),
    "PRISM_N6E9": ("wedge", 6),
    "WEDGE6": ("wedge", 6),
    "H1": ("hexahedron", 8),
    "HEX8": ("hexahedron", 8),
    "HEXA8": ("hexahedron", 8),
    "HEXAHEDRON": ("hexahedron", 8),
    "HEXA_N8E12": ("hexahedron", 8),
}


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _list(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{path} must be an integer")
    return int(value)


def _finite_float(value: object, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _ids(value: object, path: str, *, allow_empty: bool = False) -> np.ndarray:
    values = _list(value, path)
    result = np.asarray(
        [_integer(item, f"{path}[{index}]") for index, item in enumerate(values)],
        dtype=np.int64,
    )
    if not allow_empty and not len(result):
        raise ValueError(f"{path} must not be empty")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{path} must not contain duplicate IDs")
    result.setflags(write=False)
    return result


def _coordinates(value: object, path: str, count: int) -> np.ndarray:
    rows = _list(value, path)
    try:
        result = np.asarray(rows, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must contain numeric XYZ rows") from exc
    if result.shape != (count, 3):
        raise ValueError(f"{path} must have shape ({count}, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{path} must contain only finite coordinates")
    result.setflags(write=False)
    return result


def _required_string(
    mapping: Mapping[str, object], key: str, expected: str, path: str
) -> None:
    if mapping.get(key) != expected:
        raise ValueError(f"{path}.{key} must be {expected!r}")


@dataclass(frozen=True)
class DeformationRegionDescription:
    deform_id: int
    motion_id: int
    position_pre_geom: float
    position_deform_mesh: float
    node_ids: np.ndarray
    reference_coordinates: np.ndarray
    target_coordinates: np.ndarray
    current_coordinates: np.ndarray

    def interpolation_ratio(self, position: float) -> float:
        value = _finite_float(position, "position")
        span = self.position_deform_mesh - self.position_pre_geom
        if abs(span) <= np.finfo(float).eps:
            raise ValueError(
                f"DEFORM {self.deform_id} has identical reference and target positions"
            )
        return (value - self.position_pre_geom) / span

    def coordinates_at(self, position: float) -> np.ndarray:
        ratio = self.interpolation_ratio(position)
        return self.reference_coordinates + ratio * (
            self.target_coordinates - self.reference_coordinates
        )


@dataclass(frozen=True)
class DeformationDescription:
    enabled: bool
    regions: tuple[DeformationRegionDescription, ...]

    def region(self, deform_id: int) -> DeformationRegionDescription:
        for region in self.regions:
            if region.deform_id == deform_id:
                return region
        raise KeyError(f"unknown deform_id {deform_id}")


def parse_deformation_description(value: object) -> DeformationDescription:
    root = _mapping(value, "description")
    if _integer(root.get("schema_version"), "description.schema_version") != 1:
        raise ValueError("description.schema_version must be 1")
    enabled = root.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("description.enabled must be a boolean")
    _required_string(root, "mode", "absolute", "description")
    _required_string(root, "unit", "m", "description")
    _required_string(
        root, "coordinate_frame", "global_after_rigid", "description"
    )

    raw_regions = _list(root.get("regions"), "description.regions")
    regions: list[DeformationRegionDescription] = []
    for index, raw_region in enumerate(raw_regions):
        path = f"description.regions[{index}]"
        region = _mapping(raw_region, path)
        node_ids = _ids(region.get("node_ids"), f"{path}.node_ids")
        count = len(node_ids)
        regions.append(
            DeformationRegionDescription(
                deform_id=_integer(region.get("deform_id"), f"{path}.deform_id"),
                motion_id=_integer(region.get("motion_id"), f"{path}.motion_id"),
                position_pre_geom=_finite_float(
                    region.get("position_pre_geom"), f"{path}.position_pre_geom"
                ),
                position_deform_mesh=_finite_float(
                    region.get("position_deform_mesh"),
                    f"{path}.position_deform_mesh",
                ),
                node_ids=node_ids,
                reference_coordinates=_coordinates(
                    region.get("reference_coordinates"),
                    f"{path}.reference_coordinates",
                    count,
                ),
                target_coordinates=_coordinates(
                    region.get("target_coordinates"),
                    f"{path}.target_coordinates",
                    count,
                ),
                current_coordinates=_coordinates(
                    region.get("current_coordinates"),
                    f"{path}.current_coordinates",
                    count,
                ),
            )
        )

    deform_ids = [region.deform_id for region in regions]
    if len(set(deform_ids)) != len(deform_ids):
        raise ValueError("description.regions must not contain duplicate deform_id values")
    if enabled and not regions:
        raise ValueError("enabled deformation description must contain a region")
    return DeformationDescription(enabled=enabled, regions=tuple(regions))


def element_type_to_meshio(element_type: object, node_count: int) -> str:
    if not isinstance(element_type, str) or not element_type.strip():
        raise ValueError("element_type must be a non-empty string")
    name = element_type.strip().upper().replace("-", "_")
    mapped = _ELEMENT_TYPE_ALIASES.get(name)
    if mapped is None:
        raise ValueError(f"unsupported EMSolution element type {element_type!r}")
    cell_type, expected_count = mapped
    if node_count != expected_count:
        raise ValueError(
            f"element type {element_type!r} requires {expected_count} nodes, "
            f"got {node_count}"
        )
    return cell_type


@dataclass(frozen=True)
class DeformationTopologyRegion:
    deform_id: int
    motion_id: int
    position_pre_geom: float
    position_deform_mesh: float
    return_node_ids: np.ndarray
    return_indices: np.ndarray
    node_ids: np.ndarray
    node_roles: tuple[str, ...]
    node_index_by_id: Mapping[int, int]
    mesh: meshio.Mesh

    def indices_for_role(self, role: str) -> np.ndarray:
        if role not in _NODE_ROLES:
            raise ValueError(f"unknown node role {role!r}")
        return np.asarray(
            [index for index, value in enumerate(self.node_roles) if value == role],
            dtype=int,
        )


@dataclass(frozen=True)
class DeformationTopology:
    mesh_revision: int
    regions: tuple[DeformationTopologyRegion, ...]

    def region(self, deform_id: int) -> DeformationTopologyRegion:
        for region in self.regions:
            if region.deform_id == deform_id:
                return region
        raise KeyError(f"unknown deform_id {deform_id}")


def _parse_boundary_sets(
    value: object,
    node_ids: np.ndarray,
    node_roles: tuple[str, ...],
    path: str,
) -> None:
    boundary_sets = _mapping(value, path)
    if set(boundary_sets) != _NODE_ROLES:
        raise ValueError(f"{path} must contain fixed, rigid, and deformable")
    expected_by_role = {
        role: {int(node_ids[index]) for index, value in enumerate(node_roles) if value == role}
        for role in _NODE_ROLES
    }
    for role in _NODE_ROLES:
        actual = set(_ids(boundary_sets[role], f"{path}.{role}", allow_empty=True).tolist())
        if actual != expected_by_role[role]:
            raise ValueError(f"{path}.{role} does not match nodes.roles")


def _mesh_from_region(
    raw_mesh: Mapping[str, object], path: str
) -> tuple[meshio.Mesh, np.ndarray, tuple[str, ...], Mapping[int, int]]:
    _required_string(
        raw_mesh,
        "scope",
        "deformable_and_rigid_volume_elements_with_fixed_boundary_nodes",
        path,
    )
    nodes = _mapping(raw_mesh.get("nodes"), f"{path}.nodes")
    node_ids = _ids(nodes.get("ids"), f"{path}.nodes.ids")
    coordinates = _coordinates(
        nodes.get("reference_coordinates"),
        f"{path}.nodes.reference_coordinates",
        len(node_ids),
    )
    raw_roles = _list(nodes.get("roles"), f"{path}.nodes.roles")
    if len(raw_roles) != len(node_ids):
        raise ValueError(f"{path}.nodes.roles must have one value per node")
    node_roles = tuple(str(role) for role in raw_roles)
    unknown_roles = set(node_roles) - _NODE_ROLES
    if unknown_roles:
        raise ValueError(f"{path}.nodes.roles contains unknown roles {unknown_roles}")
    node_index = {int(node_id): index for index, node_id in enumerate(node_ids)}
    _parse_boundary_sets(
        raw_mesh.get("boundary_sets"), node_ids, node_roles, f"{path}.boundary_sets"
    )

    raw_elements = _list(raw_mesh.get("volume_elements"), f"{path}.volume_elements")
    grouped_cells: dict[str, list[list[int]]] = {}
    grouped_ids: dict[str, list[int]] = {}
    grouped_materials: dict[str, list[int]] = {}
    grouped_roles: dict[str, list[int]] = {}
    seen_element_ids: set[int] = set()
    for element_index, raw_element in enumerate(raw_elements):
        element_path = f"{path}.volume_elements[{element_index}]"
        element = _mapping(raw_element, element_path)
        element_id = _integer(element.get("element_id"), f"{element_path}.element_id")
        if element_id in seen_element_ids:
            raise ValueError(f"{path}.volume_elements contains duplicate element IDs")
        seen_element_ids.add(element_id)
        role = element.get("role")
        if role not in _ELEMENT_ROLES:
            raise ValueError(f"{element_path}.role must be rigid or deformable")
        element_node_ids = _ids(element.get("node_ids"), f"{element_path}.node_ids")
        try:
            connectivity = [node_index[int(node_id)] for node_id in element_node_ids]
        except KeyError as exc:
            raise ValueError(
                f"{element_path}.node_ids contains an ID absent from mesh.nodes"
            ) from exc
        cell_type = element_type_to_meshio(
            element.get("element_type"), len(element_node_ids)
        )
        _integer(element.get("element_type_id"), f"{element_path}.element_type_id")
        grouped_cells.setdefault(cell_type, []).append(connectivity)
        grouped_ids.setdefault(cell_type, []).append(element_id)
        grouped_materials.setdefault(cell_type, []).append(
            _integer(element.get("material_id"), f"{element_path}.material_id")
        )
        grouped_roles.setdefault(cell_type, []).append(_ROLE_CODES[str(role)])

    if not grouped_cells:
        raise ValueError(f"{path}.volume_elements must not be empty")
    cell_types = list(grouped_cells)
    mesh = meshio.Mesh(
        points=np.asarray(coordinates, dtype=float).copy(),
        cells=[(cell_type, np.asarray(grouped_cells[cell_type], dtype=int)) for cell_type in cell_types],
        point_data={
            "id": np.asarray(node_ids, dtype=np.int64).copy(),
            "role_code": np.asarray([_ROLE_CODES[role] for role in node_roles], dtype=np.int8),
        },
        cell_data={
            "element_id": [np.asarray(grouped_ids[cell_type], dtype=np.int64) for cell_type in cell_types],
            "material_id": [np.asarray(grouped_materials[cell_type], dtype=np.int64) for cell_type in cell_types],
            "role_code": [np.asarray(grouped_roles[cell_type], dtype=np.int8) for cell_type in cell_types],
        },
    )
    return mesh, node_ids, node_roles, MappingProxyType(node_index)


def parse_deformation_topology(
    value: object,
    description: DeformationDescription | Mapping[str, object] | None = None,
    *,
    coordinate_tolerance: float = 1.0e-12,
) -> DeformationTopology:
    if coordinate_tolerance < 0.0 or not np.isfinite(coordinate_tolerance):
        raise ValueError("coordinate_tolerance must be non-negative and finite")
    parsed_description = (
        parse_deformation_description(description)
        if isinstance(description, Mapping)
        else description
    )
    root = _mapping(value, "topology")
    if _integer(root.get("schema_version"), "topology.schema_version") != 2:
        raise ValueError("topology.schema_version must be 2")
    if root.get("enabled") is not True:
        raise ValueError("topology.enabled must be true")
    _required_string(root, "unit", "m", "topology")
    _required_string(root, "coordinate_frame", "global", "topology")
    if root.get("immutable_during_session") is not True:
        raise ValueError("topology.immutable_during_session must be true")
    mesh_revision = _integer(root.get("mesh_revision"), "topology.mesh_revision")

    raw_regions = _list(root.get("regions"), "topology.regions")
    regions: list[DeformationTopologyRegion] = []
    for region_index, raw_region in enumerate(raw_regions):
        path = f"topology.regions[{region_index}]"
        region = _mapping(raw_region, path)
        deform_id = _integer(region.get("deform_id"), f"{path}.deform_id")
        motion_id = _integer(region.get("motion_id"), f"{path}.motion_id")
        position_pre_geom = _finite_float(
            region.get("position_pre_geom"), f"{path}.position_pre_geom"
        )
        position_deform_mesh = _finite_float(
            region.get("position_deform_mesh"), f"{path}.position_deform_mesh"
        )
        return_node_ids = _ids(region.get("return_node_ids"), f"{path}.return_node_ids")
        raw_mesh = _mapping(region.get("mesh"), f"{path}.mesh")
        mesh, node_ids, node_roles, node_index = _mesh_from_region(raw_mesh, f"{path}.mesh")
        try:
            return_indices = np.asarray(
                [node_index[int(node_id)] for node_id in return_node_ids], dtype=int
            )
        except KeyError as exc:
            raise ValueError(f"{path}.return_node_ids contains an unknown node ID") from exc
        fixed_ids = {
            int(node_ids[index])
            for index, role in enumerate(node_roles)
            if role == "fixed"
        }
        if fixed_ids.intersection(return_node_ids.tolist()):
            raise ValueError(f"{path}.return_node_ids must not contain fixed nodes")
        expected_return_ids = {
            int(node_ids[index])
            for index, role in enumerate(node_roles)
            if role != "fixed"
        }
        if set(return_node_ids.tolist()) != expected_return_ids:
            raise ValueError(
                f"{path}.return_node_ids must contain every rigid and deformable node"
            )

        topology_region = DeformationTopologyRegion(
            deform_id=deform_id,
            motion_id=motion_id,
            position_pre_geom=position_pre_geom,
            position_deform_mesh=position_deform_mesh,
            return_node_ids=return_node_ids,
            return_indices=return_indices,
            node_ids=node_ids,
            node_roles=node_roles,
            node_index_by_id=node_index,
            mesh=mesh,
        )
        if parsed_description is not None:
            try:
                description_region = parsed_description.region(deform_id)
            except KeyError as exc:
                raise ValueError(f"{path}.deform_id is absent from description") from exc
            if motion_id != description_region.motion_id:
                raise ValueError(f"{path}.motion_id does not match description")
            if not np.isclose(
                position_pre_geom,
                description_region.position_pre_geom,
                rtol=0.0,
                atol=coordinate_tolerance,
            ) or not np.isclose(
                position_deform_mesh,
                description_region.position_deform_mesh,
                rtol=0.0,
                atol=coordinate_tolerance,
            ):
                raise ValueError(f"{path} motion positions do not match description")
            if not np.array_equal(return_node_ids, description_region.node_ids):
                raise ValueError(f"{path}.return_node_ids does not match description order")
            reference = mesh.points[return_indices]
            if not np.allclose(
                reference,
                description_region.reference_coordinates,
                rtol=0.0,
                atol=coordinate_tolerance,
            ):
                error = float(
                    np.max(np.abs(reference - description_region.reference_coordinates))
                )
                raise ValueError(
                    f"{path} reference coordinates differ from description "
                    f"by up to {error:.6g} m"
                )
        regions.append(topology_region)

    deform_ids = [region.deform_id for region in regions]
    if len(set(deform_ids)) != len(deform_ids):
        raise ValueError("topology.regions must not contain duplicate deform_id values")
    if parsed_description is not None and set(deform_ids) != {
        region.deform_id for region in parsed_description.regions
    }:
        raise ValueError("topology and description must contain the same deform_id values")
    if not regions:
        raise ValueError("topology.regions must not be empty")
    return DeformationTopology(mesh_revision=mesh_revision, regions=tuple(regions))


def validate_deformation_payload(
    value: object,
    description: DeformationDescription | Mapping[str, object],
) -> None:
    parsed_description = (
        parse_deformation_description(description)
        if isinstance(description, Mapping)
        else description
    )
    root = _mapping(value, "payload")
    _required_string(root, "mode", "absolute", "payload")
    _required_string(root, "unit", "m", "payload")
    _required_string(root, "coordinate_frame", "global_after_rigid", "payload")
    if "mesh_revision" in root and (
        isinstance(root["mesh_revision"], bool)
        or not isinstance(root["mesh_revision"], (int, str))
    ):
        raise ValueError("payload.mesh_revision must be an int or str")
    raw_regions = _list(root.get("regions"), "payload.regions")
    if len(raw_regions) != len(parsed_description.regions):
        raise ValueError("payload must contain every DEFORM region exactly once")
    seen: set[int] = set()
    for index, raw_region in enumerate(raw_regions):
        path = f"payload.regions[{index}]"
        region = _mapping(raw_region, path)
        deform_id = _integer(region.get("deform_id"), f"{path}.deform_id")
        if deform_id in seen:
            raise ValueError("payload contains a duplicate deform_id")
        seen.add(deform_id)
        try:
            expected = parsed_description.region(deform_id)
        except KeyError as exc:
            raise ValueError(f"payload contains unknown deform_id {deform_id}") from exc
        node_ids = _ids(region.get("node_ids"), f"{path}.node_ids")
        if set(node_ids.tolist()) != set(expected.node_ids.tolist()):
            raise ValueError(f"{path}.node_ids is not a complete DEFORM region")
        _coordinates(region.get("coordinates"), f"{path}.coordinates", len(node_ids))
