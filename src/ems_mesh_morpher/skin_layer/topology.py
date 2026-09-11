from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import meshio
import numpy as np

from .boundary_roles import BoundaryClassification, BoundaryRole
from .core_morph import CoreMorphingResult
from .offset import OffsetResult
from .quality import evaluate_element_quality
from .surface import SurfaceTopology, _face_geometry, _points3d


@dataclass(frozen=True)
class TopologyBuildResult:
    mesh: meshio.Mesh
    inner_node_map: dict[int, int]
    layer_face_to_element_id: dict[int, int]
    layer_node_maps: tuple[dict[int, int], ...]
    layer_face_to_element_ids: dict[int, tuple[int, ...]]
    layer_block_indices: tuple[int, ...]
    inner_surface_block_indices: tuple[int, ...]
    side_surface_block_indices: tuple[int, ...]


@dataclass(frozen=True)
class _NewBlock:
    cell_type: str
    connectivity: np.ndarray
    property_ids: np.ndarray
    material_ids: np.ndarray
    face_indices: tuple[int, ...] = ()
    category: str = ""
    layer_indices: np.ndarray | None = None


def _existing_cell_values(
    mesh: meshio.Mesh,
    key: str,
    fallback: str | None,
    *,
    default: int,
) -> list[np.ndarray]:
    source_key = key if key in mesh.cell_data else fallback
    if source_key is not None and source_key in mesh.cell_data:
        return [np.asarray(values).copy() for values in mesh.cell_data[source_key]]
    return [np.full(len(block.data), default, dtype=int) for block in mesh.cells]


def _next_positive_id(arrays: list[np.ndarray]) -> int:
    flattened = [np.asarray(array, dtype=int).reshape(-1) for array in arrays]
    maximum = max(
        (int(np.max(values)) for values in flattened if values.size),
        default=0,
    )
    return max(maximum + 1, 1)


def _new_point_data(
    mesh: meshio.Mesh,
    source_nodes: np.ndarray,
    layer_count: int,
    skin_layer_property_id: int,
    inner_surface_property_id: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    result = {key: np.asarray(values).copy() for key, values in mesh.point_data.items()}
    if "id" not in result:
        if "gmsh:node_tags" in result:
            result["id"] = np.asarray(result["gmsh:node_tags"], dtype=int).copy()
        else:
            result["id"] = np.arange(1, len(mesh.points) + 1, dtype=int)
    id_arrays = [np.asarray(result["id"], dtype=int)]
    if "gmsh:node_tags" in result:
        id_arrays.append(np.asarray(result["gmsh:node_tags"], dtype=int))
    next_id = _next_positive_id(id_arrays)
    new_count = len(source_nodes) * layer_count
    new_ids = np.arange(next_id, next_id + new_count, dtype=int)

    for key, values in tuple(result.items()):
        source = np.asarray(values)
        if len(source) != len(mesh.points):
            raise ValueError(f"point_data['{key}'] length does not match mesh points")
        if key in {"id", "gmsh:node_tags"}:
            appended = new_ids.astype(source.dtype, copy=False)
        elif key == "gmsh:dim_tags":
            tags = []
            for layer_index in range(layer_count):
                is_inner_surface = layer_index == layer_count - 1
                tags.append(
                    np.tile(
                        np.array(
                            [
                                2 if is_inner_surface else 3,
                                inner_surface_property_id
                                if is_inner_surface
                                else skin_layer_property_id,
                            ],
                            dtype=source.dtype,
                        ),
                        (len(source_nodes), 1),
                    )
                )
            appended = np.vstack(tags)
        else:
            appended = np.concatenate(
                [source[source_nodes].copy() for _ in range(layer_count)],
                axis=0,
            )
        result[key] = np.concatenate((source, appended), axis=0)
    return result, new_ids


def _oriented_layer_connectivity(
    cell_type: str,
    inner_nodes: tuple[int, ...],
    outer_nodes: tuple[int, ...],
    points: np.ndarray,
) -> tuple[int, ...]:
    candidate = tuple((*inner_nodes, *outer_nodes))
    quality = evaluate_element_quality(cell_type, points[list(candidate)])
    if quality.minimum_scaled_jacobian > 0.0:
        return candidate
    if cell_type == "wedge":
        candidate = (
            inner_nodes[0],
            inner_nodes[2],
            inner_nodes[1],
            outer_nodes[0],
            outer_nodes[2],
            outer_nodes[1],
        )
    else:
        candidate = (
            inner_nodes[0],
            inner_nodes[3],
            inner_nodes[2],
            inner_nodes[1],
            outer_nodes[0],
            outer_nodes[3],
            outer_nodes[2],
            outer_nodes[1],
        )
    quality = evaluate_element_quality(cell_type, points[list(candidate)])
    if quality.minimum_scaled_jacobian <= 0.0:
        raise ValueError(f"could not orient generated {cell_type} layer element")
    return candidate


def _edge_in_face_order(
    face_nodes: tuple[int, ...], edge: tuple[int, int]
) -> tuple[int, int]:
    edge_set = set(edge)
    for index, node in enumerate(face_nodes):
        other = face_nodes[(index + 1) % len(face_nodes)]
        if {node, other} == edge_set:
            return node, other
    raise ValueError(f"edge {edge} is not part of face {face_nodes}")


def _side_connectivity(
    skin_nodes: tuple[int, ...],
    edge: tuple[int, int],
    outer_node_map: dict[int, int] | None,
    inner_node_map: dict[int, int],
    points: np.ndarray,
    desired_normal: np.ndarray,
) -> tuple[int, int, int, int]:
    first, second = _edge_in_face_order(skin_nodes, edge)
    outer_first = first if outer_node_map is None else outer_node_map[first]
    outer_second = second if outer_node_map is None else outer_node_map[second]
    candidate = (
        outer_first,
        outer_second,
        inner_node_map[second],
        inner_node_map[first],
    )
    normal, _ = _face_geometry(points[list(candidate)])
    if float(np.dot(normal, desired_normal)) < 0.0:
        candidate = tuple(reversed(candidate))
    return candidate


def _face_material(mesh: meshio.Mesh, block_index: int, element_index: int) -> int:
    if "material_id" not in mesh.cell_data:
        return 1
    values = np.asarray(mesh.cell_data["material_id"][block_index]).reshape(-1)
    return int(values[element_index])


def _normalised_cell_data(mesh: meshio.Mesh) -> dict[str, list[np.ndarray]]:
    result = {
        key: [np.asarray(values).copy() for values in arrays]
        for key, arrays in mesh.cell_data.items()
    }
    if "property_id" not in result:
        result["property_id"] = _existing_cell_values(
            mesh, "property_id", "gmsh:physical", default=1
        )
    if "element_id" not in result:
        if "gmsh:element_tags" in mesh.cell_data:
            result["element_id"] = _existing_cell_values(
                mesh, "element_id", "gmsh:element_tags", default=0
            )
        else:
            next_id = 1
            arrays = []
            for block in mesh.cells:
                arrays.append(np.arange(next_id, next_id + len(block.data), dtype=int))
                next_id += len(block.data)
            result["element_id"] = arrays
    if "skin_layer_index" not in result:
        result["skin_layer_index"] = [
            np.zeros(len(block.data), dtype=int) for block in mesh.cells
        ]
    return result


def _new_cell_values(
    key: str,
    existing: list[np.ndarray],
    block: _NewBlock,
    element_ids: np.ndarray,
) -> np.ndarray:
    if key in {"element_id", "gmsh:element_tags"}:
        return element_ids.copy()
    if key in {"property_id", "gmsh:physical", "gmsh:geometrical"}:
        return block.property_ids.copy()
    if key == "material_id":
        return block.material_ids.copy()
    if key == "skin_layer_index":
        if block.layer_indices is None:
            return np.zeros(len(block.connectivity), dtype=int)
        return block.layer_indices.copy()
    sample = np.asarray(existing[0])
    shape = (len(block.connectivity), *sample.shape[1:])
    return np.zeros(shape, dtype=sample.dtype)


def _add_field_data(
    field_data: dict[str, np.ndarray],
    name: str,
    property_id: int,
    dimension: int,
) -> None:
    value = np.array([property_id, dimension], dtype=int)
    if name in field_data and not np.array_equal(field_data[name], value):
        raise ValueError(f"field_data name '{name}' already has a different definition")
    field_data[name] = value


def build_layered_mesh(
    mesh: meshio.Mesh,
    surface: SurfaceTopology,
    classification: BoundaryClassification,
    offset: OffsetResult,
    core_morph: CoreMorphingResult,
    selected_volume_elements: Mapping[int, np.ndarray],
    *,
    skin_layer_property_id: int,
    inner_surface_property_id: int,
    side_surface_property_id: int | None = None,
    skin_layer_name: str = "skin_layer",
    inner_surface_name: str = "skin_inner_surface",
    layer_fractions: tuple[float, ...] = (1.0,),
) -> TopologyBuildResult:
    if not np.any(offset.active_nodes):
        raise ValueError("no skin boundary faces were selected")
    fractions = np.asarray(layer_fractions, dtype=float)
    if (
        fractions.ndim != 1
        or len(fractions) == 0
        or not np.all(np.isfinite(fractions))
        or np.any(fractions <= 0.0)
        or np.any(np.diff(fractions) <= 0.0)
        or not np.isclose(fractions[-1], 1.0)
    ):
        raise ValueError(
            "layer_fractions must be finite, strictly increasing, and end at 1"
        )
    source_nodes = np.flatnonzero(offset.active_nodes)
    layer_node_maps = tuple(
        {
            int(source): len(mesh.points) + layer_index * len(source_nodes) + local
            for local, source in enumerate(source_nodes)
        }
        for layer_index in range(len(fractions))
    )
    inner_node_map = layer_node_maps[-1]

    original_points = _points3d(mesh.points).copy()
    original_points[core_morph.movable_nodes] += core_morph.displacements[
        core_morph.movable_nodes
    ]
    boundary_points = _points3d(mesh.points)[source_nodes]
    layer_points = [
        boundary_points + fraction * offset.displacements[source_nodes]
        for fraction in fractions
    ]
    points = np.vstack((original_points, *layer_points))
    point_data, _ = _new_point_data(
        mesh,
        source_nodes,
        len(fractions),
        skin_layer_property_id,
        inner_surface_property_id,
    )

    cells = [np.asarray(block.data, dtype=int).copy() for block in mesh.cells]
    for block_index, selection in selected_volume_elements.items():
        for element_index in np.flatnonzero(selection):
            cells[block_index][element_index] = np.array(
                [
                    inner_node_map.get(int(node), int(node))
                    for node in cells[block_index][element_index]
                ],
                dtype=int,
            )

    for face_index, face in enumerate(surface.faces):
        if classification.roles[face_index] == BoundaryRole.SKIN:
            continue
        if face.surface_block_index is None or face.surface_element_index is None:
            continue
        existing_nodes = cells[face.surface_block_index][face.surface_element_index]
        cells[face.surface_block_index][face.surface_element_index] = np.array(
            [inner_node_map.get(int(node), int(node)) for node in existing_nodes],
            dtype=int,
        )

    layer_wedges: list[tuple[int, ...]] = []
    wedge_faces: list[int] = []
    wedge_materials: list[int] = []
    wedge_layer_indices: list[int] = []
    layer_hexes: list[tuple[int, ...]] = []
    hex_faces: list[int] = []
    hex_materials: list[int] = []
    hex_layer_indices: list[int] = []
    inner_triangles: list[tuple[int, ...]] = []
    inner_triangle_faces: list[int] = []
    inner_quads: list[tuple[int, ...]] = []
    inner_quad_faces: list[int] = []

    for layer_offset, current_node_map in enumerate(layer_node_maps):
        previous_node_map = (
            None if layer_offset == 0 else layer_node_maps[layer_offset - 1]
        )
        layer_index = layer_offset + 1
        for face_index in classification.skin_face_indices:
            face = surface.faces[face_index]
            outer = tuple(
                node
                if previous_node_map is None
                else previous_node_map[node]
                for node in face.node_indices
            )
            inner = tuple(current_node_map[node] for node in face.node_indices)
            material = _face_material(
                mesh, face.owner_block_index, face.owner_element_index
            )
            if face.face_type == "triangle":
                layer_wedges.append(
                    _oriented_layer_connectivity("wedge", inner, outer, points)
                )
                wedge_faces.append(face_index)
                wedge_materials.append(material)
                wedge_layer_indices.append(layer_index)
            else:
                layer_hexes.append(
                    _oriented_layer_connectivity("hexahedron", inner, outer, points)
                )
                hex_faces.append(face_index)
                hex_materials.append(material)
                hex_layer_indices.append(layer_index)

    for face_index in classification.skin_face_indices:
        face = surface.faces[face_index]
        inner = tuple(inner_node_map[node] for node in face.node_indices)
        if face.face_type == "triangle":
            inner_triangles.append(inner)
            inner_triangle_faces.append(face_index)
        else:
            inner_quads.append(inner)
            inner_quad_faces.append(face_index)

    side_quads: list[tuple[int, ...]] = []
    side_properties: list[int] = []
    side_layer_indices: list[int] = []
    for edge, face_indices in surface.edge_to_faces.items():
        if len(face_indices) != 2:
            continue
        skin_indices = [
            index
            for index in face_indices
            if classification.roles[index] == BoundaryRole.SKIN
        ]
        if len(skin_indices) != 1:
            continue
        other_index = (
            face_indices[0]
            if face_indices[1] == skin_indices[0]
            else face_indices[1]
        )
        other = surface.faces[other_index]
        side_property = (
            side_surface_property_id
            if side_surface_property_id is not None
            else other.surface_property_id
            if other.surface_property_id is not None
            else inner_surface_property_id
        )
        for layer_offset, current_node_map in enumerate(layer_node_maps):
            previous_node_map = (
                None if layer_offset == 0 else layer_node_maps[layer_offset - 1]
            )
            side_quads.append(
                _side_connectivity(
                    surface.faces[skin_indices[0]].node_indices,
                    edge,
                    previous_node_map,
                    current_node_map,
                    points,
                    other.normal,
                )
            )
            side_properties.append(side_property)
            side_layer_indices.append(layer_offset + 1)

    new_blocks: list[_NewBlock] = []
    if layer_wedges:
        new_blocks.append(
            _NewBlock(
                "wedge",
                np.asarray(layer_wedges, dtype=int),
                np.full(len(layer_wedges), skin_layer_property_id, dtype=int),
                np.asarray(wedge_materials, dtype=int),
                tuple(wedge_faces),
                "layer",
                np.asarray(wedge_layer_indices, dtype=int),
            )
        )
    if layer_hexes:
        new_blocks.append(
            _NewBlock(
                "hexahedron",
                np.asarray(layer_hexes, dtype=int),
                np.full(len(layer_hexes), skin_layer_property_id, dtype=int),
                np.asarray(hex_materials, dtype=int),
                tuple(hex_faces),
                "layer",
                np.asarray(hex_layer_indices, dtype=int),
            )
        )
    if inner_triangles:
        new_blocks.append(
            _NewBlock(
                "triangle",
                np.asarray(inner_triangles, dtype=int),
                np.full(len(inner_triangles), inner_surface_property_id, dtype=int),
                np.ones(len(inner_triangles), dtype=int),
                tuple(inner_triangle_faces),
                "inner",
            )
        )
    if inner_quads:
        new_blocks.append(
            _NewBlock(
                "quad",
                np.asarray(inner_quads, dtype=int),
                np.full(len(inner_quads), inner_surface_property_id, dtype=int),
                np.ones(len(inner_quads), dtype=int),
                tuple(inner_quad_faces),
                "inner",
            )
        )
    if side_quads:
        new_blocks.append(
            _NewBlock(
                "quad",
                np.asarray(side_quads, dtype=int),
                np.asarray(side_properties, dtype=int),
                np.ones(len(side_quads), dtype=int),
                (),
                "side",
                np.asarray(side_layer_indices, dtype=int),
            )
        )

    cell_data = _normalised_cell_data(mesh)
    for key, arrays in cell_data.items():
        if len(arrays) != len(mesh.cells):
            raise ValueError(f"cell_data['{key}'] block count does not match mesh cells")
    id_sources = [np.asarray(values) for values in cell_data["element_id"]]
    if "gmsh:element_tags" in cell_data:
        id_sources.extend(np.asarray(values) for values in cell_data["gmsh:element_tags"])
    next_element_id = _next_positive_id(id_sources)

    layer_face_to_element_id: dict[int, int] = {}
    layer_face_to_element_id_lists: dict[int, list[int]] = {}
    layer_block_indices: list[int] = []
    inner_block_indices: list[int] = []
    side_block_indices: list[int] = []
    new_element_ids: list[np.ndarray] = []
    for block in new_blocks:
        element_ids = np.arange(
            next_element_id,
            next_element_id + len(block.connectivity),
            dtype=int,
        )
        next_element_id += len(block.connectivity)
        new_element_ids.append(element_ids)
        block_index = len(cells)
        cells.append(block.connectivity)
        if block.category == "layer":
            layer_block_indices.append(block_index)
            for face_index, element_id in zip(
                block.face_indices, element_ids, strict=True
            ):
                layer_face_to_element_id_lists.setdefault(face_index, []).append(
                    int(element_id)
                )
                layer_face_to_element_id[face_index] = int(element_id)
        elif block.category == "inner":
            inner_block_indices.append(block_index)
        elif block.category == "side":
            side_block_indices.append(block_index)

    for key, arrays in cell_data.items():
        for block, element_ids in zip(new_blocks, new_element_ids, strict=True):
            arrays.append(_new_cell_values(key, arrays, block, element_ids))

    field_data = {
        key: np.asarray(value).copy()
        for key, value in (mesh.field_data or {}).items()
    }
    _add_field_data(field_data, skin_layer_name, skin_layer_property_id, 3)
    _add_field_data(field_data, inner_surface_name, inner_surface_property_id, 2)

    point_sets = {
        name: np.asarray(indices, dtype=int).copy()
        for name, indices in mesh.point_sets.items()
    }
    point_sets[inner_surface_name] = np.fromiter(
        inner_node_map.values(), dtype=int
    )
    for layer_index, node_map in enumerate(layer_node_maps, start=1):
        point_sets[f"{skin_layer_name}_{layer_index}_nodes"] = np.fromiter(
            node_map.values(), dtype=int
        )

    cell_sets = {
        name: [np.asarray(indices, dtype=int).copy() for indices in arrays]
        for name, arrays in mesh.cell_sets.items()
    }
    for arrays in cell_sets.values():
        arrays.extend(
            np.empty(0, dtype=int) for _ in range(len(cells) - len(arrays))
        )

    def category_set(block_indices: tuple[int, ...]) -> list[np.ndarray]:
        selected = set(block_indices)
        return [
            np.arange(len(cells[index]), dtype=int)
            if index in selected
            else np.empty(0, dtype=int)
            for index in range(len(cells))
        ]

    cell_sets[skin_layer_name] = category_set(tuple(layer_block_indices))
    cell_sets[inner_surface_name] = category_set(tuple(inner_block_indices))
    for layer_index in range(1, len(layer_node_maps) + 1):
        arrays = [np.empty(0, dtype=int) for _ in cells]
        for block_offset, block in enumerate(new_blocks):
            if block.category != "layer" or block.layer_indices is None:
                continue
            arrays[len(mesh.cells) + block_offset] = np.flatnonzero(
                block.layer_indices == layer_index
            )
        cell_sets[f"{skin_layer_name}_{layer_index}"] = arrays

    output = meshio.Mesh(
        points=points,
        cells=[
            (mesh.cells[index].type, data)
            if index < len(mesh.cells)
            else (new_blocks[index - len(mesh.cells)].cell_type, data)
            for index, data in enumerate(cells)
        ],
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        point_sets=point_sets,
        cell_sets=cell_sets,
        gmsh_periodic=mesh.gmsh_periodic,
        info=dict(mesh.info or {}),
    )
    return TopologyBuildResult(
        mesh=output,
        inner_node_map=inner_node_map,
        layer_face_to_element_id=layer_face_to_element_id,
        layer_node_maps=layer_node_maps,
        layer_face_to_element_ids={
            face_index: tuple(element_ids)
            for face_index, element_ids in layer_face_to_element_id_lists.items()
        },
        layer_block_indices=tuple(layer_block_indices),
        inner_surface_block_indices=tuple(inner_block_indices),
        side_surface_block_indices=tuple(side_block_indices),
    )


def build_single_layer_mesh(*args, **kwargs) -> TopologyBuildResult:
    """Backward-compatible wrapper for the original Phase 10C builder."""
    return build_layered_mesh(*args, layer_fractions=(1.0,), **kwargs)
