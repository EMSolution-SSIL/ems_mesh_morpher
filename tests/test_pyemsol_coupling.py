from __future__ import annotations

import copy

import meshio
import numpy as np
import pytest

from ems_mesh_morpher import (
    MorphingQualityError,
    PyEMSolConstraintConfig,
    PyEMSolMorphingAdapter,
    build_linear_deformation_payload,
    element_type_to_meshio,
    evaluate_relative_quality,
    parse_deformation_description,
    parse_deformation_topology,
    validate_deformation_payload,
)


def _description() -> dict[str, object]:
    reference = [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ]
    target = [
        [1.5, 0.0, 0.0],
        [1.5, 1.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
    ]
    return {
        "schema_version": 1,
        "enabled": True,
        "mode": "absolute",
        "unit": "m",
        "coordinate_frame": "global_after_rigid",
        "regions": [
            {
                "deform_id": 1,
                "motion_id": 7,
                "position_pre_geom": 0.0,
                "position_deform_mesh": 1.0,
                "node_ids": [20, 21, 30, 31],
                "reference_coordinates": reference,
                "target_coordinates": target,
                "current_coordinates": reference,
            }
        ],
    }


def _topology() -> dict[str, object]:
    return {
        "schema_version": 2,
        "enabled": True,
        "mesh_revision": 0,
        "unit": "m",
        "coordinate_frame": "global",
        "immutable_during_session": True,
        "regions": [
            {
                "deform_id": 1,
                "motion_id": 7,
                "position_pre_geom": 0.0,
                "position_deform_mesh": 1.0,
                "return_node_ids": [20, 21, 30, 31],
                "mesh": {
                    "scope": "deformable_and_rigid_volume_elements_with_fixed_boundary_nodes",
                    "nodes": {
                        "ids": [10, 11, 20, 21, 30, 31],
                        "reference_coordinates": [
                            [0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [2.0, 1.0, 0.0],
                        ],
                        "roles": [
                            "fixed",
                            "fixed",
                            "deformable",
                            "deformable",
                            "rigid",
                            "rigid",
                        ],
                    },
                    "volume_elements": [
                        {
                            "element_id": 100,
                            "element_type_id": 10,
                            "element_type": "Q1",
                            "material_id": 2,
                            "role": "deformable",
                            "node_ids": [10, 20, 21, 11],
                        },
                        {
                            "element_id": 101,
                            "element_type_id": 10,
                            "element_type": "Q1",
                            "material_id": 1,
                            "role": "rigid",
                            "node_ids": [20, 30, 31, 21],
                        },
                    ],
                    "boundary_sets": {
                        "fixed": [10, 11],
                        "rigid": [30, 31],
                        "deformable": [20, 21],
                    },
                },
            }
        ],
    }


@pytest.mark.parametrize(
    ("name", "node_count", "expected"),
    [
        ("TR1", 3, "triangle"),
        ("Q1", 4, "quad"),
        ("T1", 4, "tetra"),
        ("PY1", 5, "pyramid"),
        ("PR1", 6, "wedge"),
        ("H1", 8, "hexahedron"),
        ("HEXA_N8E12", 8, "hexahedron"),
    ],
)
def test_element_type_mapping(name: str, node_count: int, expected: str) -> None:
    assert element_type_to_meshio(name, node_count) == expected


def test_schema_parsing_builds_mesh_and_external_id_map() -> None:
    description = parse_deformation_description(_description())
    topology = parse_deformation_topology(_topology(), description)
    region = topology.region(1)

    assert region.mesh.cells[0].type == "quad"
    np.testing.assert_array_equal(region.mesh.point_data["id"], [10, 11, 20, 21, 30, 31])
    np.testing.assert_array_equal(region.return_indices, [2, 3, 4, 5])
    np.testing.assert_array_equal(region.indices_for_role("fixed"), [0, 1])
    np.testing.assert_array_equal(region.indices_for_role("rigid"), [4, 5])


def test_schema_parser_does_not_mutate_inputs() -> None:
    description = _description()
    topology = _topology()
    description_before = copy.deepcopy(description)
    topology_before = copy.deepcopy(topology)

    parse_deformation_topology(topology, description)

    assert description == description_before
    assert topology == topology_before


def test_linear_payload_matches_reference_interpolation() -> None:
    payload = build_linear_deformation_payload(
        _description(), 0.5, mesh_revision=np.int64(4)
    )

    assert payload["mesh_revision"] == 4
    assert payload["regions"][0]["node_ids"] == [20, 21, 30, 31]
    np.testing.assert_allclose(
        payload["regions"][0]["coordinates"],
        [[1.25, 0.0, 0.0], [1.25, 1.0, 0.0], [2.5, 0.0, 0.0], [2.5, 1.0, 0.0]],
    )
    validate_deformation_payload(payload, _description())


def test_linear_payload_supports_multiple_deform_regions() -> None:
    description = _description()
    second = copy.deepcopy(description["regions"][0])
    second["deform_id"] = 2
    second["motion_id"] = 8
    second["node_ids"] = [120, 121, 130, 131]
    second["reference_coordinates"] = (
        np.asarray(second["reference_coordinates"]) + [0.0, 10.0, 0.0]
    ).tolist()
    second["target_coordinates"] = (
        np.asarray(second["target_coordinates"]) + [0.0, 10.0, 0.0]
    ).tolist()
    second["current_coordinates"] = copy.deepcopy(
        second["reference_coordinates"]
    )
    description["regions"].append(second)

    payload = build_linear_deformation_payload(description, {1: 0.5, 2: 0.25})

    assert [region["deform_id"] for region in payload["regions"]] == [1, 2]
    np.testing.assert_allclose(
        payload["regions"][1]["coordinates"][0], [1.125, 10.0, 0.0]
    )


def test_multiple_regions_reject_scalar_position() -> None:
    description = _description()
    second = copy.deepcopy(description["regions"][0])
    second["deform_id"] = 2
    second["motion_id"] = 8
    second["node_ids"] = [120, 121, 130, 131]
    description["regions"].append(second)

    with pytest.raises(ValueError, match="position mapping"):
        build_linear_deformation_payload(description, 0.5)


def test_position_rejects_boolean() -> None:
    with pytest.raises(ValueError, match="not booleans"):
        build_linear_deformation_payload(_description(), True)


@pytest.mark.parametrize("method", ["idw", "rbf", "weighted_laplace"])
def test_topology_morphing_preserves_rigid_targets(method: str) -> None:
    adapter = PyEMSolMorphingAdapter.prepare(
        _description(), topology=_topology(), method=method
    )

    result = adapter.evaluate_result(1.0)
    coordinates = np.asarray(result.payload["regions"][0]["coordinates"])

    np.testing.assert_allclose(coordinates[2:], [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0]])
    assert np.all(np.isfinite(coordinates))
    assert result.quality_reports[1].orientation_flip_count == 0
    assert result.quality_reports[1].element_count == 2


def test_weighted_laplace_reuses_reference_endpoint_field() -> None:
    adapter = PyEMSolMorphingAdapter.prepare(
        _description(), topology=_topology(), method="weighted_laplace"
    )

    full = np.asarray(adapter.evaluate(1.0)["regions"][0]["coordinates"])
    half = np.asarray(adapter.evaluate(0.5)["regions"][0]["coordinates"])

    np.testing.assert_allclose(full, [[1.5, 0.0, 0.0], [1.5, 1.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0]])
    np.testing.assert_allclose(half, [[1.25, 0.0, 0.0], [1.25, 1.0, 0.0], [2.5, 0.0, 0.0], [2.5, 1.0, 0.0]])
    assert len(adapter._endpoint_displacements) == 1


def test_repeated_motion_is_evaluated_from_the_reference_mesh() -> None:
    adapter = PyEMSolMorphingAdapter.prepare(
        _description(), topology=_topology(), method="weighted_laplace"
    )

    first = adapter.evaluate(0.75)
    reference = adapter.evaluate(0.0)
    repeated = adapter.evaluate(0.75)

    np.testing.assert_allclose(
        reference["regions"][0]["coordinates"],
        _description()["regions"][0]["reference_coordinates"],
    )
    np.testing.assert_allclose(
        repeated["regions"][0]["coordinates"],
        first["regions"][0]["coordinates"],
    )


def test_custom_rigid_targets_support_arbitrary_motion() -> None:
    adapter = PyEMSolMorphingAdapter.prepare(
        _description(), topology=_topology(), method="weighted_laplace"
    )
    payload = adapter.evaluate(
        rigid_targets={1: {30: [2.0, 0.2, 0.0], 31: [2.0, 1.2, 0.0]}}
    )
    coordinates = np.asarray(payload["regions"][0]["coordinates"])

    np.testing.assert_allclose(coordinates[2:], [[2.0, 0.2, 0.0], [2.0, 1.2, 0.0]])


def test_rigid_translation_supports_independent_xyz_motion() -> None:
    adapter = PyEMSolMorphingAdapter.prepare(
        _description(), topology=_topology(), method="weighted_laplace"
    )
    payload = adapter.evaluate(rigid_translations={1: [0.2, -0.3, 0.0]})
    coordinates = np.asarray(payload["regions"][0]["coordinates"])

    np.testing.assert_allclose(coordinates[2:], [[2.2, -0.3, 0.0], [2.2, 0.7, 0.0]])


def test_rigid_translation_rejects_conflicting_targets() -> None:
    adapter = PyEMSolMorphingAdapter.prepare(
        _description(), topology=_topology(), method="idw"
    )
    with pytest.raises(ValueError, match="overlap"):
        adapter.evaluate(
            rigid_targets={1: {30: [2.0, 0.0, 0.0], 31: [2.0, 1.0, 0.0]}},
            rigid_translations={1: [0.1, 0.0, 0.0]},
        )


def test_custom_rigid_targets_require_every_rigid_node() -> None:
    adapter = PyEMSolMorphingAdapter.prepare(
        _description(), topology=_topology(), method="idw"
    )
    with pytest.raises(ValueError, match="every rigid node"):
        adapter.evaluate(rigid_targets={1: {30: [2.0, 0.2, 0.0]}})


def test_quality_gate_rejects_orientation_flip() -> None:
    adapter = PyEMSolMorphingAdapter.prepare(
        _description(), topology=_topology(), method="weighted_laplace"
    )
    with pytest.raises(MorphingQualityError, match="orientation flips"):
        adapter.evaluate(
            rigid_targets={1: {30: [-1.0, 0.0, 0.0], 31: [-1.0, 1.0, 0.0]}}
        )


def test_topology_rejects_description_order_mismatch() -> None:
    topology = _topology()
    topology["regions"][0]["return_node_ids"] = [21, 20, 30, 31]

    with pytest.raises(ValueError, match="does not match description order"):
        parse_deformation_topology(topology, _description())


def test_zero_displacement_deformable_nodes_can_be_inferred_as_anchors() -> None:
    description = _description()
    topology = _topology()
    topology_nodes = topology["regions"][0]["mesh"]["nodes"]
    topology_nodes["roles"] = [
        "deformable",
        "deformable",
        "deformable",
        "deformable",
        "rigid",
        "rigid",
    ]
    topology["regions"][0]["mesh"]["boundary_sets"] = {
        "fixed": [],
        "rigid": [30, 31],
        "deformable": [10, 11, 20, 21],
    }
    description_region = description["regions"][0]
    description_region["node_ids"] = [10, 11, 20, 21, 30, 31]
    description_region["reference_coordinates"] = topology_nodes[
        "reference_coordinates"
    ]
    description_region["target_coordinates"] = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 1.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
    ]
    description_region["current_coordinates"] = topology_nodes[
        "reference_coordinates"
    ]
    topology["regions"][0]["return_node_ids"] = [10, 11, 20, 21, 30, 31]

    adapter = PyEMSolMorphingAdapter.prepare(
        description, topology=topology, method="weighted_laplace"
    )
    coordinates = np.asarray(adapter.evaluate(1.0)["regions"][0]["coordinates"])

    np.testing.assert_allclose(coordinates[:2], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert adapter.constraint_summaries[1].explicit_fixed_count == 0
    assert adapter.constraint_summaries[1].inferred_fixed_count == 2


def test_zero_displacement_anchor_inference_can_be_disabled() -> None:
    adapter = PyEMSolMorphingAdapter.prepare(
        _description(),
        topology=_topology(),
        method="idw",
        constraint_config=PyEMSolConstraintConfig(
            infer_zero_displacement_anchors=False
        ),
    )

    adapter.evaluate(1.0)

    assert adapter.constraint_summaries[1].inferred_fixed_count == 0


def test_payload_rejects_missing_node() -> None:
    payload = build_linear_deformation_payload(_description(), 0.0)
    payload["regions"][0]["node_ids"].pop()
    payload["regions"][0]["coordinates"].pop()

    with pytest.raises(ValueError, match="complete DEFORM region"):
        validate_deformation_payload(payload, _description())


@pytest.mark.parametrize(
    ("cell_type", "points"),
    [
        ("triangle", [[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        ("quad", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
        ("tetra", [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        (
            "pyramid",
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]],
        ),
        (
            "wedge",
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
        ),
        (
            "hexahedron",
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
        ),
    ],
)
def test_relative_quality_supports_target_linear_element_types(
    cell_type: str, points: list[list[float]]
) -> None:
    reference = np.asarray(points, dtype=float)
    mesh = meshio.Mesh(reference, [(cell_type, [list(range(len(reference)))])])
    target = reference + np.array([0.2, -0.1, 0.3])

    report = evaluate_relative_quality(mesh, target)

    assert report.element_count == 1
    assert report.orientation_flip_count == 0
    assert report.degenerate_count == 0
    assert report.minimum_measure_ratio == pytest.approx(1.0)
