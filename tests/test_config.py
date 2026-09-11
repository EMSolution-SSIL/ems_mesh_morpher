import pytest

from ems_mesh_morpher.config import config_from_dict


def test_property_id_ranges_are_expanded_inclusively():
    config = config_from_dict(
        {
            "geometry": {"moving_radius": 1.0, "fixed_radius": 2.0},
            "regions": {
                "moving_property_ids": [10, {"start": 20, "stop": 24, "step": 2}],
                "deformable_property_ids": [30],
            },
        }
    )

    assert config.regions.moving_property_ids == (10, 20, 22, 24)
    assert config.regions.deformable_property_ids == (30,)


def test_legacy_airgap_radii_can_select_outer_moving_boundary():
    config = config_from_dict(
        {
            "geometry": {
                "airgap_inner_radius": 1.0,
                "airgap_outer_radius": 2.0,
                "moving_boundary": "outer",
            }
        }
    )

    assert config.geometry.moving_radius == 2.0
    assert config.geometry.fixed_radius == 1.0


def test_general_motion_config_parses_phase_2_and_3_fields():
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {
                "moving": {
                    "property_ids": [10],
                    "geometry": {
                        "type": "box",
                        "minimum": [0.0, 0.0],
                        "maximum": [1.0, 1.0],
                    },
                    "combine": "intersection",
                },
                "morphing_zone": {
                    "geometry": {
                        "type": "annulus",
                        "center": [0.0, 0.0],
                        "inner_radius": 1.0,
                        "outer_radius": 2.0,
                    }
                },
            },
            "motion": {
                "type": "rotation",
                "center": [0.0, 0.0],
                "angle_degrees": 2.5,
            },
            "morphing": {"type": "distance_blend"},
        }
    )

    assert config.mode == "mesh_motion"
    assert config.regions.moving.property_ids == (10,)
    assert config.regions.moving.combine == "intersection"
    assert config.regions.morphing_zone is not None
    assert config.motion.type == "rotation"
    assert config.motion.angle_degrees == 2.5
    assert config.morphing.type == "distance_blend"


def test_general_motion_defaults_to_rigid_motion_without_propagation():
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {"moving": {"point_indices": [0]}},
            "motion": {"type": "translation", "displacement": [1.0, 0.0]},
        }
    )

    assert config.morphing.type == "none"


def test_idw_morphing_options_are_parsed() -> None:
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {"moving": {"point_indices": [0]}},
            "motion": {"type": "translation", "displacement": [1.0, 0.0]},
            "morphing": {
                "type": "idw",
                "power": 3.0,
                "neighbors": 12,
                "radius": 2.5,
                "smoothing": 0.01,
                "outside_policy": "zero",
                "query_chunk_size": 256,
            },
        }
    )

    assert config.morphing.type == "idw"
    assert config.morphing.power == 3.0
    assert config.morphing.neighbors == 12
    assert config.morphing.radius == 2.5
    assert config.morphing.smoothing == 0.01
    assert config.morphing.outside_policy == "zero"
    assert config.morphing.query_chunk_size == 256


def test_rbf_morphing_options_are_parsed() -> None:
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {"moving": {"point_indices": [0]}},
            "motion": {"type": "translation", "displacement": [0.1, 0.0]},
            "morphing": {
                "type": "rbf",
                "kernel": "wendland_c2",
                "neighbors": 24,
                "radius": 0.5,
                "radius_scale": 1.4,
                "regularization": 1.0e-10,
                "polynomial_degree": 0,
                "outside_policy": "zero",
            },
            "quality": {"enabled": False},
        }
    )

    assert config.morphing.type == "rbf"
    assert config.morphing.kernel == "wendland_c2"
    assert config.morphing.neighbors == 24
    assert config.morphing.radius == pytest.approx(0.5)
    assert config.morphing.radius_scale == pytest.approx(1.4)
    assert config.morphing.regularization == pytest.approx(1.0e-10)
    assert config.morphing.polynomial_degree == 0
    assert config.morphing.outside_policy == "zero"


@pytest.mark.parametrize("morphing_type", ["laplace", "weighted_laplace"])
def test_weighted_laplace_options_are_parsed(morphing_type: str) -> None:
    config = config_from_dict(
        {
            "mode": "mesh_motion",
            "regions": {"moving": {"point_indices": [0]}},
            "motion": {"type": "translation", "displacement": [0.1, 0.0]},
            "morphing": {
                "type": morphing_type,
                "weighting": "inverse_distance",
                "distance_power": 2.0,
                "minimum_distance": 1.0e-10,
            },
            "quality": {"enabled": False},
        }
    )

    assert config.morphing.type == morphing_type
    assert config.morphing.weighting == "inverse_distance"
    assert config.morphing.distance_power == pytest.approx(2.0)
    assert config.morphing.minimum_distance == pytest.approx(1.0e-10)
