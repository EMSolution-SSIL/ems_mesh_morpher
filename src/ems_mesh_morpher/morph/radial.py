import numpy as np

from .blend import shape_blend_weights


def radial_blend_weights(
    points: np.ndarray,
    center: tuple[float, float],
    moving_radius: float,
    fixed_radius: float,
    blend: str = "smoothstep",
) -> np.ndarray:
    xy = points[:, :2] - np.array(center, dtype=float)
    radius = np.linalg.norm(xy, axis=1)
    denom = abs(float(moving_radius) - float(fixed_radius))
    if denom == 0.0:
        raise ValueError("moving_radius and fixed_radius must differ")

    t = np.abs(radius - float(fixed_radius)) / denom
    return shape_blend_weights(t, blend)


def apply_radial_blend(
    points: np.ndarray,
    node_mask: np.ndarray,
    displacement: tuple[float, float],
    center: tuple[float, float],
    moving_radius: float,
    fixed_radius: float,
    blend: str = "smoothstep",
) -> None:
    weights = radial_blend_weights(points, center, moving_radius, fixed_radius, blend=blend)
    delta = np.zeros(points.shape[1], dtype=float)
    if len(displacement) > points.shape[1]:
        raise ValueError("displacement has more dimensions than mesh points")
    delta[: len(displacement)] = displacement
    points[node_mask] += weights[node_mask, None] * delta
