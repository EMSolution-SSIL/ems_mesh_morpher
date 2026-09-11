from __future__ import annotations

import numpy as np

from .blend import shape_blend_weights


def nearest_reference(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query_points, dtype=float)
    reference = np.asarray(reference_points, dtype=float)
    if len(reference) == 0:
        raise ValueError("nearest-reference calculation requires at least one reference point")
    if chunk_size <= 0:
        raise ValueError("distance chunk size must be positive")

    distances = np.empty(len(query), dtype=float)
    indices = np.empty(len(query), dtype=int)
    for query_start in range(0, len(query), chunk_size):
        query_chunk = query[query_start : query_start + chunk_size]
        best_squared = np.full(len(query_chunk), np.inf, dtype=float)
        best_indices = np.zeros(len(query_chunk), dtype=int)
        for reference_start in range(0, len(reference), chunk_size):
            reference_chunk = reference[reference_start : reference_start + chunk_size]
            difference = query_chunk[:, None, :] - reference_chunk[None, :, :]
            squared = np.einsum("qrd,qrd->qr", difference, difference)
            local_indices = np.argmin(squared, axis=1)
            local_squared = squared[np.arange(len(query_chunk)), local_indices]
            improved = local_squared < best_squared
            best_squared[improved] = local_squared[improved]
            best_indices[improved] = reference_start + local_indices[improved]
        stop = query_start + len(query_chunk)
        distances[query_start:stop] = np.sqrt(best_squared)
        indices[query_start:stop] = best_indices
    return distances, indices


def distance_blend_weights(
    points: np.ndarray,
    node_mask: np.ndarray,
    moving_mask: np.ndarray,
    fixed_mask: np.ndarray,
    blend: str = "smoothstep",
    chunk_size: int = 2048,
) -> np.ndarray:
    if not np.any(moving_mask):
        raise ValueError("distance_blend requires at least one moving node")
    if not np.any(fixed_mask):
        raise ValueError("distance_blend requires at least one fixed or protected node")

    weights = np.zeros(len(points), dtype=float)
    query = np.asarray(points)[node_mask]
    distance_to_moving, _ = nearest_reference(query, np.asarray(points)[moving_mask], chunk_size)
    distance_to_fixed, _ = nearest_reference(query, np.asarray(points)[fixed_mask], chunk_size)
    denominator = distance_to_moving + distance_to_fixed
    raw = np.divide(
        distance_to_fixed,
        denominator,
        out=np.full_like(denominator, 0.5),
        where=denominator > 0.0,
    )
    weights[node_mask] = shape_blend_weights(raw, blend)
    return weights


def nearest_moving_displacements(
    points: np.ndarray,
    query_mask: np.ndarray,
    moving_mask: np.ndarray,
    displacement: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    moving_indices = np.flatnonzero(moving_mask)
    if len(moving_indices) == 0:
        raise ValueError("displacement interpolation requires at least one moving node")
    _, local_indices = nearest_reference(
        np.asarray(points)[query_mask], np.asarray(points)[moving_mask], chunk_size
    )
    return displacement[moving_indices[local_indices]]
