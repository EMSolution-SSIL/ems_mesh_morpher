import numpy as np


def shape_blend_weights(values: np.ndarray, blend: str) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    if blend == "linear":
        return clipped
    if blend == "smoothstep":
        return clipped * clipped * (3.0 - 2.0 * clipped)
    raise ValueError(f"unsupported blend: {blend}")
