"""Scale- and translation-normalized features for undirected city pairs."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from tsp_learning.problem import TSPInstance

EDGE_FEATURE_DIM = 5


def edge_feature_array(instance: TSPInstance) -> NDArray[np.float32]:
    """Return symmetric geometric features with shape ``(n, n, 5)``.

    Features are absolute coordinate offsets, Euclidean separation, and the
    sorted endpoint radii relative to the instance centroid. Scaling by the RMS
    radius makes the representation invariant to translation and uniform scale.
    """

    centered = instance.coordinates - instance.coordinates.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(centered, axis=1)
    scale = float(np.sqrt(np.mean(np.square(radii))))
    if scale <= np.finfo(np.float64).eps:
        scale = 1.0
    offsets = centered[:, np.newaxis, :] - centered[np.newaxis, :, :]
    absolute_offsets = np.abs(offsets) / scale
    separation = np.linalg.norm(offsets, axis=2) / scale
    radius_i = np.broadcast_to(radii[:, np.newaxis], separation.shape) / scale
    radius_j = np.broadcast_to(radii[np.newaxis, :], separation.shape) / scale
    features = np.stack(
        (
            absolute_offsets[:, :, 0],
            absolute_offsets[:, :, 1],
            separation,
            np.minimum(radius_i, radius_j),
            np.maximum(radius_i, radius_j),
        ),
        axis=2,
    )
    return np.asarray(features, dtype=np.float32)


def edge_feature_tensor(instance: TSPInstance) -> torch.Tensor:
    """Return edge features as a CPU float tensor."""

    return torch.from_numpy(edge_feature_array(instance))
