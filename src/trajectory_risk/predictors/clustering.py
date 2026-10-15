"""Turning N raw stochastic samples into k candidates with confidences.

Two methods, one per model, both ported from the upstream repos:

* ``kmeans_trajectories`` (ContextVAE's ``utils.clustering``): k-means on the
  whole flattened trajectory. Candidates are synthesized cluster CENTRES,
  confidence = cluster size / N.
* ``final_position_clustering`` (SocialVAE's ``utils.FPC``): "FPC" stands for
  Final Position Clustering, not farthest-point clustering. k-means on the
  GOAL (last point) only, then each centroid is replaced by the nearest real
  raw sample, so candidates are actual model draws. Upstream returns no
  confidence; we assign every raw sample to its nearest chosen goal and
  report the share, the same meaning as the k-means path.

Either way, confidence is "fraction of this model's samples near here",
which is not a calibrated probability of what will actually happen.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans

from .base import Candidate, sort_candidates


def kmeans_trajectories(samples: np.ndarray, k: int, seed: Optional[int] = None) -> List[Candidate]:
    """samples: (N, H, 2) -> k candidates (cluster centres), highest share first."""
    n, h, _ = samples.shape
    k = min(k, n)
    flat = samples.reshape(n, h * 2)
    km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(flat)
    counts = np.bincount(km.labels_, minlength=k)
    centres = km.cluster_centers_.reshape(k, h, 2)
    return sort_candidates([Candidate(centres[i], counts[i] / n) for i in range(k)])


def _kmeans_points(k: int, data: np.ndarray, rng: np.random.Generator, iters: int = 1000) -> np.ndarray:
    """Plain Lloyd's k-means on (N, 2) points, as in upstream SocialVAE utils.kmeans."""
    centroids = data[rng.permutation(len(data))[:k]].copy()
    for _ in range(iters):
        d = np.linalg.norm(data[None, :, :] - centroids[:, None, :], axis=-1)  # k x N
        closest = d.argmin(axis=0)
        new = np.array([data[closest == j].mean(axis=0) if np.any(closest == j)
                        else data[rng.integers(len(data))] for j in range(len(centroids))])
        if np.linalg.norm(new - centroids) < 1e-4:
            centroids = new
            break
        centroids = new
    return centroids


def final_position_clustering(samples: np.ndarray, k: int, seed: Optional[int] = None) -> List[Candidate]:
    """samples: (N, H, 2) -> k candidates that are real raw samples, highest share first."""
    rng = np.random.default_rng(seed)
    n = len(samples)
    k = min(k, n)
    goals = samples[:, -1, :]
    centroids = _kmeans_points(k, goals, rng)
    chosen = np.linalg.norm(centroids[:, None, :] - goals[None, :, :], axis=-1).argmin(axis=1)
    reps = samples[chosen]
    assign = np.linalg.norm(goals[:, None, :] - reps[None, :, -1, :], axis=-1).argmin(axis=1)
    counts = np.bincount(assign, minlength=k)
    return sort_candidates([Candidate(reps[i], counts[i] / n) for i in range(k)])
