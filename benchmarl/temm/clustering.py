#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass
class ClusterResult:
    labels: Tensor
    variances: Tensor
    medoid_indices: Tensor
    normalized_intra_variance: float


def pairwise_distances(embeddings: Tensor, metric: str = "euclidean") -> Tensor:
    embeddings = embeddings.float()
    if embeddings.numel() == 0:
        return torch.zeros((0, 0), dtype=torch.float)
    if metric == "euclidean":
        distances = torch.cdist(embeddings, embeddings)
    elif metric == "cosine":
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=-1, eps=1e-8)
        distances = 1.0 - normalized @ normalized.T
        distances.clamp_(min=0.0)
    else:
        raise ValueError(f"Unsupported TEMM distance metric: {metric}")
    return distances


def agglomerative_cluster(
    embeddings: Tensor,
    threshold: float,
    metric: str = "euclidean",
    prefer_sklearn: bool = True,
) -> ClusterResult:
    """Cluster embeddings with optional sklearn and a small pure-torch fallback."""

    embeddings = embeddings.float().detach().cpu()
    n_items = embeddings.shape[0]
    if n_items == 0:
        empty = torch.empty(0, dtype=torch.long)
        return ClusterResult(empty, torch.empty(0), empty, 0.0)
    if n_items == 1:
        return ClusterResult(
            labels=torch.zeros(1, dtype=torch.long),
            variances=torch.zeros(1),
            medoid_indices=torch.zeros(1, dtype=torch.long),
            normalized_intra_variance=0.0,
        )

    if prefer_sklearn and importlib.util.find_spec("sklearn") is not None:
        labels = _sklearn_agglomerative(embeddings, threshold, metric)
    else:
        labels = _fallback_agglomerative(embeddings, threshold, metric)
    return summarize_clusters(embeddings, labels, metric)


def choose_best_clustering(
    embeddings: Tensor,
    thresholds: Sequence[float],
    metric: str,
    cluster_penalty: float,
) -> tuple[ClusterResult, float]:
    best_result = None
    best_threshold = None
    best_score = float("inf")
    for threshold in thresholds:
        result = agglomerative_cluster(embeddings, threshold, metric)
        n_clusters = len(torch.unique(result.labels)) if result.labels.numel() else 0
        score = result.normalized_intra_variance + cluster_penalty * n_clusters
        if score < best_score:
            best_score = score
            best_result = result
            best_threshold = float(threshold)
    assert best_result is not None and best_threshold is not None
    return best_result, best_threshold


def summarize_clusters(
    embeddings: Tensor, labels: Tensor, metric: str
) -> ClusterResult:
    distances = pairwise_distances(embeddings, metric)
    global_scale = float(distances.max().item()) if distances.numel() else 0.0
    if global_scale <= 1e-8:
        global_scale = 1.0

    variances = []
    medoid_indices = []
    normalized_sum = 0.0
    for label in sorted(int(label) for label in torch.unique(labels)):
        indices = (labels == label).nonzero(as_tuple=True)[0]
        cluster_embeddings = embeddings[indices]
        centroid = cluster_embeddings.mean(0)
        sq_dist = ((cluster_embeddings - centroid) ** 2).sum(-1)
        variance = float(sq_dist.mean().item()) if sq_dist.numel() else 0.0
        cluster_distances = distances[indices][:, indices]
        medoid_local = int(cluster_distances.mean(1).argmin().item())
        variances.append(variance)
        medoid_indices.append(int(indices[medoid_local].item()))
        normalized_sum += (variance**0.5 / global_scale) * len(indices)

    normalized = normalized_sum / max(1, embeddings.shape[0])
    return ClusterResult(
        labels=labels.long(),
        variances=torch.tensor(variances, dtype=torch.float),
        medoid_indices=torch.tensor(medoid_indices, dtype=torch.long),
        normalized_intra_variance=float(max(0.0, min(1.0, normalized))),
    )


def _fallback_agglomerative(
    embeddings: Tensor, threshold: float, metric: str
) -> Tensor:
    distances = pairwise_distances(embeddings, metric)
    max_distance = float(distances.max().item()) if distances.numel() else 0.0
    cutoff = threshold * max_distance if threshold <= 1.0 else threshold
    clusters = [{index} for index in range(embeddings.shape[0])]

    while len(clusters) > 1:
        best_pair = None
        best_distance = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                left = torch.tensor(sorted(clusters[i]), dtype=torch.long)
                right = torch.tensor(sorted(clusters[j]), dtype=torch.long)
                average = float(distances[left][:, right].mean().item())
                if average < best_distance:
                    best_distance = average
                    best_pair = (i, j)
        if best_pair is None or best_distance > cutoff:
            break
        i, j = best_pair
        clusters[i] = clusters[i].union(clusters[j])
        del clusters[j]

    labels = torch.empty(embeddings.shape[0], dtype=torch.long)
    for label, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = label
    return _compact_labels(labels)


def _sklearn_agglomerative(embeddings: Tensor, threshold: float, metric: str) -> Tensor:
    from sklearn.cluster import AgglomerativeClustering

    distances = pairwise_distances(embeddings, metric)
    max_distance = float(distances.max().item()) if distances.numel() else 0.0
    cutoff = threshold * max_distance if threshold <= 1.0 else threshold
    kwargs = {
        "n_clusters": None,
        "distance_threshold": cutoff,
        "linkage": "average",
    }
    try:
        model = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="precomputed", **kwargs)
    labels = torch.tensor(model.fit_predict(distances.numpy()), dtype=torch.long)
    return _compact_labels(labels)


def _compact_labels(labels: Tensor) -> Tensor:
    unique_labels = sorted(torch.unique(labels).tolist())
    mapping = {int(label): index for index, label in enumerate(unique_labels)}
    return torch.tensor([mapping[int(label)] for label in labels], dtype=torch.long)
