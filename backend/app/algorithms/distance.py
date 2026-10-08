from __future__ import annotations
import numpy as np
from ..models.schemas import City

def generate_distance_matrix(cities: list[City], metric: str) -> tuple[list[str], np.ndarray]:
    ids = [c.id for c in cities]
    xy = np.array([[c.x, c.y] for c in cities], dtype=float)
    diff = xy[:, None, :] - xy[None, :, :]
    if metric == "euclidean":
        mat = np.sqrt(np.sum(diff * diff, axis=2))
    elif metric == "manhattan":
        mat = np.sum(np.abs(diff), axis=2)
    else:
        raise ValueError(f"Metric tidak didukung: {metric}")
    np.fill_diagonal(mat, 0.0)
    return ids, mat

def calculate_route_distance(route: list[str], city_ids: list[str], matrix: np.ndarray) -> float:
    if not route: return 0.0
    pos = {cid: i for i, cid in enumerate(city_ids)}
    idx = [pos[c] for c in route]
    return float(sum(matrix[idx[i], idx[(i+1) % len(idx)]] for i in range(len(idx))))
