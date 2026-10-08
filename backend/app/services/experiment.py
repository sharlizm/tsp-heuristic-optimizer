from __future__ import annotations

import random
import statistics
from typing import Any

import numpy as np

from ..algorithms.construction import construct
from ..algorithms.distance import calculate_route_distance
from ..algorithms.simulated_annealing import simulated_annealing
from ..algorithms.tabu_search import tabu_search


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"best": 0.0, "worst": 0.0, "average": 0.0, "std_dev": 0.0}
    return {
        "best": min(values),
        "worst": max(values),
        "average": statistics.mean(values),
        "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _build_initial_route(payload: dict[str, Any], rng: random.Random):
    ids = payload["ids"]
    matrix = np.asarray(payload["matrix"], dtype=float)
    route = construct(
        payload["initial_solution"],
        ids,
        matrix,
        payload.get("starting_city"),
        rng,
        payload.get("manual_route"),
    )
    return route, ids, matrix


def _run_one(payload: dict[str, Any], algorithm: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    route, ids, matrix = _build_initial_route(payload, rng)
    initial_distance = calculate_route_distance(route, ids, matrix)

    common = {
        "run": payload.get("run_number"),
        "algorithm": algorithm.upper(),
        "initial_distance": initial_distance,
    }

    if algorithm == "sa":
        result = simulated_annealing(
            route,
            ids,
            matrix,
            rng=rng,
            initial_temperature=float(payload["sa_initial_temperature"]),
            alpha=float(payload["sa_alpha"]),
            min_temperature=float(payload["sa_min_temperature"]),
            max_iteration=int(payload["max_iteration"]),
            operator=payload["operator"],
        )
        return {
            **common,
            "best_distance": result["best_distance"],
            "runtime": result["runtime"],
            "iterations": result["iterations"],
        }

    result = tabu_search(
        route,
        ids,
        matrix,
        rng=rng,
        max_iteration=int(payload["max_iteration"]),
        tabu_size=int(payload["tabu_size"]),
        neighborhood_size=int(payload["neighborhood_size"]),
        operator=payload["operator"],
        aspiration=bool(payload.get("aspiration", True)),
        stopping_condition=payload.get("ts_stopping_condition", "iterations"),
        no_improvement_limit=int(payload.get("ts_no_improvement_limit", 50)),
    )
    return {
        **common,
        "best_distance": result["best_distance"],
        "runtime": result["runtime"],
        "iterations": result["iterations"],
    }


def multi_run(payload: dict[str, Any]) -> dict[str, Any]:
    runs = max(1, int(payload.get("runs", 1)))
    algorithm = payload.get("algorithm", "sa")
    base_seed = int(payload.get("seed", 12345))

    algorithms = ["sa", "ts"] if algorithm == "sa_vs_ts" else [algorithm]
    results: list[dict[str, Any]] = []

    for run_number in range(1, runs + 1):
        for offset, algo in enumerate(algorithms):
            cfg = dict(payload)
            cfg["run_number"] = run_number
            cfg["operator"] = payload.get("operator") or (payload.get("ts_operator") if algo == "ts" else payload.get("sa_operator"))
            cfg["max_iteration"] = int(payload.get("max_iteration", payload.get("ts_max_iteration", payload.get("sa_max_iteration", 300))))
            # Give every (run, algorithm) pair an independent, reproducible stream.
            seed = base_seed + (run_number - 1) * 100000 + offset
            results.append(_run_one(cfg, algo, seed))

    grouped: dict[str, dict[str, float]] = {}
    for algo in algorithms:
        values = [row["best_distance"] for row in results if row["algorithm"].lower() == algo]
        grouped[algo.upper()] = summarize(values)

    return {"runs": results, "summary": grouped}
