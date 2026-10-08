from __future__ import annotations
import random
from ..models.schemas import City
from .distance import calculate_route_distance


def _start_index(city_ids: list[str], starting_city: str | None, rng: random.Random) -> int:
    if starting_city and starting_city in city_ids: return city_ids.index(starting_city)
    return rng.randrange(len(city_ids))

def nearest_neighbor(city_ids, matrix, starting_city=None, rng=None):
    rng = rng or random.Random()
    start = _start_index(city_ids, starting_city, rng)
    unvisited = set(range(len(city_ids))); unvisited.remove(start)
    route = [city_ids[start]]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: (matrix[current, j], city_ids[j]))
        route.append(city_ids[nxt]); unvisited.remove(nxt); current = nxt
    return route

def insertion(city_ids, matrix, mode, starting_city=None, rng=None):
    rng = rng or random.Random()
    start = _start_index(city_ids, starting_city, rng)
    remaining = set(range(len(city_ids))); remaining.remove(start)
    if not remaining: return [city_ids[start]]
    second = min(remaining, key=lambda j: (matrix[start, j], city_ids[j]))
    if mode == "farthest":
        second = max(remaining, key=lambda j: (matrix[start, j], city_ids[j]))
    route = [start, second]; remaining.remove(second)
    while remaining:
        if mode == "nearest":
            k = min(remaining, key=lambda j: min(matrix[j, r] for r in route))
        elif mode == "farthest":
            k = max(remaining, key=lambda j: min(matrix[j, r] for r in route))
        elif mode == "cheapest":
            k = min(remaining, key=lambda j: min(matrix[route[t], j] + matrix[j, route[(t+1)%len(route)]] - matrix[route[t], route[(t+1)%len(route)]] for t in range(len(route))))
        else:  # arbitrary
            k = min(remaining)
        best_pos = None; best_delta = float("inf")
        for t in range(len(route)):
            a, b = route[t], route[(t+1)%len(route)]
            delta = matrix[a,k] + matrix[k,b] - matrix[a,b]
            if delta < best_delta:
                best_delta = delta; best_pos = t+1
        route.insert(best_pos, k); remaining.remove(k)
    return [city_ids[i] for i in route]

def construct(method: str, city_ids, matrix, starting_city=None, rng=None, manual_route=None):
    if method == "nn": return nearest_neighbor(city_ids, matrix, starting_city, rng)
    if method == "ni": return insertion(city_ids, matrix, "nearest", starting_city, rng)
    if method == "ci": return insertion(city_ids, matrix, "cheapest", starting_city, rng)
    if method == "fi": return insertion(city_ids, matrix, "farthest", starting_city, rng)
    if method == "ai": return insertion(city_ids, matrix, "arbitrary", starting_city, rng)
    if method == "manual": return list(manual_route or [])
    raise ValueError(f"Unknown construction method: {method}")
