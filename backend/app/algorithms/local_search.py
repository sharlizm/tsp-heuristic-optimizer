from __future__ import annotations
from .distance import calculate_route_distance
from .neighborhood import generate_neighbors

def local_search(route, city_ids, matrix, operators, strategy="best", max_rounds=100, rng=None):
    if not operators: return route[:], calculate_route_distance(route, city_ids, matrix), []
    current = route[:]
    current_distance = calculate_route_distance(current, city_ids, matrix)
    history=[]
    for round_no in range(1, max_rounds+1):
        neigh = generate_neighbors(current, operators, rng=rng)
        best = None
        if strategy == "first":
            for move, cand in neigh:
                d = calculate_route_distance(cand, city_ids, matrix)
                if d + 1e-12 < current_distance:
                    best = (move,cand,d); break
        else:
            for move, cand in neigh:
                d = calculate_route_distance(cand, city_ids, matrix)
                if best is None or d < best[2]: best=(move,cand,d)
            if best and best[2] + 1e-12 >= current_distance: best=None
        if not best: break
        move,cand,d=best; current=cand; current_distance=d
        history.append({"round":round_no,"move":repr(move),"distance":d,"route":current[:]})
    return current,current_distance,history
