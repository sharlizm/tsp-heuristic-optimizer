from __future__ import annotations
import math, random, time
from .distance import calculate_route_distance
from .neighborhood import generate_neighbors

def simulated_annealing(route, city_ids, matrix, *, initial_temperature, alpha, min_temperature, max_iteration,
                         operator, stopping_condition="both", reheating=False, reheat_interval=100, rng=None):
    rng = rng or random.Random()
    start_t=time.perf_counter()
    current=route[:]; current_d=calculate_route_distance(current,city_ids,matrix)
    best=current[:]; best_d=current_d; temp=initial_temperature
    history=[]; temperatures=[]
    for it in range(1,max_iteration+1):
        neigh=generate_neighbors(current,[operator],1,rng)
        move,candidate=neigh[0]
        cand_d=calculate_route_distance(candidate,city_ids,matrix)
        delta=cand_d-current_d
        prob=1.0 if delta <= 0 else math.exp(-delta/max(temp,1e-12))
        accepted = delta <= 0 or rng.random() < prob
        if accepted:
            current = candidate
            current_d = cand_d
        if current_d < best_d:
            best=current[:]; best_d=current_d
        history.append({"iteration":it,"current_route":current[:],"current_distance":current_d,
                        "best_route":best[:],"best_distance":best_d,"temperature":temp,
                        "candidate_route":candidate[:],"candidate_distance":cand_d,"delta":delta,
                        "acceptance":accepted,"acceptance_probability":prob,"move":repr(move)})
        temperatures.append(temp)
        temp *= alpha
        if reheating and it % reheat_interval == 0 and temp < initial_temperature * 0.1:
            temp = initial_temperature
        if stopping_condition == "temperature" and temp <= min_temperature: break
        if stopping_condition == "both" and temp <= min_temperature: break
    return {"best_route":best,"best_distance":best_d,"iterations":len(history),"runtime":time.perf_counter()-start_t,
            "history":history,"temperature_history":temperatures}
