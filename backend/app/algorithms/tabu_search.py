from __future__ import annotations
import time, random
from .distance import calculate_route_distance
from .neighborhood import generate_neighbors, move_key

def tabu_search(route, city_ids, matrix, *, max_iteration, tabu_size, neighborhood_size,
                 operator, aspiration=True, stopping_condition="iterations", no_improvement_limit=50, rng=None):
    rng = rng or random.Random(); start=time.perf_counter()
    current=route[:]; current_d=calculate_route_distance(current,city_ids,matrix)
    best=current[:]; best_d=current_d
    tabu: list[tuple[str,int]]=[]; history=[]; no_improve=0
    for it in range(1,max_iteration+1):
        neighbors=generate_neighbors(current,[operator],neighborhood_size,rng)
        evaluated=[]
        for move,cand in neighbors:
            d=calculate_route_distance(cand,city_ids,matrix)
            key=move_key(move)
            is_tabu=any(k==key for k,_ in tabu)
            allowed=(not is_tabu) or (aspiration and d < best_d)
            evaluated.append((d,move,cand,is_tabu,allowed))
        admissible=[x for x in evaluated if x[4]]
        pool=admissible or evaluated
        if not pool: break
        d,move,cand,is_tabu,allowed=min(pool,key=lambda x:(x[0],repr(x[1])))
        selected_key=move_key(move)
        aspiration_used = bool(is_tabu and aspiration and d < best_d)
        removed=None
        current=cand; current_d=d
        tabu=[(k,age+1) for k,age in tabu]
        tabu.insert(0,(selected_key,0))
        if len(tabu)>tabu_size: removed=tabu.pop()[0]
        improved=False
        if current_d < best_d:
            best=current[:]; best_d=current_d; no_improve=0; improved=True
        else: no_improve += 1
        history.append({"iteration":it,"current_route":current[:],"current_distance":current_d,
                        "candidate_route":cand[:],"candidate_distance":d,"selected_move":repr(move),
                        "best_route":best[:],"best_distance":best_d,
                        "tabu_list":[{"move":k,"age":age} for k,age in tabu],"tabu_size":len(tabu),
                        "move_added":selected_key,"move_removed":removed,"was_tabu":is_tabu,
                        "aspiration_used":aspiration_used,"improved":improved,
                        "neighborhood_size":len(evaluated),"neighborhood":[{"move":repr(m),"distance":dd,"tabu":tb,"allowed":al} for dd,m,c,tb,al in evaluated]})
        if stopping_condition in ("no_improvement","both") and no_improve >= no_improvement_limit: break
    return {"best_route":best,"best_distance":best_d,"iterations":len(history),"runtime":time.perf_counter()-start,
            "history":history}
