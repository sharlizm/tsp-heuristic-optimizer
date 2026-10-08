import random
import numpy as np
from backend.app.algorithms.distance import generate_distance_matrix, calculate_route_distance
from backend.app.algorithms.construction import construct
from backend.app.algorithms.neighborhood import generate_neighbors, apply_move
from backend.app.algorithms.local_search import local_search
from backend.app.algorithms.simulated_annealing import simulated_annealing
from backend.app.algorithms.tabu_search import tabu_search
from backend.app.models.schemas import City

CITIES=[City(id=f"C{i}",x=x,y=y) for i,(x,y) in enumerate([(10,20),(20,40),(35,15),(50,35),(65,20),(80,45)],1)]

def test_distances():
    ids,e=generate_distance_matrix(CITIES,"euclidean"); _,m=generate_distance_matrix(CITIES,"manhattan")
    assert ids[0]=="C1"; assert abs(e[0,1]-np.sqrt(500))<1e-9; assert m[0,1]==30

def test_construction_methods_valid():
    ids,m=generate_distance_matrix(CITIES,"euclidean")
    for method in ["nn","ni","ci","fi","ai"]:
        r=construct(method,ids,m,"C1",random.Random(1)); assert sorted(r)==sorted(ids); assert len(r)==len(ids)

def test_route_and_neighbors():
    route=[c.id for c in CITIES]
    ids,m=generate_distance_matrix(CITIES,"euclidean")
    for op in ["swap","2-opt","3-opt"]:
        ns=generate_neighbors(route,[op],10,random.Random(1)); assert ns and all(sorted(c)==sorted(route) for _,c in ns)

def test_local_search_sa_ts():
    ids,m=generate_distance_matrix(CITIES,"euclidean"); route=construct("nn",ids,m,"C1",random.Random(1))
    ls,ld,_=local_search(route,ids,m,["2-opt"],"best",rng=random.Random(9)); assert ld <= calculate_route_distance(route,ids,m)+1e-9
    sa=simulated_annealing(ls,ids,m,initial_temperature=100,alpha=.9,min_temperature=.01,max_iteration=40,operator="2-opt",rng=random.Random(5))
    ts=tabu_search(ls,ids,m,max_iteration=30,tabu_size=5,neighborhood_size=10,operator="2-opt",rng=random.Random(5))
    assert len(sa["history"])>0 and len(sa["temperature_history"])==len(sa["history"])
    assert len(ts["history"])>0 and sorted(sa["best_route"])==sorted(ids) and sorted(ts["best_route"])==sorted(ids)
