from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

Metric = Literal["euclidean", "manhattan"]
Construction = Literal["nn", "ni", "ci", "fi", "ai", "manual"]
Operator = Literal["swap", "2-opt", "3-opt"]
Strategy = Literal["first", "best"]
StartingMode = Literal["random", "specific", "manual"]

class City(BaseModel):
    id: str = Field(min_length=1)
    x: float
    y: float

class RouteRequest(BaseModel):
    route: list[str]

class OptimizeRequest(BaseModel):
    cities: list[City]
    distance_metric: Metric = "euclidean"
    initial_solution: Construction = "nn"
    starting_mode: StartingMode = "random"
    starting_city: str | None = None
    manual_route: list[str] | None = None
    local_search_swap: bool = False
    local_search_2opt: bool = True
    local_search_3opt: bool = False
    local_search_strategy: Strategy = "best"
    local_search_before_sa: bool = True
    local_search_after_sa: bool = False
    local_search_after_ts: bool = False
    sa_enabled: bool = True
    ts_enabled: bool = True
    sa_initial_temperature: float = Field(default=100.0, gt=0)
    sa_alpha: float = Field(default=0.95, gt=0, lt=1)
    sa_min_temperature: float = Field(default=0.01, gt=0)
    sa_max_iteration: int = Field(default=500, ge=1, le=100000)
    sa_operator: Operator = "2-opt"
    sa_stopping_condition: Literal["temperature", "iterations", "both"] = "both"
    sa_reheating: bool = False
    sa_reheat_interval: int = Field(default=100, ge=1)
    ts_max_iteration: int = Field(default=250, ge=1, le=100000)
    ts_tabu_size: int = Field(default=10, ge=1, le=100000)
    ts_neighborhood_size: int = Field(default=50, ge=1, le=100000)
    ts_operator: Operator = "2-opt"
    ts_aspiration: bool = True
    ts_stopping_condition: Literal["iterations", "no_improvement", "both"] = "iterations"
    ts_no_improvement_limit: int = Field(default=50, ge=1)
    random_seed: int | None = 12345

    @field_validator("cities")
    @classmethod
    def min_cities(cls, v: list[City]) -> list[City]:
        if len(v) < 3:
            raise ValueError("Minimal 3 kota.")
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("City ID harus unik.")
        return v
