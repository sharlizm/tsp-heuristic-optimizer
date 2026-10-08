from __future__ import annotations
from typing import Iterable
from ..models.schemas import City

def validate_cities(cities: list[City]) -> list[str]:
    errors: list[str] = []
    if len(cities) < 3:
        errors.append("Minimal 3 kota diperlukan.")
    ids = [c.id for c in cities]
    if len(ids) != len(set(ids)):
        errors.append("City ID harus unik.")
    for i, c in enumerate(cities, start=1):
        if not c.id.strip(): errors.append(f"Baris {i}: City ID kosong.")
    return errors

def validate_route(route: Iterable[str], city_ids: Iterable[str]) -> tuple[bool, list[str]]:
    route = list(route); ids = list(city_ids)
    errors: list[str] = []
    if len(route) != len(ids): errors.append("Route harus mengunjungi semua kota tepat satu kali.")
    if len(set(route)) != len(route): errors.append("Route mengandung kota duplikat.")
    if set(route) != set(ids): errors.append("Route memiliki kota yang hilang atau tidak dikenal.")
    return (not errors, errors)
