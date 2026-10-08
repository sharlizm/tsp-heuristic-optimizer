from __future__ import annotations
from itertools import combinations

def apply_move(route: list[str], move: tuple) -> list[str]:
    r = route[:]
    kind = move[0]
    if kind == "swap":
        _, i, j = move; r[i], r[j] = r[j], r[i]
    elif kind == "2-opt":
        _, i, j = move; r[i:j+1] = reversed(r[i:j+1])
    elif kind == "3-opt":
        _, i, j, k, pattern = move
        prefix, a, b, c, suffix = r[:i], r[i:j], r[j:k], r[k:k+1], r[k+1:]
        # Four valid 3-opt-style reconnects from three cut points, keeping the closing city implicit.
        choices = [
            prefix + a[::-1] + b + c + suffix,
            prefix + a + b[::-1] + c + suffix,
            prefix + b + a + c + suffix,
            prefix + b[::-1] + a + c + suffix,
        ]
        return choices[pattern % len(choices)]
    return r

def move_key(move: tuple) -> str:
    return repr(move)

def generate_moves(n: int, operators: list[str]) -> list[tuple]:
    moves: list[tuple] = []
    if "swap" in operators:
        moves.extend(("swap", i, j) for i, j in combinations(range(n), 2))
    if "2-opt" in operators:
        moves.extend(("2-opt", i, j) for i in range(n-1) for j in range(i+1, n))
    if "3-opt" in operators and n >= 6:
        for i in range(1, n-4):
            for j in range(i+1, n-2):
                for k in range(j+1, n-1):
                    for p in range(4): moves.append(("3-opt", i, j, k, p))
    return moves

def generate_neighbors(route: list[str], operators: list[str], max_size: int | None = None, rng=None):
    import random
    rng = rng or random.Random()
    moves = generate_moves(len(route), operators)
    rng.shuffle(moves)
    if max_size: moves = moves[:max_size]
    return [(m, apply_move(route, m)) for m in moves]
