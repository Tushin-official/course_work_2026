# qubo_cycle.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

Edge = Tuple[int, int]          # (i,j)
QuadKey = Tuple[int, int]       # (u,v) with u < v

@dataclass(frozen=True)
class VarMap:
    """
    Mapping between directed edges (i,j) and QUBO variable index u in [0..M-1].
    """
    N: int                       # number of stock nodes (without dummy)
    edges: List[Edge]
    e2u: Dict[Edge, int]
    u2e: Dict[int, Edge]

def build_var_map(N: int) -> VarMap:
    nodes = list(range(0, N + 1))  # 0..N (0 is dummy)
    edges: List[Edge] = []
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            edges.append((i, j))
    e2u = {e: k for k, e in enumerate(edges)}
    u2e = {k: e for e, k in e2u.items()}
    return VarMap(N=N, edges=edges, e2u=e2u, u2e=u2e)

def _add_lin(lin: np.ndarray, u: int, coeff: float):
    lin[u] += coeff

def _add_quad(quad: Dict[QuadKey, float], u: int, v: int, coeff: float):
    if u == v:
        # should not happen for our construction (quadratic only between distinct vars)
        return
    a, b = (u, v) if u < v else (v, u)
    quad[(a, b)] = quad.get((a, b), 0.0) + coeff

def build_cycle_qubo(
    w: np.ndarray,
    tabu: Optional[np.ndarray] = None,
    mc: float = 1.0,
    mp: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[QuadKey, float], VarMap]:
    """
    Build QUBO for the optimal cycle (dummy-node formulation).

    Returns:
      lin[u]                     linear coefficients
      quad[(u,v)] for u<v        quadratic coefficients
      var_map                    mapping u <-> (i,j)

    Semantics: minimize
      E(z) = sum_u lin[u]*z_u + sum_{u<v} quad[u,v]*z_u*z_v,
    where z_u in {0,1} corresponds to b_{i,j}.
    """
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("w must be square")
    N = w.shape[0] - 1
    if w.shape != (N + 1, N + 1):
        raise ValueError("w must be (N+1,N+1) with dummy node 0")

    var_map = build_var_map(N)
    M = len(var_map.edges)

    if tabu is None:
        tabu = np.zeros((N + 1, N + 1), dtype=int)
    else:
        tabu = np.asarray(tabu, dtype=int)
        if tabu.shape != (N + 1, N + 1):
            raise ValueError("tabu must be (N+1,N+1) aligned with w (including dummy)")

    # choose penalty strength if not provided
    if mp is None:
        max_abs_w = float(np.max(np.abs(w))) if w.size else 1.0
        mp = 10.0 * max(1.0, max_abs_w)

    lin = np.zeros(M, dtype=float)
    quad: Dict[QuadKey, float] = {}

    # -------------------
    # H_cost = sum w_ij b_ij
    # -------------------
    for (i, j), u in var_map.e2u.items():
        coeff = w[i, j]
        if coeff != 0.0:
            _add_lin(lin, u, mc * coeff)

    # Precompute outgoing/incoming edge var indices
    out_vars: List[List[int]] = [[] for _ in range(N + 1)]
    in_vars: List[List[int]] = [[] for _ in range(N + 1)]
    for (i, j), u in var_map.e2u.items():
        out_vars[i].append(u)
        in_vars[j].append(u)

    # -------------------
    # H_penalty terms from Eq.(3) in the paper
    # -------------------

    # (1) outflow <= 1: sum_i sum_{j != j'} b_{i,j} b_{i,j'}
    for i in range(N + 1):
        vs = out_vars[i]
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                _add_quad(quad, vs[a], vs[b], mp * 1.0)

    # (2) inflow <= 1: sum_j sum_{i != i'} b_{i,j} b_{i',j}
    for j in range(N + 1):
        vs = in_vars[j]
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                _add_quad(quad, vs[a], vs[b], mp * 1.0)

    # (3) flow balance: sum_i (out_i - in_i)^2
    # Expand (sum out)^2 + (sum in)^2 - 2 (sum out)(sum in)
    for i in range(N + 1):
        outs = out_vars[i]
        ins = in_vars[i]

        # out^2 contributes: sum z + 2 sum_{a<b} z_a z_b
        for u in outs:
            _add_lin(lin, u, mp * 1.0)
        for a in range(len(outs)):
            for b in range(a + 1, len(outs)):
                _add_quad(quad, outs[a], outs[b], mp * 2.0)

        # in^2 contributes
        for u in ins:
            _add_lin(lin, u, mp * 1.0)
        for a in range(len(ins)):
            for b in range(a + 1, len(ins)):
                _add_quad(quad, ins[a], ins[b], mp * 2.0)

        # -2*out*in cross
        for u in outs:
            for v in ins:
                _add_quad(quad, u, v, mp * (-2.0))

    # (4) forbid both directions on same undirected edge: sum_{i,j} b_{i,j} b_{j,i}
    # Use i<j to avoid double counting (same effect up to scaling).
    for i in range(N + 1):
        for j in range(i + 1, N + 1):
            u = var_map.e2u[(i, j)]
            v = var_map.e2u[(j, i)]
            _add_quad(quad, u, v, mp * 1.0)

    # (5) tabu: sum_{i,j} T_{i,j} b_{0,j} b_{i,0}
    # IMPORTANT: this follows Eq.(3) exactly (indices as in the paper).
    # In terms of "pair endpoints": successor of dummy is j, predecessor is i.
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if tabu[i, j] == 0:
                continue
            u = var_map.e2u[(0, j)]   # b_{0,j}
            v = var_map.e2u[(i, 0)]   # b_{i,0}
            _add_quad(quad, u, v, mp * float(tabu[i, j]))

    return lin, quad, var_map