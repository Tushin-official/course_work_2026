# pair_search.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from qubo_cycle import build_cycle_qubo, VarMap
from ising_mapping import qubo_to_ising

# Import your SB solvers (BSB/DSB) from where you defined them
from SB_modules import BSB, DSB

@dataclass
class CycleSolution:
    ok: bool
    cycle_nodes: List[int]            # includes 0 at start and end, e.g. [0, short, ..., long, 0]
    short_node: int                   # node after 0
    long_node: int                    # node before 0
    path_nodes: List[int]             # [short, ..., long]
    path_weight: float                # sum of w along short->...->long
    edges_selected: List[Tuple[int,int]]

def _decode_edges(z: np.ndarray, var_map: VarMap) -> List[Tuple[int,int]]:
    z = z.astype(int).reshape(-1)
    edges = []
    for u, bit in enumerate(z):
        if bit == 1:
            edges.append(var_map.u2e[u])
    return edges

def verify_single_dummy_cycle(
    selected_edges: List[Tuple[int,int]],
    N: int,
) -> Tuple[bool, Optional[List[int]]]:
    """
    Verify the solution is exactly ONE directed cycle containing dummy node 0,
    excluding:
      - a cycle without dummy
      - split cycles
    as described around Fig.3 in the paper.
    Returns (ok, cycle_nodes_with_0_start_end).
    """
    if not selected_edges:
        return False, None

    out = {i: [] for i in range(N+1)}
    inn = {i: [] for i in range(N+1)}
    for i, j in selected_edges:
        out[i].append(j)
        inn[j].append(i)

    # degree checks: in/out must be <=1
    for i in range(N+1):
        if len(out[i]) > 1 or len(inn[i]) > 1:
            return False, None
        if len(out[i]) != len(inn[i]):  # flow balance
            return False, None

    # dummy must be used
    if len(out[0]) != 1 or len(inn[0]) != 1:
        return False, None

    # walk cycle from dummy
    cycle = [0]
    cur = 0
    visited = {0}
    while True:
        nxts = out[cur]
        if len(nxts) != 1:
            return False, None
        nxt = nxts[0]
        cycle.append(nxt)
        if nxt == 0:
            break
        if nxt in visited:
            return False, None
        visited.add(nxt)
        cur = nxt

    # ensure all selected edges are exactly those on this cycle
    cycle_edges = set(zip(cycle[:-1], cycle[1:]))
    if len(cycle_edges) != len(selected_edges):
        return False, None
    if set(selected_edges) != cycle_edges:
        return False, None

    # forbid trivial 0->a->0 (would mean short==long)
    if len(cycle) < 4:  # [0, a, b, 0] is minimum valid for two different stocks
        return False, None

    return True, cycle

def extract_pair_and_weight(
    cycle: List[int],
    w: np.ndarray,
) -> Tuple[int,int,List[int],float]:
    """
    cycle: [0, short, ..., long, 0]
    short = cycle[1], long = cycle[-2]
    path is short->...->long along the cycle (excluding dummy).
    """
    short = cycle[1]
    long = cycle[-2]

    # path nodes are the interior segment from short to long
    path_nodes = cycle[1:-1]  # [short, ..., long]

    # compute weight along path short->...->long
    total = 0.0
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        total += float(w[a, b])
    return short, long, path_nodes, total

class TabuList:
    """
    Stores tabu as matrix T[i,j] used exactly like Eq.(3): T[i,j]*b_{0,j}*b_{i,0}.
    Interpreting endpoints:
      successor of dummy is j (short),
      predecessor of dummy is i (long).
    If you want to forbid (short, long) in human terms, call forbid(short, long),
    which sets T[long, short] = 1 (matching Eq.(3)).
    """
    def __init__(self, N: int):
        self.N = N
        self.T = np.zeros((N+1, N+1), dtype=int)

    def forbid(self, short: int, long: int):
        if short == long:
            return
        self.T[long, short] = 1

    def clear(self):
        self.T.fill(0)

def solve_best_cycle_with_sb(
    w: np.ndarray,
    tabu: Optional[TabuList] = None,
    sb_variant: str = "BSB",
    n_runs: int = 20,
    n_iter: int = 200,
    dt: float = 0.65,
    mc: float = 1.0,
    mp: Optional[float] = None,
    seed: Optional[int] = None,
    return_best_traj: bool = False,
):
    """
    Runs SB multiple times with different seeds and returns best valid cycle solution.
    """
    w = np.asarray(w, dtype=float)
    N = w.shape[0] - 1
    if tabu is None:
        tabu = TabuList(N)

    lin, quad, var_map = build_cycle_qubo(w, tabu=tabu.T, mc=mc, mp=mp)
    J, h, _ = qubo_to_ising(lin, quad)

    best: Optional[CycleSolution] = None
    best_traj = None
    base_seed = 0 if seed is None else int(seed)

    for r in range(n_runs):
        run_seed = base_seed + r

        if sb_variant.upper() == "BSB":
            solver = BSB(J, h=h, n_iter=n_iter, dt=dt, seed=run_seed)
        elif sb_variant.upper() == "DSB":
            solver = DSB(J, h=h, n_iter=n_iter, dt=dt, seed=run_seed)
        else:
            raise ValueError("sb_variant must be 'BSB' or 'DSB'")

        # if return_best_traj:
        #     traj = solver.run(record_trajectory=True)
        # else:
        solver.run(record_trajectory=False)
        traj = None

        # spins from x
        x = solver.x.reshape(-1)
        s = np.where(x >= 0.0, 1, -1)
        z = ((s + 1) // 2).astype(int)

        selected_edges = _decode_edges(z, var_map)

        ok, cycle = verify_single_dummy_cycle(selected_edges, N)
        if not ok or cycle is None:
            continue

        short, long, path_nodes, path_weight = extract_pair_and_weight(cycle, w)
        # Hard tabu verification after decoding.
        # QUBO tabu is only a penalty, but SB is heuristic,
        # so we additionally reject already forbidden endpoint pairs.
        if tabu is not None and int(tabu.T[long, short]) == 1:
            continue

        sol = CycleSolution(
            ok=True,
            cycle_nodes=cycle,
            short_node=short,
            long_node=long,
            path_nodes=path_nodes,
            path_weight=path_weight,
            edges_selected=selected_edges,
        )

        if best is None or sol.path_weight < best.path_weight:
            best = sol
            if return_best_traj:
                best_traj = traj

    return (best, best_traj) if return_best_traj else best

def find_multiple_pairs(
    w: np.ndarray,
    threshold: float,
    max_pairs: int = 10,
    sb_variant: str = "BSB",
    n_runs_per_pair: int = 30,
    n_iter: int = 200,
    dt: float = 0.65,
    mc: float = 1.0,
    mp: Optional[float] = None,
    seed: Optional[int] = None,
    debug: bool = False,
    # NEW:
    return_traj: bool = False,
):
    w = np.asarray(w, dtype=float)
    N = w.shape[0] - 1
    tabu = TabuList(N)

    results = []
    base_seed = 0 if seed is None else int(seed)

    for k in range(max_pairs):
        if return_traj:
            sol, traj = solve_best_cycle_with_sb(
                w=w, tabu=tabu, sb_variant=sb_variant,
                n_runs=n_runs_per_pair, n_iter=n_iter, dt=dt,
                mc=mc, mp=mp, seed=base_seed + 1000 * k,
                return_best_traj=True,
            )
        else:
            sol = solve_best_cycle_with_sb(
                w=w, tabu=tabu, sb_variant=sb_variant,
                n_runs=n_runs_per_pair, n_iter=n_iter, dt=dt,
                mc=mc, mp=mp, seed=base_seed + 1000 * k,
                return_best_traj=False,
            )
            traj = None

        if sol is None:
            if debug:
                print(f"[k={k}] no valid dummy-cycle found")
            break

        if debug:
            print(f"[k={k}] best valid cycle weight={sol.path_weight}")

        if sol.path_weight >= threshold:
            if debug:
                print(f"[k={k}] stop: weight {sol.path_weight} >= threshold {threshold}")
            break

        results.append((sol, traj) if return_traj else sol)
        tabu.forbid(short=sol.short_node, long=sol.long_node)

    return results
