# verify_bruteforce.py
import numpy as np

def brute_force_min_simple_paths(w: np.ndarray):
    """
    Exact min simple-path costs for all ordered pairs among nodes 1..N (dummy excluded).
    Complexity: O(N^2 * 2^N). For N<=15 OK; for N=6 trivial.
    Returns dist[N+1, N+1], where dist[s,t] = min cost from s to t, s!=t, s,t in 1..N.
    """
    w = np.asarray(w, dtype=float)
    N = w.shape[0] - 1
    INF = 1e100

    dist = np.full((N+1, N+1), INF, dtype=float)

    # map graph nodes 1..N -> bit positions 0..N-1
    for s in range(1, N+1):
        dp = np.full((1<<N, N+1), INF, dtype=float)
        sbit = 1 << (s-1)
        dp[sbit, s] = 0.0

        for mask in range(1<<N):
            if (mask & sbit) == 0:
                continue
            for v in range(1, N+1):
                cur = dp[mask, v]
                if cur >= INF/2:
                    continue
                for u in range(1, N+1):
                    if u == v:
                        continue
                    ub = 1 << (u-1)
                    if mask & ub:
                        continue
                    nm = mask | ub
                    cand = cur + w[v, u]
                    if cand < dp[nm, u]:
                        dp[nm, u] = cand

        # take best over all masks that include s and t (any length)
        for t in range(1, N+1):
            if t == s:
                continue
            best = np.min(dp[:, t])
            dist[s, t] = best

    return dist

def sum_path_weight(w: np.ndarray, path_nodes):
    total = 0.0
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        total += float(w[a, b])
    return total

def verify_solutions(w: np.ndarray, solutions):
    dist = brute_force_min_simple_paths(w)

    # global best pair by exact search
    N = w.shape[0] - 1
    exact_best = None
    exact_best_val = 1e100
    for s in range(1, N+1):
        for t in range(1, N+1):
            if s == t:
                continue
            if dist[s, t] < exact_best_val:
                exact_best_val = dist[s, t]
                exact_best = (s, t)

    print("Exact global best:", exact_best, "value=", exact_best_val)

    for k, sol in enumerate(solutions):
        s = sol.short_node
        t = sol.long_node
        # your sol.path_nodes are without dummy, e.g. [s,...,t]
        calc = sum_path_weight(w, sol.path_nodes)
        exact = dist[s, t]

        print(f"\n[k={k}] short={s} long={t}")
        print("  reported:", sol.path_weight)
        print("  recomputed:", calc, "diff=", calc - sol.path_weight)
        print("  exact min simple-path:", exact, "gap=", sol.path_weight - exact)

        if abs(calc - sol.path_weight) > 1e-9:
            print("  !! mismatch: printed path weight != stored weight")

        # heuristic note: SB should have sol.path_weight close to exact (often equal for small N)
        # If gap > 0: SB found a suboptimal path for that pair.
