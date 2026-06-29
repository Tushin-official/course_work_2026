# dtw_similarity.py
from __future__ import annotations
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Tuple

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    INF = 1e100
    dp = np.full((n + 1, m + 1), INF, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / (n + m))

def _utc_day(ts_ms: int) -> int:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return int(dt.strftime("%Y%m%d"))

def build_daily_sequences_from_hourly_klines(klines_1h: List[list]) -> Dict[int, np.ndarray]:
    """
    1h klines -> {YYYYMMDD: close/base_open_sequence}
    base_open = first hour open of that UTC day.
    """
    per_day: Dict[int, List[Tuple[int, float, float]]] = {}  # day -> list(open_time, open, close)
    for k in klines_1h:
        open_time = int(k[0])
        o = float(k[1])
        c = float(k[4])
        day = _utc_day(open_time)
        per_day.setdefault(day, []).append((open_time, o, c))

    out: Dict[int, np.ndarray] = {}
    for day, vals in per_day.items():
        vals.sort(key=lambda x: x[0])
        base = vals[0][1] if vals[0][1] != 0 else 1.0
        out[day] = np.array([c / base for (_, _, c) in vals], dtype=float)
    return out

def compute_similarity_matrix(
    symbols: List[str],
    daily_seq: Dict[str, Dict[int, np.ndarray]],
    days_back: int = 5,
    exclude_last_day: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    Returns (S, used_days)
    S in [0,1], diagonal = 0.
    If exclude_last_day=True, drops the latest UTC day before taking last `days_back`.
    """
    common_days = None
    for sym in symbols:
        days = set(daily_seq[sym].keys())
        common_days = days if common_days is None else (common_days & days)
    if not common_days:
        raise ValueError("No common days across symbols for similarity calculation.")

    days_sorted = sorted(common_days)
    if exclude_last_day and len(days_sorted) >= 2:
        days_sorted = days_sorted[:-1]  # drop latest day (often "today" partial)

    used_days = days_sorted[-days_back:] if len(days_sorted) >= days_back else days_sorted
    if len(used_days) == 0:
        raise ValueError("No days left for similarity after excluding last day.")

    N = len(symbols)
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            ds = []
            for day in used_days:
                a = daily_seq[symbols[i]][day]
                b = daily_seq[symbols[j]][day]
                ds.append(dtw_distance(a, b))
            d = float(np.mean(ds)) if ds else 0.0
            D[i, j] = D[j, i] = d

    # distance -> similarity via min-max inversion
    off_diag = D[~np.eye(N, dtype=bool)]
    dmin = float(np.min(off_diag)) if off_diag.size else 0.0
    dmax = float(np.max(off_diag)) if off_diag.size else 1.0
    if dmax <= dmin + 1e-12:
        S = np.ones((N, N), dtype=float)
    else:
        S = 1.0 - (D - dmin) / (dmax - dmin)
        S = np.clip(S, 0.0, 1.0)

    np.fill_diagonal(S, 0.0)
    return S, used_days
