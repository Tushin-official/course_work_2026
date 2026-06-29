import numpy as np


def _as_rate_vector(x, n: int, name: str) -> np.ndarray:
    """
    Convert scalar or array-like fee rate to vector of shape (n,).
    Rates are fractions, e.g. 0.001 means 0.1%.
    """
    arr = np.asarray(x, dtype=float)

    if arr.ndim == 0:
        value = float(arr)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return np.full(n, value, dtype=float)

    arr = arr.reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} must be scalar or have shape ({n},), got {arr.shape}")

    if np.any(arr < 0):
        raise ValueError(f"{name} must contain only non-negative values")

    return arr.astype(float)


def build_market_weights(
    ask: np.ndarray,
    bid: np.ndarray,
    sim: np.ndarray,
    fee_rate: float = 0.0,
    fee_buy_rate=None,
    fee_sell_rate=None,
) -> np.ndarray:
    """
    Build directed market graph weights with optional maker/taker fee.

    Nodes:
      0      dummy node
      1..N   assets

    Edge meaning:
      i -> j means short/sell asset i and long/buy asset j.

    Original weight:
      w[i,j] = sim[i-1,j-1] * (ask[j-1] - bid[i-1])

    Fee-adjusted weight:
      w_fee[i,j] =
          sim[i-1,j-1] * (ask[j-1] - bid[i-1])
          + fee_sell_rate[i-1] * bid[i-1]
          + fee_buy_rate[j-1] * ask[j-1]

    If fee_buy_rate and fee_sell_rate are not passed, fee_rate is used for both sides.

    Important:
      Fees are NOT multiplied by similarity.
      Dummy edges remain zero.
    """
    ask = np.asarray(ask, dtype=float).reshape(-1)
    bid = np.asarray(bid, dtype=float).reshape(-1)
    sim = np.asarray(sim, dtype=float)

    n = ask.shape[0]

    if bid.shape[0] != n:
        raise ValueError("ask and bid must have same length")

    if sim.shape != (n, n):
        raise ValueError(f"sim must be (N,N), got {sim.shape}")

    if fee_buy_rate is None:
        fee_buy = _as_rate_vector(fee_rate, n, "fee_rate")
    else:
        fee_buy = _as_rate_vector(fee_buy_rate, n, "fee_buy_rate")

    if fee_sell_rate is None:
        fee_sell = _as_rate_vector(fee_rate, n, "fee_rate")
    else:
        fee_sell = _as_rate_vector(fee_sell_rate, n, "fee_sell_rate")

    w = np.zeros((n + 1, n + 1), dtype=float)

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                continue

            short_idx = i - 1
            long_idx = j - 1

            gross = sim[short_idx, long_idx] * (
                ask[long_idx] - bid[short_idx]
            )

            fee_cost = (
                fee_sell[short_idx] * bid[short_idx]
                + fee_buy[long_idx] * ask[long_idx]
            )

            w[i, j] = gross + fee_cost

    return w