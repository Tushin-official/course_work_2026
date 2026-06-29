# binance_data.py
from __future__ import annotations
import requests
import time
from typing import Dict, List, Tuple

BASE_URL = "https://data-api.binance.vision"  # market-data-only base endpoint :contentReference[oaicite:4]{index=4}

def _get(path: str, params: dict | None = None, timeout: float = 15.0):
    url = BASE_URL + path
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_book_tickers(symbols: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Returns: {symbol: (bid, ask)}
    Endpoint: GET /api/v3/ticker/bookTicker :contentReference[oaicite:5]{index=5}
    """
    # Binance supports multi-symbol via `symbols` parameter too; but simplest: loop.
    out: Dict[str, Tuple[float, float]] = {}
    for sym in symbols:
        j = _get("/api/v3/ticker/bookTicker", {"symbol": sym})
        bid = float(j["bidPrice"])
        ask = float(j["askPrice"])
        out[sym] = (bid, ask)
        time.sleep(0.05)
    return out

def fetch_klines(symbol: str, interval: str, limit: int = 500) -> List[list]:
    """
    Endpoint: GET /api/v3/klines :contentReference[oaicite:6]{index=6}
    Returns raw kline arrays:
      [
        [
          openTime, open, high, low, close, volume,
          closeTime, quoteAssetVolume, numberOfTrades,
          takerBuyBaseAssetVolume, takerBuyQuoteAssetVolume, ignore
        ], ...
      ]
    """
    return _get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})