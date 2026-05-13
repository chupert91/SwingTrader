"""Vercel serverless function: POST /api/sweep/{ticker}

Rewritten by vercel.json to /api/sweep?ticker=... — see api/chart.py.
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _ticker_from_query(path: str) -> str:
    qs = parse_qs(urlparse(path).query)
    return (qs.get("ticker") or [""])[0].upper()


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            ticker = _ticker_from_query(self.path)
            if not ticker:
                self._json(400, {"detail": "missing ticker"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8")) if body else {}

            from backend import backtest
            from backend.data import fetch_bars

            period = payload.pop("period", "5y")
            metric = payload.pop("metric", "sharpe")
            top_n = int(payload.pop("top_n", 10))
            min_trades = int(payload.pop("min_trades", 5))

            df = fetch_bars(ticker, period=period)
            if df.empty:
                self._json(404, {"detail": f"No data for ticker '{ticker}'"})
                return

            valid_fields = {f.name for f in dataclasses.fields(backtest.BacktestConfig)}
            cfg_kwargs = {k: v for k, v in payload.items() if k in valid_fields}
            base = backtest.BacktestConfig(**cfg_kwargs)

            raw_results = backtest.sweep(df, base)
            filtered = [
                r for r in raw_results
                if r["stats"]["trade_count"] >= min_trades
                and r["stats"].get(metric) is not None
                and math.isfinite(r["stats"].get(metric, float("nan")))
            ]
            filtered.sort(key=lambda r: r["stats"][metric], reverse=True)
            self._json(200, {
                "ticker": ticker,
                "metric": metric,
                "total_evaluated": len(raw_results),
                "filtered_count": len(filtered),
                "grid": backtest.DEFAULT_SWEEP_GRID,
                "results": filtered[:top_n],
            })
        except Exception as exc:
            self._json(500, {"detail": str(exc), "trace": traceback.format_exc()})

    def _json(self, status: int, payload: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))
