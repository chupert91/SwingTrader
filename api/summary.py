"""Vercel serverless function: GET /api/summary/{ticker}

Rewritten by vercel.json to /api/summary?ticker=... — see api/chart.py.
"""
from __future__ import annotations

import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _ticker_from_query(path: str) -> str:
    qs = parse_qs(urlparse(path).query)
    return (qs.get("ticker") or [""])[0].upper()


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            ticker = _ticker_from_query(self.path)
            if not ticker:
                self._json(400, {"detail": "missing ticker"})
                return
            qs = parse_qs(urlparse(self.path).query)
            period = (qs.get("period") or ["14mo"])[0]

            from backend.main import _prepare, _annual_slope_pct, _f

            df = _prepare(ticker, period)
            latest = df.iloc[-1]
            self._json(200, {
                "ticker": ticker,
                "current_price": _f(latest["close"]),
                "sd_position": _f(latest.get("sd_position", np.nan)),
                "r_squared": _f(latest.get("r_squared", np.nan)),
                "slope_annual_pct": _f(_annual_slope_pct(df)),
            })
        except Exception as exc:
            self._json(500, {"detail": str(exc), "trace": traceback.format_exc()})

    def _json(self, status: int, payload: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))
