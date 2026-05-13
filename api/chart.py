"""Vercel serverless function: GET /api/chart/{ticker}

Frontend hits /api/chart/AAPL. A vercel.json rewrite maps that to
/api/chart?ticker=AAPL so the brackets-in-filename pattern is avoided.
"""
from __future__ import annotations

import json
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
    def do_GET(self):
        try:
            ticker = _ticker_from_query(self.path)
            if not ticker:
                self._json(400, {"detail": "missing ticker"})
                return
            qs = parse_qs(urlparse(self.path).query)
            period = (qs.get("period") or ["14mo"])[0]

            from backend.main import _prepare, _serialize
            from backend.channels import REGRESSION_WINDOW
            from backend import ichimoku
            from backend.indicators import macd, stoch_rsi

            full = _prepare(ticker, period)
            full["stoch_rsi_k"], full["stoch_rsi_d"] = stoch_rsi(full["close"])
            full["macd_line"], full["macd_signal"], full["macd_hist"] = macd(full["close"])
            ichi = ichimoku.compute(full)

            crop_start = max(0, len(full) - REGRESSION_WINDOW)
            df = full.iloc[crop_start:].reset_index(drop=True)
            ichi_cropped = ichimoku.IchimokuComponents(
                tenkan=ichi.tenkan.iloc[crop_start:].reset_index(drop=True),
                kijun=ichi.kijun.iloc[crop_start:].reset_index(drop=True),
                senkou_a=ichi.senkou_a.iloc[crop_start:].reset_index(drop=True),
                senkou_b=ichi.senkou_b.iloc[crop_start:].reset_index(drop=True),
                chikou=ichi.chikou.iloc[crop_start:].reset_index(drop=True),
            )
            self._json(200, _serialize(df, ticker, ichi_cropped))
        except Exception as exc:
            self._json(500, {"detail": str(exc), "trace": traceback.format_exc()})

    def _json(self, status: int, payload: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))
