"""Vercel serverless function: GET/PUT /api/drawings/{ticker}

Rewritten by vercel.json to /api/drawings?ticker=... — see api/chart.py.
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
            from backend import kv
            ticker = _ticker_from_query(self.path)
            if not ticker:
                self._json(400, {"detail": "missing ticker"})
                return
            self._json(200, kv.get_ticker_drawings(ticker))
        except Exception as exc:
            self._json(500, {"detail": str(exc), "trace": traceback.format_exc()})

    def do_PUT(self):
        try:
            from backend import kv
            ticker = _ticker_from_query(self.path)
            if not ticker:
                self._json(400, {"detail": "missing ticker"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8")) if body else {}
            blob = {
                "hlines": payload.get("hlines", []) or [],
                "trendlines": payload.get("trendlines", []) or [],
                "fibs": payload.get("fibs", []) or [],
                "trades": payload.get("trades", []) or [],
            }
            kv.set_ticker_drawings(ticker, blob)
            self._json(200, {"ok": True})
        except Exception as exc:
            self._json(500, {"detail": str(exc), "trace": traceback.format_exc()})

    def _json(self, status: int, payload: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))
