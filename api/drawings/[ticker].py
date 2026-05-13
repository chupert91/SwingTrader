"""Vercel serverless function: GET/PUT /api/drawings/{ticker}

Per-ticker drawings blob ({hlines, trendlines, fibs, trades}) used by the
cross-device sync. Persisted in KV.
"""
from __future__ import annotations

import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _extract_ticker(path: str) -> str:
    p = urlparse(path).path.strip("/")
    parts = p.split("/")
    return parts[-1].upper() if parts else ""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            from backend import kv
            ticker = _extract_ticker(self.path)
            if not ticker:
                self._json(400, {"detail": "missing ticker"})
                return
            self._json(200, kv.get_ticker_drawings(ticker))
        except Exception as exc:
            self._json(500, {"detail": str(exc), "trace": traceback.format_exc()})

    def do_PUT(self):
        try:
            from backend import kv
            ticker = _extract_ticker(self.path)
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
