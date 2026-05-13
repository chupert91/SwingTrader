"""Vercel serverless function: GET/PUT /api/watchlist

Server-side mirror of the user's watchlist (synced from any device's
chart page) — used by the scheduled scan to know which tickers to check.
"""
from __future__ import annotations

import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            from backend import watchlist
            self._json(200, {"tickers": watchlist.get()})
        except Exception as exc:
            self._json(500, {"detail": str(exc), "trace": traceback.format_exc()})

    def do_PUT(self):
        try:
            from backend import watchlist

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8")) if body else {}
            tickers = payload.get("tickers", [])
            if not isinstance(tickers, list):
                self._json(400, {"detail": "tickers must be a list"})
                return
            saved = watchlist.set_(tickers)
            self._json(200, {"tickers": saved})
        except Exception as exc:
            self._json(500, {"detail": str(exc), "trace": traceback.format_exc()})

    def _json(self, status: int, payload: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))
