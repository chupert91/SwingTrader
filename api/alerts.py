"""Vercel serverless function: GET /api/alerts

Returns the most recent fired alert signals. Wraps the same KV history
the chart sidebar displays.
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


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            from backend import kv
            qs = parse_qs(urlparse(self.path).query)
            try:
                limit = max(1, min(200, int((qs.get("limit") or ["25"])[0])))
            except ValueError:
                limit = 25
            self._json(200, {"alerts": kv.history(limit=limit)})
        except Exception as exc:
            self._json(500, {"detail": str(exc), "trace": traceback.format_exc()})

    def _json(self, status: int, payload: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))
