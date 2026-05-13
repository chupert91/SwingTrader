"""Vercel serverless function: POST /api/alerts/scan

Browser-triggered scan from the chart page's "scan now" button. Same
implementation as the cron-driven /api/scan, but without the bearer-auth
check (this path is meant to be hit by the UI, not the cron).
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self._handle()

    def do_GET(self):
        # Allow GET for convenience (e.g. curl testing).
        self._handle()

    def _handle(self):
        try:
            from backend.scan import run_scan
            result = run_scan()
            status = 200
        except Exception as exc:
            logger.exception("scan failed: %s", exc)
            result = {"ok": False, "error": str(exc), "trace": traceback.format_exc()}
            status = 500
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(result, default=str).encode("utf-8"))
