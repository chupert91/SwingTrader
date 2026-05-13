"""Vercel serverless function: run the daily alert scan.

Invoked by Vercel Cron (GET request with Bearer CRON_SECRET) or manually.
Loads enabled rules from KV, fetches daily bars for each unique ticker once,
checks every rule against each ticker, sends emails for newly-fired signals,
records them in history.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# Ensure the project root is importable when Vercel runs us
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


def run_scan() -> dict:
    """Thin wrapper — delegates to backend.scan.run_scan()."""
    from backend.scan import run_scan as _impl
    return _impl()


def _check_cron_auth(headers: dict) -> bool:
    """Verify the Vercel Cron Bearer token if CRON_SECRET is set.
    If not set, allow (useful for local testing). Vercel docs:
    https://vercel.com/docs/cron-jobs/manage-cron-jobs#securing-cron-jobs"""
    expected = os.environ.get("CRON_SECRET")
    if not expected:
        return True
    auth = headers.get("authorization") or headers.get("Authorization", "")
    return auth == f"Bearer {expected}"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def _handle(self):
        headers = {k.lower(): v for k, v in self.headers.items()}
        if not _check_cron_auth(headers):
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":false,"error":"unauthorized"}')
            return
        try:
            result = run_scan()
            status = 200
        except Exception as exc:
            logger.exception("scan failed: %s", exc)
            result = {"ok": False, "error": str(exc), "trace": traceback.format_exc()}
            status = 500
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result, default=str).encode("utf-8"))
