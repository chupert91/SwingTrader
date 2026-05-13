"""Vercel serverless function: CRUD for alert rules.

GET    /api/rules           -> {"rules": [...]}
POST   /api/rules           -> create or update one rule by id; body = AlertRule dict
DELETE /api/rules?id=<uuid> -> remove the rule with this id
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            from backend import kv
            self._json(200, {"rules": kv.list_rules()})
        except Exception as exc:
            self._json(500, {"ok": False, "error": str(exc),
                             "trace": traceback.format_exc()})

    def do_POST(self):
        try:
            from backend import kv
            from backend.alert_engine import AlertRule

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8")) if body else {}
            rule = AlertRule.from_dict(payload)

            existing = kv.list_rules()
            updated = [r for r in existing if r.get("id") != rule.id]
            updated.append(rule.to_dict())
            kv.save_rules(updated)
            self._json(200, {"ok": True, "rule": rule.to_dict()})
        except Exception as exc:
            self._json(400, {"ok": False, "error": str(exc),
                             "trace": traceback.format_exc()})

    def do_DELETE(self):
        try:
            from backend import kv

            qs = parse_qs(urlparse(self.path).query)
            rule_id = (qs.get("id") or [""])[0]
            if not rule_id:
                self._json(400, {"ok": False, "error": "missing id"})
                return
            existing = kv.list_rules()
            updated = [r for r in existing if r.get("id") != rule_id]
            kv.save_rules(updated)
            self._json(200, {"ok": True, "removed": rule_id})
        except Exception as exc:
            self._json(500, {"ok": False, "error": str(exc),
                             "trace": traceback.format_exc()})

    def _json(self, status: int, payload: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))
