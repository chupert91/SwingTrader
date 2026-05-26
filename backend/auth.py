"""Shared-password session auth.

If APP_PASSWORD env var is set, all UI/API routes require a valid signed
session cookie issued by POST /login. Cron endpoints stay exempt — they
already authenticate via CRON_SECRET Bearer header.

If APP_PASSWORD is unset the middleware is a no-op, so local dev without
the env var keeps working unchanged.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

COOKIE_NAME = "qsig_session"
SESSION_TTL_SECONDS = 30 * 24 * 3600  # 30 days

# Paths reachable without a session cookie. Cron paths keep their own
# CRON_SECRET Bearer gate inside the handler; listing them here just
# means the session middleware doesn't second-guess them.
_EXEMPT_PATHS = {
    "/login", "/login.html", "/logout",
    "/api/scan", "/api/discover",
    "/api/ai/cron", "/api/stock/cron",
}


def is_enabled() -> bool:
    return bool(os.environ.get("APP_PASSWORD"))


def _signing_key() -> bytes:
    # Sign sessions with SESSION_SECRET if provided, otherwise derive from
    # APP_PASSWORD so a single env var is enough to turn auth on.
    s = os.environ.get("SESSION_SECRET") or os.environ.get("APP_PASSWORD") or ""
    return s.encode("utf-8")


def verify_password(submitted: str) -> bool:
    expected = os.environ.get("APP_PASSWORD") or ""
    if not expected:
        return False
    return hmac.compare_digest(submitted.encode("utf-8"), expected.encode("utf-8"))


def make_session_token() -> str:
    payload = {"exp": int(time.time()) + SESSION_TTL_SECONDS, "n": secrets.token_hex(8)}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).rstrip(b"=")
    sig = hmac.new(_signing_key(), body, hashlib.sha256).hexdigest()
    return f"{body.decode('ascii')}.{sig}"


def verify_session_token(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    body, sig = token.rsplit(".", 1)
    expected_sig = hmac.new(_signing_key(), body.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return False
    try:
        padded = body + "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        return int(payload.get("exp", 0)) > int(time.time())
    except (ValueError, json.JSONDecodeError):
        return False


def is_exempt(path: str) -> bool:
    return path in _EXEMPT_PATHS
