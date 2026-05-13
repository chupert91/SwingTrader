"""Send alert emails via SMTP. Config from environment variables.

Required env vars (all string; load from .env via os.environ):
    SMTP_HOST       e.g. smtp.gmail.com
    SMTP_PORT       e.g. 587
    SMTP_USER       SMTP username (often same as ALERT_FROM)
    SMTP_PASSWORD   Gmail app password or equivalent
    ALERT_FROM      from address
    ALERT_TO        comma-separated list of recipients

If any of these are missing, send() logs a warning and returns False.
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

from backend.alerts import Signal

logger = logging.getLogger(__name__)

_REQUIRED = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "ALERT_FROM", "ALERT_TO")


def _config() -> dict[str, str] | None:
    missing = [k for k in _REQUIRED if not os.environ.get(k)]
    if missing:
        logger.warning("Email not configured; missing env vars: %s", ", ".join(missing))
        return None
    return {k: os.environ[k] for k in _REQUIRED}


def send(signal: Signal) -> bool:
    cfg = _config()
    if cfg is None:
        return False

    stars = "★" * signal.confidence + "☆" * (3 - signal.confidence)
    direction_word = "LONG" if signal.direction == "long" else "SHORT"
    subject = f"[{stars}] {signal.ticker} {direction_word} signal — {signal.sd_position:+.2f}σ"

    body_lines = [
        f"Ticker:     {signal.ticker}",
        f"Direction:  {direction_word}",
        f"Bar date:   {signal.bar_date}",
        f"Price:      ${signal.price:.2f}",
        f"SD pos:     {signal.sd_position:+.2f}σ",
        f"Confidence: {signal.confidence}/3 ({stars})",
        "",
        "Confirmations:",
    ]
    body_lines += [f"  • {c}" for c in signal.confirmations] if signal.confirmations else ["  (none)"]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["ALERT_FROM"]
    msg["To"] = cfg["ALERT_TO"]
    msg.set_content("\n".join(body_lines))

    try:
        with smtplib.SMTP(cfg["SMTP_HOST"], int(cfg["SMTP_PORT"]), timeout=15) as s:
            s.ehlo()
            s.starttls()
            s.login(cfg["SMTP_USER"], cfg["SMTP_PASSWORD"])
            s.send_message(msg)
        logger.info("Sent alert email for %s %s", signal.ticker, signal.direction)
        return True
    except Exception as exc:
        logger.exception("Failed to send alert email: %s", exc)
        return False
