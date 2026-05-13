"""Send alert emails via the Resend API.

Required env:
    RESEND_API_KEY    Resend API key (starts with re_)
    ALERT_FROM        From address (must be a verified Resend domain or onboarding@resend.dev)

If RESEND_API_KEY is missing this falls back to the existing SMTP path
(backend/email_alerts.py) so local dev keeps working with Gmail.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

from backend.alert_engine import RuleSignal

logger = logging.getLogger(__name__)

RESEND_URL = "https://api.resend.com/emails"


def send_signal(signal: RuleSignal, to: str) -> bool:
    """Send an email for a triggered alert. Returns True on success."""
    if not to:
        logger.info("No notify_email on rule; skipping send for %s", signal.ticker)
        return False

    subject, body = _format(signal)

    api_key = os.environ.get("RESEND_API_KEY")
    if api_key:
        return _send_resend(api_key, to, subject, body)
    # Fallback to SMTP for local dev
    return _send_smtp_fallback(to, subject, body, signal)


def _format(signal: RuleSignal) -> tuple[str, str]:
    arrow = "▲" if signal.direction == "long" else "▼"
    side_word = "LONG" if signal.direction == "long" else "SHORT"

    # Translate option pcts -> underlying pcts for the human
    stop_under = signal.exit_stop_pct / max(signal.leverage, 0.001)
    target_under = signal.exit_target_pct / max(signal.leverage, 0.001)

    subject = (f"{arrow} {signal.ticker} {side_word} @ {signal.sd_position:+.2f}σ "
               f"(trend {signal.trend_pct:+.0f}%/yr)")

    lines = [
        f"Rule:          {signal.rule_name}",
        f"Ticker:        {signal.ticker}",
        f"Direction:     {side_word}",
        f"Bar date:      {signal.bar_date}",
        f"Price:         ${signal.price:.2f}",
        f"SD position:   {signal.sd_position:+.2f}σ",
        f"Trend (252d):  {signal.trend_pct:+.1f}% annualized",
        "",
        "Suggested exits (advisory):",
        f"  Target: +{signal.exit_target_pct:.0f}% option "
        f"(~+{target_under:.1f}% underlying @ {signal.leverage:g}x)",
        f"  Stop:   -{signal.exit_stop_pct:.0f}% option "
        f"(~-{stop_under:.1f}% underlying @ {signal.leverage:g}x)",
    ]
    if signal.confirmations:
        lines.append("")
        lines.append("Indicator notes (not gating):")
        lines += [f"  - {c}" for c in signal.confirmations]
    return subject, "\n".join(lines)


def _send_resend(api_key: str, to: str, subject: str, body: str) -> bool:
    from_addr = os.environ.get("ALERT_FROM", "onboarding@resend.dev")
    payload = {
        "from": from_addr,
        "to": [addr.strip() for addr in to.split(",") if addr.strip()],
        "subject": subject,
        "text": body,
    }
    req = urllib.request.Request(
        RESEND_URL,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
    )
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        logger.info("Sent Resend email to %s: %s", to, subject)
        return True
    except Exception as exc:
        logger.exception("Resend send failed: %s", exc)
        return False


def _send_smtp_fallback(to: str, subject: str, body: str, signal: RuleSignal) -> bool:
    """If Resend isn't configured, fall back to the old SMTP path so local
    dev with Gmail credentials still works. ALERT_TO env override applies."""
    try:
        import smtplib
        from email.message import EmailMessage
    except ImportError:
        return False

    required = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "ALERT_FROM")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        logger.warning("Resend not configured and SMTP also missing: %s", ", ".join(missing))
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["ALERT_FROM"]
    msg["To"] = to
    msg.set_content(body)
    try:
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ["SMTP_PORT"]), timeout=15) as s:
            s.ehlo()
            s.starttls()
            s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            s.send_message(msg)
        logger.info("Sent SMTP email to %s: %s", to, subject)
        return True
    except Exception as exc:
        logger.exception("SMTP send failed: %s", exc)
        return False


__all__ = ["send_signal"]
