"""Background scanner: every 10 min during US market hours, scan the watchlist."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time
from zoneinfo import ZoneInfo

from backend import alerts, email_alerts, state, watchlist
from backend.data import fetch_bars

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
SCAN_INTERVAL_SECONDS = 10 * 60
EMAIL_MIN_CONFIDENCE = 2


def is_market_hours(now: datetime | None = None) -> bool:
    now = now or datetime.now(ET)
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def scan_once() -> list[alerts.Signal]:
    """One scan pass: check each watchlist ticker for a new signal, fire if so.

    Returns the list of Signals that were freshly fired (i.e. emailed + recorded).
    """
    tickers = watchlist.get()
    fired: list[alerts.Signal] = []
    for ticker in tickers:
        try:
            df = fetch_bars(ticker, period="14mo")
            if df.empty:
                continue
            df = alerts.prepare_df(df)
            sig = alerts.detect(ticker, df)
            if sig is None:
                continue
            if not state.should_fire(sig):
                continue
            state.record(sig)
            if sig.confidence >= EMAIL_MIN_CONFIDENCE:
                email_alerts.send(sig)
            fired.append(sig)
            logger.info(
                "Fired %s %s @ %s (conf %d/3)",
                sig.ticker, sig.direction, sig.bar_date, sig.confidence,
            )
        except Exception as exc:
            logger.exception("Error scanning %s: %s", ticker, exc)
    return fired


async def run_forever() -> None:
    logger.info("Scanner started; interval %ds", SCAN_INTERVAL_SECONDS)
    while True:
        try:
            if is_market_hours():
                fired = scan_once()
                if fired:
                    logger.info("Scan complete: %d signals fired", len(fired))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Scanner loop error: %s", exc)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)
