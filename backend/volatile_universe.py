"""High-vol thematic universe for the AI options trader's signal scan.

Replaces the SP500 universe (backend/sp500_tickers.py) — see research/
out/volatile_universe_sweep.txt. Same signal logic on this universe vs.
SP500 produced PF 2.32 vs PF 1.41 (research) and matches the user's real
2025 trading PF of 2.09 (research/out/personal_trade_audit.txt).

Inclusion rules (mirrors research/volatile_universe.py):
  - All 15 user-traded 2025 tickers
  - Adjacent high-IV thematic names: AI compute, AI software, quantum,
    crypto-adjacent, critical-minerals/nuclear, EV/mobility, fintech,
    consumer apps, biotech, space/defense, tech megacap.
  - Liquidity (>=$50M 20-day ADV) is enforced at scan time, so this list
    can be a superset.

Names are bucketed by theme so the discover/scan output can carry a
`theme` tag for the UI.
"""
from __future__ import annotations

THEMES: dict[str, list[str]] = {
    # User's actual 2025 trades — always present
    "user_trades": [
        "AMD", "PLTR", "NVDA", "TSLA", "GOOG", "AAPL",
        "IONQ", "MP", "NVTS", "BBAI", "MSTR", "DUOL",
        "HOOD", "SMCI", "APLD",
    ],
    "ai_compute": [
        "NVDA", "AMD", "AVGO", "MRVL", "ARM", "SMCI", "APLD", "MU",
        "TSM", "AMAT", "LRCX", "KLAC", "ASML", "ANET", "DELL",
    ],
    "ai_software": [
        "PLTR", "AI", "SOUN", "BBAI", "INNV", "PATH", "SNOW",
        "DDOG", "MDB", "NET", "CRWD", "ZS", "PANW", "GTLB",
    ],
    "quantum": [
        "IONQ", "RGTI", "QUBT", "QBTS", "ARQQ",
    ],
    "crypto": [
        "MSTR", "COIN", "HOOD", "MARA", "RIOT", "CIFR", "IREN",
        "HUT", "BITF", "CLSK", "WULF", "BTBT", "BITB", "IBIT",
    ],
    "critical_power": [
        "MP", "USAR", "LAC", "UEC", "CCJ", "LEU", "SMR",
        "OKLO", "NNE", "VST", "CEG", "BWXT", "TLN",
    ],
    "ev_mobility": [
        "TSLA", "RIVN", "LCID", "NIO", "XPEV", "ACHR", "JOBY",
    ],
    "fintech": [
        "HOOD", "COIN", "AFRM", "SOFI", "UPST", "PYPL",
    ],
    "consumer_apps": [
        "DUOL", "RBLX", "SHOP", "ABNB", "DASH", "UBER", "LYFT", "PINS",
    ],
    "biotech": [
        "MRNA", "BNTX", "NVAX", "BIIB", "ALNY", "LLY", "NVO",
    ],
    "space_defense": [
        "RKLB", "ASTS", "BKSY", "ACHR", "JOBY", "LMT", "RTX", "NOC",
    ],
    "tech_mega": [
        "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NVDA",
        "TSLA", "ORCL", "NFLX", "AVGO", "AMD",
    ],
}

_THEME_PRIORITY = ["user_trades", "quantum", "crypto", "critical_power",
                   "ai_compute", "ai_software", "ev_mobility", "fintech",
                   "biotech", "space_defense", "consumer_apps", "tech_mega"]


def universe() -> list[str]:
    """De-duplicated flat ticker list across all themes."""
    seen: set[str] = set()
    out: list[str] = []
    for tks in THEMES.values():
        for tk in tks:
            if tk not in seen:
                seen.add(tk)
                out.append(tk)
    return out


def theme_of(tk: str) -> str:
    """First (highest-priority) theme a ticker belongs to. user_trades wins
    if the user actually trades it; otherwise the most distinctive theme."""
    for theme in _THEME_PRIORITY:
        if tk in THEMES.get(theme, ()):
            return theme
    return "other"


# Backwards-compat constant. Some callers expect a constant list rather
# than calling universe(); both stay in sync because universe() is
# evaluated once at import.
VOLATILE_UNIVERSE: list[str] = universe()


__all__ = ["THEMES", "universe", "theme_of", "VOLATILE_UNIVERSE"]
