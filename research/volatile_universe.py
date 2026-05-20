"""High-vol thematic universe for the mean-reversion bot research.

Built to match what the user actually trades, not SP500. Bucketed by theme
so the sweep can break out per-theme PF. Every name here is high-beta and
news/theme-driven — exactly the kind of name where 2-sigma touches happen
often and bounces are sharp.

Inclusion rules:
  - All 15 tickers from the user's actual 2025 trades (personal_trade_audit)
  - Adjacent high-IV thematic names in the same buckets
  - Reasonable options liquidity (mostly $1B+ market cap as of 2025-Q1)
  - Will be filtered by ADV>=$50M and 252d-history at sim time anyway

Note: many of these are post-2022 IPOs / SPACs (RKLB, ASTS, OKLO, SMR,
NNE, USAR, RGTI, QUBT, ARQQ etc) — fetch_bars_bulk will drop names that
don't have a full 252-day history. That's fine; we keep them in the list
so the sim picks them up as data backfills.
"""
from __future__ import annotations

# Tagged by theme. Same ticker can appear in multiple themes — we de-dupe
# below.
THEMES: dict[str, list[str]] = {
    # User's actual 2025 trades — always include
    "user_trades": [
        "AMD", "PLTR", "NVDA", "TSLA", "GOOG", "AAPL",
        "IONQ", "MP", "NVTS", "BBAI", "MSTR", "DUOL",
        "HOOD", "SMCI", "APLD",
    ],
    # AI / compute / data center
    "ai_compute": [
        "NVDA", "AMD", "AVGO", "MRVL", "ARM", "SMCI", "APLD", "MU",
        "TSM", "AMAT", "LRCX", "KLAC", "ASML", "ANET", "DELL",
    ],
    # AI software / data / app layer
    "ai_software": [
        "PLTR", "AI", "SOUN", "BBAI", "INNV", "PATH", "SNOW",
        "DDOG", "MDB", "NET", "CRWD", "ZS", "PANW", "GTLB",
    ],
    # Quantum computing
    "quantum": [
        "IONQ", "RGTI", "QUBT", "QBTS", "ARQQ",
    ],
    # Crypto-adjacent (miners + treasury + brokers + ETFs)
    "crypto": [
        "MSTR", "COIN", "HOOD", "MARA", "RIOT", "CIFR", "IREN",
        "HUT", "BITF", "CLSK", "WULF", "BTBT", "BITB", "IBIT",
    ],
    # Critical minerals / rare earth / nuclear / power
    "critical_power": [
        "MP", "USAR", "LAC", "UEC", "CCJ", "LEU", "SMR",
        "OKLO", "NNE", "VST", "CEG", "BWXT", "TLN",
    ],
    # EV / autonomous / mobility
    "ev_mobility": [
        "TSLA", "RIVN", "LCID", "NIO", "XPEV", "ACHR", "JOBY",
    ],
    # Fintech / consumer fintech
    "fintech": [
        "HOOD", "COIN", "AFRM", "SOFI", "UPST", "PYPL", "SQ",
    ],
    # Software / consumer apps (high-IV mid-caps)
    "consumer_apps": [
        "DUOL", "RBLX", "SHOP", "ABNB", "DASH", "UBER", "LYFT", "PINS",
    ],
    # Biotech (high-vol — but liquidity varies; ADV gate will filter)
    "biotech": [
        "MRNA", "BNTX", "NVAX", "BIIB", "ALNY", "LLY", "NVO",
    ],
    # Space / defense (new-era high-vol)
    "space_defense": [
        "RKLB", "ASTS", "BKSY", "ACHR", "JOBY", "LMT", "RTX", "NOC",
    ],
    # Tech megacap (anchor / context)
    "tech_mega": [
        "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NVDA",
        "TSLA", "ORCL", "NFLX", "AVGO", "AMD",
    ],
}


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
    """First theme (priority: user_trades > quantum > crypto > ...) a ticker
    belongs to. Used only for breakout reports."""
    priority = ["user_trades", "quantum", "crypto", "critical_power",
                "ai_compute", "ai_software", "ev_mobility", "fintech",
                "biotech", "space_defense", "consumer_apps", "tech_mega"]
    for theme in priority:
        if tk in THEMES.get(theme, ()):
            return theme
    return "other"


if __name__ == "__main__":
    u = universe()
    print(f"Volatile universe: {len(u)} unique tickers")
    by_theme: dict[str, list[str]] = {}
    for tk in u:
        by_theme.setdefault(theme_of(tk), []).append(tk)
    for theme, tks in by_theme.items():
        print(f"  {theme:18s} ({len(tks):2d}): {' '.join(tks)}")
