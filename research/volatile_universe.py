"""Re-export shim: the canonical universe lives in backend/volatile_universe.

The volatile thematic universe is used by BOTH production (backend/
ai_strategy.scan_candidates and backend/discover) AND the research
sweeps, so the source of truth is in backend/. This module just
re-exports so research scripts can `from research.volatile_universe
import universe, theme_of` without changing.
"""
from backend.volatile_universe import THEMES, universe, theme_of, VOLATILE_UNIVERSE

__all__ = ["THEMES", "universe", "theme_of", "VOLATILE_UNIVERSE"]


if __name__ == "__main__":
    u = universe()
    print(f"Volatile universe: {len(u)} unique tickers")
    by_theme: dict[str, list[str]] = {}
    for tk in u:
        by_theme.setdefault(theme_of(tk), []).append(tk)
    for theme, tks in by_theme.items():
        print(f"  {theme:18s} ({len(tks):2d}): {' '.join(tks)}")
