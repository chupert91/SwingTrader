"""Embedded S&P 500 ticker universe used by the discovery scanner.

This is a curated subset of the largest S&P 500 names (~120 tickers,
representing roughly the top ~75% of index market cap). It's enough to
surface real discoveries without bloating the per-scan runtime. The full
~500 list is stable enough that you can expand it whenever you want —
generate the latest list locally via:

    import pandas as pd
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    print(sorted(t.replace(".", "-") for t in tables[0]["Symbol"]))

Then paste the result into SP500_TICKERS below.

Tickers with class-share suffixes (BRK.B, BF.B) are normalized to the
hyphen form (BRK-B, BF-B) which is what yfinance expects.
"""
from __future__ import annotations

SP500_TICKERS: list[str] = [
    # Tech & semis
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN", "TSLA",
    "AVGO", "ORCL", "ADBE", "CRM", "AMD", "INTC", "QCOM", "MU",
    "AMAT", "LRCX", "KLAC", "INTU", "PANW", "ANET", "CSCO", "IBM",
    "NOW", "PLTR", "SNPS", "CDNS", "MRVL", "ON", "FTNT", "ADSK",
    "NFLX", "DIS", "TMUS", "T", "VZ", "CMCSA", "ABNB", "UBER",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP",
    "BLK", "SCHW", "C", "SPGI", "MMC", "ICE", "PGR", "USB", "PNC",
    "TFC", "AON", "MET", "AIG", "TRV", "ALL", "AFL", "PRU", "COF",
    "AMP", "KKR", "BX", "APO",
    # Healthcare / pharma / biotech
    "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "PFE", "AMGN",
    "DHR", "ISRG", "BMY", "GILD", "ELV", "CI", "MDT", "BSX", "SYK",
    "VRTX", "REGN", "CVS", "HUM",
    # Consumer & retail
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "PM", "MO", "NKE",
    "SBUX", "TJX", "TGT", "LOW", "HD", "CMG", "BKNG", "MAR", "HLT",
    "YUM", "ROST", "DG", "DLTR", "KMB", "CL", "MDLZ", "GIS",
    # Industrials & defense
    "CAT", "GE", "BA", "HON", "LMT", "RTX", "NOC", "GD", "DE",
    "UNP", "UPS", "FDX", "CSX", "NSC", "ETN", "EMR", "ITW", "ROP",
    "PH", "MMM", "TT", "LIN", "APD", "ECL", "SHW",
    # Energy & utilities
    "XOM", "CVX", "COP", "EOG", "PSX", "MPC", "VLO", "OXY", "SLB",
    "WMB", "OKE", "NEE", "SO", "DUK", "AEP", "EXC", "SRE", "D",
    # Real estate & materials
    "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "FCX", "NEM", "NUE",
    # Misc / additional well-known
    "ACN", "ADP", "FI", "PAYX", "MSCI", "MCO", "GS", "GE", "GEHC",
    "DELL", "HPQ", "WDC", "JCI", "TDG", "AXON", "VRSK",
]

# De-duplicate while preserving order (some names land in multiple sector buckets).
_seen = set()
SP500_TICKERS = [t for t in SP500_TICKERS if not (t in _seen or _seen.add(t))]
