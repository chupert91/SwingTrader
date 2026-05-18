# Oversold Mean-Reversion Options Playbook

Validated from this research arc: 183 S&P names, 5 years of yfinance data.

## The trade

**Entry signal**
- Stock crosses to **z <= -2.0 on the 252-day log-regression channel** — a *fresh* cross (it was above -2 yesterday), not a name that's been sitting there.
- Only take **z between -2.0 and -2.5**. Skip anything deeper than -2.5.
- Long only (oversold -> call). Do not trade the overbought/put side systematically.

**Instrument**
- OTM call, ~5% OTM (strike choice driven by liquidity — degree barely affects outcome).
- 45-60 DTE. Not short-dated, not ATM.

**Exit**
- Take profit at **+30% on the option**.
- Median time to hit ~ **5 trading days** from the touch. If it hasn't worked in ~10 trading days, the edge is gone — close it.
- Expect to be **down ~8-11% on the option first** before it works. That drawdown is normal and is the reason for the 2-month expiry — do not panic out into it.

## Quality tiers (how many S&P names hit oversold the same day)

| Same-day oversold count | Hit +30% within 10 td | Action |
|---|---|---|
| 1-2 (lone name) | 29% | Smallest size or skip — usually real bad news |
| 3-7 (small) | 32% | Normal size |
| **8-19 (moderate cluster)** | **44%** | **Biggest size — best setup** |
| 20+ (panic day) | 34% | Tradeable, but more drawdown — scale in |

## What to avoid (tested, does not work)

- **Most-extreme deviations** (3 sigma, 4 sigma, parabolic): worse, not better — 14% vs 35% hit rate.
- **Overbought puts** (the AMD-style trade): ~22% best case, does not improve on the log channel. Opportunistic only, not a system.
- **Short-dated or ATM options** on these: theta kills them before the reversion (capitulation setup goes from +42% on OTM-60d to dead on ATM-14d).
- **The slope>30%/yr trend filter** for this specific trade: it selects the weakest regime for a 2-month OTM option.

## How to verify it yourself

Every number above comes from a script you can re-run (183 S&P names, 5 years of yfinance data):

| Claim | Script | What it outputs |
|---|---|---|
| Extremity hurts; puts are weak | `python research/fast_exit_extremes.py` | hit-rate table by \|z\| bucket, both directions |
| Log channel does not rescue puts | `python research/log_vs_linear_extremes.py` | linear vs log side-by-side |
| Breadth inverted-U | `python research/cluster_breadth.py` | hit-rate by same-day breadth |
| Instrument inversion | `python research/otm_reversion.py` | OTM-2mo PnL by regime |
| Today's candidates | `python research/oversold_scanner.py` | live ranked list w/ z, freshness, reversion %, RV, liquidity |

Each writes a table to stdout and a file to `research/out/`. The numbers here are pulled directly from those outputs — run them and you get the same tables.

## Caveats (so you can weigh the numbers)

- Option PnL is modeled with Black-Scholes and a simplified IV assumption (no volatility skew). Real put results are likely worse than modeled; calls are roughly fair. Validate against your actual fills.
- 5-year window excludes the 2020 crash.
- No commissions/slippage modeled — a ~4% edge erodes if your round-trip cost is high.

Treat the percentages as relative guidance (which setups beat which), not precise expected returns. The ranking is robust; the exact magnitudes are not.
