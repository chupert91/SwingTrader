# Puts Playbook — Why the Bot Doesn't Trade Puts

Documenting the full research arc that established the put leg as **permanently shelved** in the autonomous bot. Five independent backtests, three signal formulations, one clear diagnosis. Calls-only is the bot — see `oversold_playbook.md` for the trade that ships.

## TL;DR

- User's real 2025 puts: **PF 3.80, 70% win rate, 20 trades, +$3,813 net.** Real, measurable edge.
- Every automated signal we tested on the same universe **fails the PF 1.5 ship gate**. Best PF anywhere: 0.98. Most cells PF 0.5-0.85.
- The edge is exit-timing and contextual entry awareness (catalyst, sector tone, intraday read, gut feel on "pause vs continuation"). Daily-bar technical features cannot replicate it.
- **Recommendation: keep trading puts manually. Bot stays calls-only.**

## The real-world numbers

From `research/personal_trade_audit.py` parsing `reference/DownloadTxnHistory.csv`:

```
20 paired option trades  (May - Dec 2025)
  win rate     70.0%  (14W / 6L)
  profit factor 3.80
  expectancy   +$191 / trade
  total P/L    +$3,813
  median hold  2 trading days  (vs 6d for the calls book)
  median DTE   68 days
  median exit  +9.1% on option premium
```

Compare to the calls book over the same period: PF 1.67, 79% WR, 12d median hold. **Per-trade puts edge was *better* than calls edge in the user's real trading.**

## The asymmetry the data revealed

Headline finding from `puts_extreme_band_sweep.txt`:

```
CALL leg [-4.0, -3.5]:  PF 12.59  (gold zone — forced-seller cap, snap-back)
PUT  leg [+3.5, +4.0]:  PF  0.50  (melt-up — no forced-buyer cap, grind-higher)
~25x ratio
```

Equity markets break down then snap back; melt-ups grind higher with no equivalent statistical cap. Mean-reversion is a *crash* phenomenon. There is no symmetric put-side edge in the daily-bar feature space.

## What we tested and why each failed

### 1. Symmetric +2σ first-touch mirror
- **File**: `research/volatile_universe_puts_sweep.py`
- **Signal**: z in [+2.0, +2.5], first-touch crossing, ADV >= $50M, tier-ranked
- **Best cell**: R6-mirror exits — PF 0.82, CAGR -5.0%, 67 trades
- **Why it failed**: stocks at +2σ keep ripping more often than not (4.5% WR on z >= +3.5)

### 2. Extreme bands (6 bands × 5 exit configs)
- **File**: `research/puts_extreme_band_sweep.py`
- **Bands tested**: [+2,+2.5] | [+2.5,+3] | [+3,+3.5] | [+3.5,+4] | open>=+3.5 | open>=+4
- **Exits tested**: R6, FAST20 (TP20/t5/$200), FAST30, REVERT+1, FAIL-FAST
- **Best PF**: 0.98 at [+2.5,+3.0] + REVERT+1
- **Best CAGR**: +2.1% at open>=+4 + REVERT+1 (PF 0.85)
- **Why it failed**: deeper bands have less data and still don't mean-revert. The symmetric "deep is gold" pattern doesn't hold on the upside.

### 3. Parabolic-exhaustion forward test
- **Files**: `personal_puts_pattern_analysis.py` → `puts_threshold_sensitivity.py` → `puts_parabolic_exhaustion_sweep.py`
- **Threshold sweep on user's 20 trades** found `ret_10d >= 15% AND dist_sma20 >= 15%` gives **91% WR on the sample** (10/14 wins kept, 5/6 losers cut).
- **Forward test on 5y volatile universe**: PF max 0.80 (filter A + R6-MIR, CAGR +7.2%, n=91). All 9 cells failed gate.
- **Why it failed**: the parabolic state itself isn't predictive — most parabolic stocks keep ripping. The user picks WHICH parabolic to short by context the model can't see.

### 4. Post-entry path analysis (diagnostic, not a sim)
- **File**: `research/personal_puts_post_entry_paths.py`
- Revealed the actual pattern: **user catches a brief 1-3 day pause inside an ongoing uptrend.**
- 11/14 winners' underlying bounces +7% median in the 5 days AFTER user's exit.
- 5/14 winners exited *exactly at the underlying's trough*. Elite-level micro-top read.
- Winners' day-before-entry was already -2.4%; losers' was flat. **User enters AFTER pullback has started**, not at the peak.

### 5. Breather-scalp with green-bar exit
- **File**: `research/puts_breather_scalp_sweep.py`
- **Signal**: parabolic + pullback in progress (red bar OR off 3d-high)
- **Exit**: first green underlying bar OR TP20 OR t3/t5 OR $200 cap
- **3 entry × 3 exit = 9 cells**. Best PF 0.61. All failed.
- **What we learned**: the green-bar exit DOES help (+16pts of CAGR vs no green-bar exit) but can't fix a bad entry signal. 63% of parabolic stocks' red bars are 1-day shake-outs before resuming the rip.

## The diagnosis

Three independent backtests on different signal formulations, all failing PF 1.5. The bottleneck isn't calibration — it's information.

**What the bot has**: daily OHLCV bars, computed features (returns, RSI, SMA distance, realized vol, z-score, volume).

**What the user has that the bot doesn't**:
- **Catalyst awareness**: *why* the stock ran (real news, hype, options gamma, earnings reaction). Same parabolic shape but different reasons predict different outcomes.
- **Sector / macro tone**: whether broader sentiment is also topiness-y. Lone parabolics in a healthy tape continue; parabolics in a frothy market reverse.
- **Intraday entry timing**: 5 of 14 winners had 0-day holds, suggesting same-session entry/exit on intraday signal (rejection wick, opening gap fail, late-day reversal).
- **Pattern recognition on "I just see it"**: the kind of read that survives interrogation as "I just knew."

## The user's exit-timing edge (separately documented)

Even if the entry can't be automated, the exit pattern is concrete:

```
Winners' underlying path (median, normalized to entry=1.0):
  entry-3:  -3.3%   (already dropping)
  entry-1:  -2.4%   (pullback in progress)
  entry+0:   0.0%   (entry day)
  entry+1:  -1.5%
  entry+3:  -2.5%   (trough usually here)
  entry+5:  +2.1%   (already bouncing)
  entry+10: +4.4%   (uptrend resumed)

Post-exit (5d after user closes a winner):
  median bounce: +7.0%
  bounced UP: 11/14 (79%)
```

Translation: **"Enter on a red bar inside a parabolic; exit when it shows the first sign of bouncing."** The model can simulate it with `green_bar_exit=True` and the green-bar exit DOES help reduce losses. But without a working entry signal, the exit alone isn't enough.

## Recommendations

### For the user

1. **Keep trading puts manually.** Real PF 3.80 is a real edge — don't second-guess it because the bot can't replicate it.
2. **Don't enable a put leg in the bot.** Three falsifications across five sweeps. The asymmetry is structural.
3. **Track manual puts alongside bot calls.** A future UI section that pulls manual put trades from brokerage history + combines with bot calls in one PnL view would give you the full picture without automating either book onto the other side.

### For future re-evaluation (if conditions change)

Revisit the put leg only if at least one of these becomes available:

- **Catalyst-aware data feed**: earnings calendar, sentiment scores, news flow, sector momentum, VIX state. Test parabolic-exhaustion filter restricted to "high-VIX OR sector-overbought" regimes.
- **Intraday signal triggers**: yfinance 1m bars exist but only ~7d back. A different data source (Polygon, IBKR) with longer intraday history would let us test the same-session entry/exit pattern.
- **New years of trades**: if the user accumulates 50+ manual puts in 2026, re-run the per-trade audit. More samples might surface a feature that's currently hidden by noise.

## How to reproduce

All scripts in `research/`:

```
personal_trade_audit.py            parses brokerage CSV, pairs BTO/STC
personal_puts_pattern_analysis.py  per-put feature analysis
personal_puts_post_entry_paths.py  underlying paths around entry
puts_threshold_sensitivity.py      threshold sweep on 20 real trades

volatile_universe_puts_sweep.py    symmetric mirror test
puts_extreme_band_sweep.py         6 bands x 5 exits
puts_parabolic_exhaustion_sweep.py forward test of thresholds
puts_breather_scalp_sweep.py       pullback+green-bar exit
```

Output text + plots land in `research/out/`. The full chain produces 80+ pages of evidence; this playbook is the executive summary.

## Final verdict

Calls leg is the bot. Puts are yours. The asymmetry between crash-side and melt-up-side mean-reversion is a real and well-documented feature of equity markets — every published "short the bubble" paper has the same caveats, and most retail short books die for the same reason your bot's put leg would.

You found this not by speculation but by running five backtests, parsing your own trade history, mapping the price paths, and watching three independent signal formulations fail with PF < 1. That's a useful negative result — it tells you where to spend research time next (on the call leg, not on the put leg).

---

*Status: shelved 2026-05-19. See `memory/put_leg_falsified.md` for the project-memory pointer.*
