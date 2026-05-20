# SwingTrader Glossary

Every acronym, formula, and project-specific term in one place. Companion to `oversold_playbook.md` and `puts_playbook.md`. When a metric appears in a research output (`research/out/*.txt`), this is where it's defined.

---

## 1. Options terms

| Term | Meaning |
|---|---|
| **DTE** | Days To Expiration — calendar days from today to the option's expiry date. Bot default: 80-120. |
| **OTM** | Out of The Money — for a call, strike > spot; for a put, strike < spot. The option has no intrinsic value at this moment. Bot uses 5% OTM. |
| **ATM** | At The Money — strike ≈ spot. |
| **ITM** | In The Money — for a call, strike < spot. The option has intrinsic value (spot − strike). |
| **TP** | Take Profit — a resting sell-limit order at a target gain. Bot default: None (sigma-revert handles exits). |
| **BTO** | Bought To Open — opening a long option position. |
| **STC** | Sold To Close — closing a long option position. |
| **Premium** | The dollar cost of the contract per share (×100 for total). A "$5 premium" on a contract costs $500 to open. Bot premium cap: $2000/contract. |
| **Strike** | The price at which the option exercises. K in formulas. |
| **Multiplier** | 100. One option contract represents 100 shares of underlying. |
| **Bid/Ask spread** | Difference between best buy and best sell quotes. Bot models 1.5% half-spread as fill cost. |

## 2. Greeks (option sensitivities)

| Greek | Means | Used for |
|---|---|---|
| **Delta** | dV/dS — option price change per $1 move in underlying. OTM call delta ≈ 0.3. | Predicts directional move impact. |
| **Gamma** | d²V/dS² — rate of change of delta. Highest near ATM. | Why OTM options can pop fast on big moves. |
| **Theta** | dV/dt — option value decay per day. Cost of time. | Why the time stop matters. |
| **Vega** | dV/dσ — option price change per 1% change in IV. | Why IV crush kills options after volatility spikes. |

Greeks are referenced in `research/theta_overlay.py` and the IV crush model. The bot does not compute them directly — `bs_call` returns the price; greeks fall out of that.

## 3. Volatility terms

| Term | Formula / definition |
|---|---|
| **RV** (Realized Volatility) | Annualized standard deviation of daily log-returns over a lookback window. `RV = std(log(close[t] / close[t-1])) * sqrt(252)`. Bot uses 20-day lookback. |
| **IV** (Implied Volatility) | The volatility number that makes the Black-Scholes formula reproduce the option's market price. Not directly observable on yfinance. |
| **IV model** | `entry_iv = clamp(RV × iv_elevation, 0.15, 1.20)`. Default `iv_elevation = 1.00` (was 1.25 in early model — relaxed after sim diverged from reality). |
| **IV crush** | IV decay from elevated entry value back to RV floor. Linear interpolation over `crush_td` trading days (default 30): `iv(d) = entry_iv - (entry_iv - iv_floor) * min(d / crush_td, 1)`. |
| **σ** (sigma) | Greek letter; can mean (a) volatility input to BS, (b) the standard-deviation distance from the regression channel in the z-score signal. Context distinguishes. |
| **bps** | Basis points. 1 bp = 0.01%. "10 bps slip" = 0.10% slippage. |

## 4. Statistics — sim output metrics

These four numbers are the conversation. They appear on every row of every sweep.

### PF — Profit Factor
```
PF = sum(gross winning $) / |sum(gross losing $)|
```
- PF > 1.0 → per-trade edge exists
- PF >= 1.5 → ship gate
- PF >= 2.0 → strong edge
- Live bot sim: 1.68. User's real 2025: 2.09. Symmetric → live calibrated to reality.

### CAGR — Compound Annual Growth Rate
```
CAGR = (final_equity / starting_equity)^(1 / years) - 1
```
- Live bot sim: +18% (5y volatile basket)
- The "annualized" return — what % the account grows per year on average.

### Sharpe Ratio
```
Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
```
- Annualized risk-adjusted return. Higher = smoother growth per unit return.
- Live bot sim: 0.85. Above 0.5 is decent; above 1.0 is excellent.

### MaxDD — Maximum Drawdown
```
MaxDD = min over t of (equity[t] - peak_equity[<=t]) / peak_equity[<=t]
```
- The worst peak-to-trough decline in the equity curve. Always negative.
- Live bot sim: -19% (down from -32% before the hybrid band ship).

### MAR Ratio
```
MAR = CAGR / |MaxDD|
```
- Return per unit of drawdown pain. Higher = better.
- Live bot sim: 0.95. The $300 stop variant hit 1.20 — best of all variants tested.

### WR — Win Rate
```
WR = (# of trades with PnL > 0) / (total trades) * 100
```
- A 28% WR with PF 2.32 is fine — fewer wins, but each bigger than the losses.
- A 60% WR with PF 0.7 is bad — many small wins, but the few losses dominate.

## 5. The signal — 252-day LOG regression channel

The core indicator.

### Formula

For each bar t with rolling 252-bar window:

```
y[i]    = log(close[i])                    for i in window
slope, intercept = OLS regression of y on x (x = 0..251)
fit[i]  = slope * i + intercept
residuals[i] = y[i] - fit[i]
sigma   = std(residuals)                   sample std (ddof=1)

z[t]    = (y[t] - fit[end]) / sigma
```

- z = 0 → price is exactly on the regression line
- z = -2 → price is 2 standard deviations BELOW the line (oversold)
- z = +2 → price is 2 standard deviations ABOVE the line (overbought)
- Bot computes z over a rolling 252-day window, so the channel re-fits every day.

### LOG vs LINEAR

Using log of close (instead of raw close) makes percentage moves additive — a $1 move from $10 to $11 (10%) gets the same residual weight as a $10 move from $100 to $110 (10%). LINEAR was tested and rejected — see `log_vs_linear_extremes` (file deleted, scripted reproducible).

### Slope (annualized)
```
slope_ann_pct = (exp(slope * 252) - 1) * 100
```
- Tells you the channel's trend direction. The bot tracks it but doesn't gate on it (slope filter was tested and DOES help some configs, hurt others — left as info-only).

## 6. The signal — entry rules (HYBRID band)

The bot fires on EITHER condition:

```
PRIMARY band:   z in [-2.5, -2.0]  AND  z just crossed -2.0 from above
DEEP cross:     z <= -3.5           AND  z just crossed -3.5 from above

EXCLUDED (dead zone):   z in (-3.5, -2.5)
```

Each candidate carries `source: "primary" | "deep"`. Plus:

- **First-touch**: only fires when `z[t-1] > band_hi AND z[t] <= band_hi`. Names already in the band don't re-fire on every bar.
- **Bars in zone**: count of bars since the touch. UI shows `FRESH` (≤1), `recent` (2-5), `STALE` (>5).
- **ADV gate**: average dollar volume over 20 bars must be ≥ $50M. Rejects illiquid names.

## 7. Tier system — same-day breadth ranking

After candidates are found, they're ranked by **breadth** = how many names crossed into z ≤ -2.0 on the same day.

| Tier | Breadth (# names) | Meaning | Action |
|---|---|---|---|
| **PANIC** | ≥ 20 | Market-wide capitulation | Take any candidate; rare |
| **PRIME** | 8-19 | Sector or broad selloff | Best risk-reward zone |
| **OK** | 3-7 | Mild correlated weakness | Take if no better signals |
| **WEAK** | 1-2 | Lone-name drop, likely bad news | Skip per playbook |
| **?** | 0 | No same-day data | Skip |

The bot accepts {PRIME, OK, PANIC} and skips {WEAK, ?}. Within tier, ranked by (1) freshness (first-touch first), (2) bars in zone (fewer first), (3) realized vol (higher first).

## 8. Black-Scholes call pricing

Used by every sim to mark options to market.

```
d1 = (log(S/K) + (r + 0.5 * σ²) * T) / (σ * sqrt(T))
d2 = d1 - σ * sqrt(T)
call = S * N(d1) - K * exp(-r * T) * N(d2)
```

Where:
- S = spot price
- K = strike
- T = time to expiry in years (DTE / 365)
- r = risk-free rate (default 4%)
- σ = implied vol (from the IV model)
- N(x) = standard normal CDF (`0.5 * (1 + erf(x / sqrt(2)))`)

### Put pricing via put-call parity
```
put = call - S + K * exp(-r * T)
```
Used in `research/volatile_universe_puts_sweep.py` for the falsified put-leg backtests.

### Intrinsic at expiry
```
call payoff = max(S - K, 0)
put  payoff = max(K - S, 0)
```

## 9. Technical indicators (used in research, not all in live signal)

### RSI(14) — Wilder's
```
gains   = max(close[t] - close[t-1], 0)
losses  = max(close[t-1] - close[t], 0)
avg_gain = Wilder-smoothed mean of gains over 14 bars
avg_loss = Wilder-smoothed mean of losses
RS  = avg_gain / avg_loss
RSI = 100 - 100 / (1 + RS)
```
- RSI > 70 = overbought; RSI < 30 = oversold
- Used in `puts_threshold_sensitivity.py` and `personal_puts_pattern_analysis.py`

### Stochastic %K (14)
```
%K = (close - low_14) / (high_14 - low_14) * 100
```
- %K = 100 → at the 14-bar high; %K = 0 → at the low
- Bot has an optional Stoch RSI overlay (default off) — "prefer" sorts oversold names first within tier; "require" drops non-oversold

### SMA(N)
```
SMA(N)[t] = mean(close[t-N+1 : t+1])
```
- Bot uses SMA(20) for the consec-bars indicator's "below SMA" gate (chart overlay only, not used as trade signal)
- `dist_sma20 = (close - SMA(20)) / SMA(20) * 100` — distance %, used in puts pattern analysis

### Consec bars
Signed indicator: positive run length for up-bars, negative for down-bars. Resets to 0 on flat. `backend/indicators_lib/consec_bars.py`. Chart-only overlay; falsified as entry gate.

## 10. Project-specific terms

| Term | Meaning |
|---|---|
| **R6** | The validated exit config. No TP, sigma-revert at z=0, $200/$300 disaster cap, 45d time stop, IV 1.00/30. Named for being the 6th row in the original real-trade-calibrated sweep. |
| **Hybrid band** | Primary `[-2.5, -2.0]` OR deep `z <= -3.5`. Skips the dead zone `(-3.5, -2.5)`. |
| **Dead zone** | `z in (-3.5, -2.5)`. Tested PF 1.04 / CAGR -18% standalone. Knife-falling-through entries that stop short of true capitulation. Explicitly excluded. |
| **Gold zone** | `z <= -3.5` (especially `[-4.0, -3.5]`). PF 12.59 standalone. Where forced sellers cap out and bounce begins. |
| **Source** | Per-candidate tag: `"primary"` (band entry) or `"deep"` (z<=-3.5 cross). Shown as UI badge. |
| **Sigma-target exit** | Close when underlying z reverts to ≥ target (default 0). Replaces TP-based exits in R6. |
| **First-touch** | The bar where z just crossed into the band on this side. Bot prefers FRESH (touched today) over STALE (touched 5+ bars ago). |
| **Sweet spot** | Historical name for the primary `[-2.5, -2.0]` band. From early research before the hybrid was discovered. |
| **Breadth** | Same-day count of crossings into z ≤ -2.0 across the universe. The "is broad market capitulating?" signal. Drives tier rank. |
| **ADV** | Average Daily Dollar Volume = mean(close × volume) over 20 bars. Bot requires ≥ $50M for liquidity. |
| **Universe** | The 102-name volatile thematic basket in `backend/volatile_universe.py`. NOT SP500. |
| **KV** | Vercel Upstash key-value store. Holds bot state, settings, trade history. Shared between local and production — local writes mutate the real bot. |
| **Cron** | The scheduled job (`/api/cron/ai`) that runs the bot's bookkeeping + entry evaluation. Fires hourly during market hours. |
| **Marketable limit** | A buy limit price set at `ask × 1.01` (1% above ask). Behaves like a market order but caps the price you'd pay. Bot uses this for entries to avoid market-order slippage. |

## 11. Sim-specific terms

| Term | Meaning |
|---|---|
| **R-config** | Numbered exit configurations in `research/oversold_call_exit_sweep.py`. R0..R6 are real-trade-calibrated variants. R6 is what ships. |
| **Patient configs (P0-P5)** | Hold-to-peak exit variants tested in the original sweep. Mostly inferior to R6. |
| **FAST20 / FAST15** | Fast-exit configs (TP 20% or 15%, time stop 3-5d). Tested for puts, didn't beat R6. |
| **Realism layer** | Cost assumptions: 1.5% half-spread, $0.50/contract commission (negligible). Tighter than the old 3% half-spread + 1% buffer that was found to over-pessimize the model. |
| **Ship gate** | PF ≥ 1.5 AND CAGR > 0 AND n ≥ 10 on the 5y volatile basket. Any new variant must clear all three to ship to production. |
| **Pinned exits / signal** | When sweeping one variable, the others are held at the live default. Otherwise rows aren't directly comparable. |
| **Capital-constrained sim** | The sim tracks $10k starting equity, deploys per the bot's max-concurrent + max-per-day rules. Returns reflect what the bot would actually have done, not idealized per-trade PnL. |
| **n** | Trade count over the 5y window. Sim outputs that report `n=15` mean 3 trades/year. Most sims show n=50 to n=250. |

## 12. Brokerage / data terms

| Term | Meaning |
|---|---|
| **yfinance** | Yahoo Finance Python library. Daily OHLCV bars. Free; rate-limited; occasional missing data. Primary data source. |
| **Alpaca** | Broker API. Used for paper-trading the bot (live order placement + position tracking). Free IEX feed. Webull was rejected (TOS/API issues). |
| **Paper keys** | Alpaca paper-trading credentials. The bot uses these for data access even in research. No real money at risk. |
| **OHLCV** | Open, High, Low, Close, Volume — the daily bar fields. |
| **IEX** | Investors Exchange. Alpaca's free real-time feed (15-min delayed for non-paying users; covers ~3% of US equity volume but enough for liquid names). |

---

## Quick reference — formulas in one block

```
z-score             z = (log(S) - fit_line) / sigma_of_residuals
realized vol        RV = std(log(S[t]/S[t-1])) * sqrt(252)
IV model            entry_iv = clamp(RV * 1.00, 0.15, 1.20)
IV crush            iv(d) = entry_iv - (entry_iv - RV) * min(d / 30, 1)
BS call             call = S*N(d1) - K*exp(-r*T)*N(d2)
                    d1 = (log(S/K) + (r + 0.5*σ²)*T) / (σ*sqrt(T))
                    d2 = d1 - σ*sqrt(T)
BS put              put = call - S + K*exp(-r*T)   (put-call parity)

profit factor       PF = sum_wins$ / |sum_losses$|
CAGR                CAGR = (final/start)^(1/years) - 1
Sharpe              Sharpe = mean(daily_ret) / std(daily_ret) * sqrt(252)
maxDD               minimum peak-to-trough decline %
MAR                 MAR = CAGR / |maxDD|

RSI                 RSI = 100 - 100/(1 + avg_gain/avg_loss)
Stoch %K            %K = (close - low_N) / (high_N - low_N) * 100
SMA(N)              mean(close over last N bars)
dist_sma20          (close - SMA(20)) / SMA(20) * 100
```

## See also

- `research/oversold_playbook.md` — the trade that ships
- `research/puts_playbook.md` — why no puts
- `.claude/skills/trade-method/commands/trade-method.md` — load full strategy context per session
