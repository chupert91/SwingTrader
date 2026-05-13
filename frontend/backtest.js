// Backtest page: reads the user's Signal config (set on the chart page) plus
// its own Risk Frame config, merges them into a BacktestConfigPayload, and
// renders /api/backtest and /api/sweep results.

const SIGNAL_KEY = "swingtrader.signal";
const SIGNAL_DEFAULTS = {
  sigma_threshold: 1.0,
  require_trend_alignment: true,
  require_stoch_extreme: true,
  stoch_oversold: 35.0,
  stoch_overbought: 65.0,
};

const RISK_KEY = "swingtrader.backtest";
const RISK_VERSION = 1;
const RISK_DEFAULTS = {
  leverage: 5.0,
  stop_loss_pct: 10.0,
  profit_target_pct: 20.0,
  allocation_pct: 25.0,
  time_stop_bars: 10,
  starting_capital: 10000,
  trail_activation_pct: null,
  trail_distance_pct: null,
  tp_sigma: null,
  sl_sigma: null,
  min_confidence: 0,
  period: "5y",
};

function loadSignal() {
  try {
    const raw = localStorage.getItem(SIGNAL_KEY);
    if (!raw) return { ...SIGNAL_DEFAULTS };
    const parsed = JSON.parse(raw);
    return { ...SIGNAL_DEFAULTS, ...parsed };
  } catch { return { ...SIGNAL_DEFAULTS }; }
}

function loadRisk() {
  try {
    const raw = localStorage.getItem(RISK_KEY);
    if (!raw) return { ...RISK_DEFAULTS };
    const parsed = JSON.parse(raw);
    if (parsed._v !== RISK_VERSION) return { ...RISK_DEFAULTS };
    return { ...RISK_DEFAULTS, ...parsed };
  } catch { return { ...RISK_DEFAULTS }; }
}

function saveRisk(r) {
  localStorage.setItem(RISK_KEY, JSON.stringify({ ...r, _v: RISK_VERSION }));
}

let signal = loadSignal();
let risk = loadRisk();
let ticker = "AAPL";

const echoEl = document.getElementById("bt-signal-echo");
const tickerInput = document.getElementById("bt-ticker-input");
const tickerForm = document.getElementById("bt-ticker-form");
const runBtn = document.getElementById("bt-run-btn");
const sweepBtn = document.getElementById("bt-sweep-btn");
const sweepMetricSel = document.getElementById("bt-sweep-metric");
const resultsPanel = document.getElementById("bt-results-panel");
const resultsEl = document.getElementById("bt-results");
const sweepPanel = document.getElementById("bt-sweep-panel");
const sweepInfoEl = document.getElementById("bt-sweep-info");
const sweepResultsEl = document.getElementById("bt-sweep-results");

function renderSignalEcho() {
  const trendStr = signal.require_trend_alignment ? "ON" : "OFF";
  const stochStr = signal.require_stoch_extreme
    ? `ON (OS ${signal.stoch_oversold} / OB ${signal.stoch_overbought})`
    : "OFF";
  echoEl.innerHTML = `
    <span class="bt-echo-item"><span class="lbl">Min σ</span> <span class="val">${signal.sigma_threshold}</span></span>
    <span class="bt-echo-item"><span class="lbl">Trend aligned</span> <span class="val">${trendStr}</span></span>
    <span class="bt-echo-item"><span class="lbl">Stoch extreme</span> <span class="val">${stochStr}</span></span>
  `;
}

function initRiskForm() {
  for (const input of document.querySelectorAll("[data-bt]")) {
    const key = input.dataset.bt;
    const val = risk[key];
    const isSelect = input.tagName === "SELECT";
    if (input.type === "checkbox") input.checked = !!val;
    else input.value = val == null ? "" : val;
    input.addEventListener("change", () => {
      if (input.type === "checkbox") risk[key] = input.checked;
      else if (isSelect) risk[key] = input.value;
      else if (input.value === "") risk[key] = null;
      else risk[key] = Number(input.value);
      saveRisk(risk);
    });
  }
}

function buildPayload() {
  const thr = Math.abs(Number(signal.sigma_threshold) || 1.0);
  return {
    long_enabled: true,
    short_enabled: true,
    long_entry_sigma: -thr,
    short_entry_sigma: thr,
    min_confidence: risk.min_confidence ?? 0,
    leverage: risk.leverage,
    stop_loss_pct: risk.stop_loss_pct,
    profit_target_pct: risk.profit_target_pct,
    trail_activation_pct: risk.trail_activation_pct,
    trail_distance_pct: risk.trail_distance_pct,
    tp_sigma: risk.tp_sigma,
    sl_sigma: risk.sl_sigma,
    time_stop_bars: risk.time_stop_bars,
    starting_capital: risk.starting_capital,
    allocation_pct: risk.allocation_pct,
    require_trend_alignment: !!signal.require_trend_alignment,
    min_trend_pct: 0.0,
    trend_direction: "any",
    require_stoch_extreme: !!signal.require_stoch_extreme,
    stoch_oversold: signal.stoch_oversold,
    stoch_overbought: signal.stoch_overbought,
    period: risk.period || "5y",
  };
}

async function runBacktest() {
  if (!ticker) return;
  runBtn.classList.add("running");
  runBtn.disabled = true;
  resultsPanel.hidden = false;
  resultsEl.innerHTML = `<div class="bt-stats"><div class="lbl">Running ${ticker}…</div></div>`;
  try {
    const resp = await fetch(`/api/backtest/${encodeURIComponent(ticker)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderStats(data);
  } catch (err) {
    resultsEl.innerHTML = `<div class="bt-error">Error: ${err.message}</div>`;
  } finally {
    runBtn.classList.remove("running");
    runBtn.disabled = false;
  }
}

async function runSweep() {
  if (!ticker) return;
  sweepBtn.classList.add("running");
  sweepBtn.disabled = true;
  sweepPanel.hidden = false;
  sweepInfoEl.textContent = `Sweeping ${ticker}... (this can take 10-20s)`;
  sweepResultsEl.innerHTML = "";
  try {
    const resp = await fetch(`/api/sweep/${encodeURIComponent(ticker)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...buildPayload(), metric: sweepMetricSel.value }),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderSweepResults(data);
  } catch (err) {
    sweepInfoEl.textContent = "";
    sweepResultsEl.innerHTML = `<div class="bt-error">Error: ${err.message}</div>`;
  } finally {
    sweepBtn.classList.remove("running");
    sweepBtn.disabled = false;
  }
}

function renderStats(data) {
  const s = data.stats;
  const sign = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "");
  const fmtPct = (v) => v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
  const reasons = s.exit_reason_counts || {};
  const reasonLine = Object.keys(reasons).length
    ? Object.entries(reasons).map(([k, v]) => `${k}:${v}`).join(" · ")
    : "—";
  resultsEl.innerHTML = `
    <div class="bt-stats">
      <div class="lbl">Return (leveraged)</div>
      <div class="val ${sign(s.total_return_pct)}">${fmtPct(s.total_return_pct)}</div>
      <div class="lbl">B&amp;H (no lev)</div>
      <div class="val ${sign(s.buy_and_hold_pct)}">${fmtPct(s.buy_and_hold_pct)}</div>
      <hr>
      <div class="lbl">Trades</div>
      <div class="val">${s.trade_count} (${s.win_count}W / ${s.loss_count}L)</div>
      <div class="lbl">Win rate</div>
      <div class="val">${s.win_rate_pct.toFixed(1)}%</div>
      <div class="lbl">Avg win</div>
      <div class="val pos">${fmtPct(s.avg_win_pct)}</div>
      <div class="lbl">Avg loss</div>
      <div class="val neg">${fmtPct(s.avg_loss_pct)}</div>
      <div class="lbl">Profit factor</div>
      <div class="val">${isFinite(s.profit_factor) ? s.profit_factor.toFixed(2) : "∞"}</div>
      <hr>
      <div class="lbl">Max drawdown</div>
      <div class="val neg">${fmtPct(s.max_drawdown_pct)}</div>
      <div class="lbl">Sharpe</div>
      <div class="val">${s.sharpe.toFixed(2)}</div>
      <hr>
      <div class="lbl">Exit reasons</div>
      <div class="val" style="font-size:10px;text-align:right">${reasonLine}</div>
    </div>
  `;
}

function renderSweepResults(data) {
  const results = data.results || [];
  sweepInfoEl.textContent = `${data.filtered_count}/${data.total_evaluated} configs passed (min 5 trades), ranked by ${data.metric}`;
  if (results.length === 0) {
    sweepResultsEl.innerHTML = `<div class="bt-error">No configs met the min-trades filter</div>`;
    return;
  }
  sweepResultsEl.innerHTML = "";
  const fmtPct = (v) => v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    const p = r.params;
    const s = r.stats;
    const row = document.createElement("div");
    row.className = "sweep-row";
    row.innerHTML = `
      <div class="params"><span class="rank">#${i + 1}</span>
        σ=${p.long_entry_sigma} · conf=${p.min_confidence} · SL=${p.stop_loss_pct}% · TP=${p.profit_target_pct}% · t=${p.time_stop_bars}b
      </div>
      <div class="stats">
        <span class="${s.total_return_pct >= 0 ? "stat-good" : "stat-bad"}">${fmtPct(s.total_return_pct)}</span>
        <span>${s.trade_count}t · ${s.win_rate_pct.toFixed(0)}% wr</span>
        <span>Sh ${s.sharpe.toFixed(2)}</span>
        <span>PF ${isFinite(s.profit_factor) ? s.profit_factor.toFixed(2) : "∞"}</span>
        <span class="stat-bad">DD ${fmtPct(s.max_drawdown_pct)}</span>
      </div>
    `;
    sweepResultsEl.appendChild(row);
  }
}

tickerForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const v = tickerInput.value.trim().toUpperCase();
  if (v) {
    ticker = v;
    tickerInput.value = v;
  }
});
runBtn.addEventListener("click", runBacktest);
sweepBtn.addEventListener("click", runSweep);

// Re-read signal from localStorage whenever the user comes back to this tab
// (so changes made on the chart page propagate without a manual reload).
window.addEventListener("focus", () => {
  signal = loadSignal();
  renderSignalEcho();
});

// Boot
ticker = tickerInput.value.trim().toUpperCase() || "AAPL";
renderSignalEcho();
initRiskForm();
