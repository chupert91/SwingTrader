const COLORS = {
  up: "#26a69a",
  down: "#ef5350",
  reg: "#ffffff",
  band: "#7d8590",
  grid: "#1f242c",
  text: "#8b949e",
  tenkan: "#3498db",
  kijun: "#c0392b",
  senkouA: "rgba(38, 166, 154, 0.55)",
  senkouB: "rgba(239, 83, 80, 0.55)",
  chikou: "#9b59b6",
};

// Inline SVG icons used across JS-rendered templates. Stroke-only so they
// inherit the surrounding text color via currentColor.
const ICON_X = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
const ICON_PLUS = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>`;

const baseOptions = {
  // autoSize: true makes the chart watch its container with a ResizeObserver
  // and re-render when the grid layout changes. Essential for dynamic panes
  // whose containers are created before the grid row is widened to fit them.
  autoSize: true,
  layout: {
    background: { type: "solid", color: "#0e1117" },
    textColor: COLORS.text,
    fontSize: 11,
    attributionLogo: false,
  },
  grid: {
    vertLines: { color: COLORS.grid },
    horzLines: { color: COLORS.grid },
  },
  rightPriceScale: { borderColor: COLORS.grid },
  timeScale: { borderColor: COLORS.grid, rightOffset: 8, barSpacing: 6 },
  crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
};

const indicatorOptions = (showTime) => ({
  ...baseOptions,
  timeScale: { ...baseOptions.timeScale, visible: showTime },
});

const containers = {
  price: document.getElementById("chart-price"),
};
const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");
const form = document.getElementById("ticker-form");
const input = document.getElementById("ticker-input");

const priceChart = LightweightCharts.createChart(containers.price, baseOptions);

// Each entry in `allCharts` participates in time-scale + crosshair sync.
// Dynamic indicator panes get pushed in by _createDynamicPane() at runtime.
const allCharts = [priceChart];

const candleSeries = priceChart.addCandlestickSeries({
  upColor: COLORS.up,
  downColor: COLORS.down,
  borderUpColor: COLORS.up,
  borderDownColor: COLORS.down,
  wickUpColor: COLORS.up,
  wickDownColor: COLORS.down,
});

// Stacked volume: total drawn first (red), buy portion drawn on top (green).
// Result: each bar shows green from 0 → buy_volume, then red from buy_volume → total.
const volumeTotalSeries = priceChart.addHistogramSeries({
  priceFormat: { type: "volume" },
  priceScaleId: "vol",
  color: COLORS.down,
  priceLineVisible: false,
  lastValueVisible: false,
});
const volumeBuySeries = priceChart.addHistogramSeries({
  priceFormat: { type: "volume" },
  priceScaleId: "vol",
  color: COLORS.up,
  priceLineVisible: false,
  lastValueVisible: false,
});
priceChart.priceScale("vol").applyOptions({
  scaleMargins: { top: 0.85, bottom: 0 },
});

function addLine(chart, color, width, style) {
  return chart.addLineSeries({
    color,
    lineWidth: width,
    lineStyle: style ?? LightweightCharts.LineStyle.Solid,
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
  });
}

const overlaySeries = {
  regression_line: addLine(priceChart, COLORS.reg, 2),
  upper_1sd: addLine(priceChart, COLORS.band, 1, LightweightCharts.LineStyle.Dotted),
  lower_1sd: addLine(priceChart, COLORS.band, 1, LightweightCharts.LineStyle.Dotted),
  upper_2sd: addLine(priceChart, COLORS.band, 1, LightweightCharts.LineStyle.Dashed),
  lower_2sd: addLine(priceChart, COLORS.band, 1, LightweightCharts.LineStyle.Dashed),
  upper_3sd: addLine(priceChart, COLORS.band, 1.5),
  lower_3sd: addLine(priceChart, COLORS.band, 1.5),
};

const ichimokuSeries = {
  tenkan: addLine(priceChart, COLORS.tenkan, 1.5),
  kijun: addLine(priceChart, COLORS.kijun, 1.5),
  senkou_a: addLine(priceChart, COLORS.senkouA, 1),
  senkou_b: addLine(priceChart, COLORS.senkouB, 1),
  chikou: addLine(priceChart, COLORS.chikou, 1, LightweightCharts.LineStyle.Dashed),
};

// Sync time scales across all panes.
let syncing = false;
function syncTimeRange(source) {
  source.timeScale().subscribeVisibleLogicalRangeChange((range) => {
    if (syncing || !range) return;
    syncing = true;
    for (const c of allCharts) {
      if (c !== source) c.timeScale().setVisibleLogicalRange(range);
    }
    syncing = false;
  });
}
allCharts.forEach(syncTimeRange);

// Sync crosshair across panes.
function syncCrosshair(source) {
  source.subscribeCrosshairMove((param) => {
    if (!param.time) {
      for (const c of allCharts) if (c !== source) c.clearCrosshairPosition();
      return;
    }
    for (const c of allCharts) {
      if (c !== source) c.setCrosshairPosition(NaN, param.time, c.priceScale("right"));
    }
  });
}
allCharts.forEach(syncCrosshair);

window.addEventListener("resize", () => {
  priceChart.applyOptions({
    width: containers.price.clientWidth,
    height: containers.price.clientHeight,
  });
  for (const pane of _dynamicPanes.values()) {
    pane.chart.applyOptions({
      width: pane.chartEl.clientWidth,
      height: pane.chartEl.clientHeight,
    });
  }
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  loadTicker(input.value.trim().toUpperCase());
});

function setStatus(msg, isError = false) {
  if (!msg) {
    statusEl.classList.remove("visible", "error");
    return;
  }
  statusEl.textContent = msg;
  statusEl.classList.add("visible");
  statusEl.classList.toggle("error", isError);
}

async function loadTicker(ticker) {
  if (!ticker) return;
  setStatus(`Loading ${ticker}...`);
  setActiveTicker(ticker);
  stopQuotePolling();
  try {
    // Fetch chart + drawings in parallel; the drawings step also handles
    // first-time migration from localStorage cache → server.
    const [resp] = await Promise.all([
      fetch(`/api/chart/${encodeURIComponent(ticker)}`),
      fetchAndMergeDrawings(ticker),
    ]);
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderChart(data);
    setStatus(null);
    startQuotePolling();
  } catch (err) {
    setStatus(`Error: ${err.message}`, true);
  }
}

function renderChart(data) {
  // Compute rightmost time on the chart (latest candle, or last Ichimoku projection).
  let right = data.candles?.length ? data.candles[data.candles.length - 1].time : null;
  const senkouA = data.ichimoku?.senkou_a || [];
  if (senkouA.length) {
    const t = senkouA[senkouA.length - 1].time;
    if (!right || String(t) > String(right)) right = t;
  }
  lastChartRightTime = right;

  // Capture every time slot the price chart will render (candles + ichimoku
  // future projection). Dynamic indicator panes pad their time axes with
  // whitespace at these times so their logical indexing matches the price
  // chart's — otherwise the time-scale sync propagates a wider range than
  // the dynamic data covers and the bottom panes look "shifted left".
  const allTimes = new Set();
  (data.candles || []).forEach(c => { if (c.time != null) allTimes.add(c.time); });
  if (data.ichimoku) {
    for (const arr of Object.values(data.ichimoku)) {
      if (Array.isArray(arr)) arr.forEach(p => { if (p?.time != null) allTimes.add(p.time); });
    }
  }
  _priceChartAllTimes = Array.from(allTimes).sort();

  // Snapshot the last bar so the live-quote poller can keep updating it
  // without re-fetching the full chart.
  const candles = data.candles || [];
  _lastCandle = candles.length ? { ...candles[candles.length - 1] } : null;

  renderDrawings(data.ticker);
  candleSeries.setData(data.candles);
  volumeTotalSeries.setData(data.volume_total);
  volumeBuySeries.setData(data.volume_buy);
  for (const [key, series] of Object.entries(overlaySeries)) {
    series.setData(data.overlays[key] || []);
  }
  if (data.ichimoku) {
    for (const [key, series] of Object.entries(ichimokuSeries)) {
      series.setData(data.ichimoku[key] || []);
    }
  }
  renderCustomIndicators(data.custom_indicators || []);
  priceChart.timeScale().fitContent();
  renderSummary(data.ticker, data.summary);
}

function renderSummary(ticker, s) {
  const sdClass = s.sd_position == null ? "" : s.sd_position > 1 ? "neg" : s.sd_position < -1 ? "pos" : "";
  summaryEl.innerHTML = `
    <div><div class="label">${ticker} <span id="live-badge" class="live-badge" hidden></span></div><div class="value" id="summary-price">$${fmt(s.current_price, 2)}</div></div>
    <div><div class="label">SD Position</div><div class="value ${sdClass}">${fmt(s.sd_position, 2)}σ</div></div>
    <div><div class="label">R²</div><div class="value">${fmt(s.r_squared, 3)}</div></div>
    <div><div class="label">Trend (annual)</div><div class="value ${s.slope_annual_pct > 0 ? "pos" : "neg"}">${fmt(s.slope_annual_pct, 1)}%</div></div>
  `;
}

// --- Live quote polling (Alpaca IEX free tier) --------------------------
// Polls /api/quote/{ticker} every 30s while the page is visible and the
// US market is open. Updates the summary price + LIVE indicator without
// disturbing the chart's bar data or any drawings.

const QUOTE_POLL_MS = 30_000;
let _quoteTimer = null;
let _quoteLastTick = 0;  // ms timestamp of last successful poll
let _lastCandle = null;  // { time, open, high, low, close } — kept in sync with the chart's last bar

function _todayET() {
  // YYYY-MM-DD in the New York timezone — matches yfinance's daily bar time format.
  try {
    const parts = new Intl.DateTimeFormat("en-CA", {
      timeZone: "America/New_York",
      year: "numeric", month: "2-digit", day: "2-digit",
    }).formatToParts(new Date());
    const m = Object.fromEntries(parts.map(p => [p.type, p.value]));
    return `${m.year}-${m.month}-${m.day}`;
  } catch { return null; }
}

function _applyLivePriceToCandle(price) {
  if (!_lastCandle || typeof price !== "number") return;
  const today = _todayET();
  if (!today) return;
  // Always keep _lastCandle.time as a string. Lightweight Charts' update()
  // mutates the input's `time` field into its internal BusinessDay object
  // representation, so we pass a shallow copy and keep our canonical string.
  if (_lastCandle.time === today) {
    _lastCandle.high = Math.max(_lastCandle.high, price);
    _lastCandle.low = Math.min(_lastCandle.low, price);
    _lastCandle.close = price;
    try { candleSeries.update({ ..._lastCandle }); } catch {}
  } else if (String(_lastCandle.time) < today) {
    _lastCandle = { time: today, open: price, high: price, low: price, close: price };
    try { candleSeries.update({ ..._lastCandle }); } catch {}
  }
}

function _marketSession() {
  // Returns "pre" | "regular" | "after" | "closed" in ET (auto-handles DST).
  // Holidays not filtered — we'd just waste a poll, no harm done.
  try {
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone: "America/New_York",
      weekday: "short",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).formatToParts(new Date());
    const m = Object.fromEntries(parts.map(p => [p.type, p.value]));
    if (m.weekday === "Sat" || m.weekday === "Sun") return "closed";
    const minutes = parseInt(m.hour, 10) * 60 + parseInt(m.minute, 10);
    if (minutes >= 4 * 60 && minutes < 9 * 60 + 30) return "pre";
    if (minutes >= 9 * 60 + 30 && minutes < 16 * 60) return "regular";
    if (minutes >= 16 * 60 && minutes < 20 * 60) return "after";
    return "closed";
  } catch { return "closed"; }
}

// Extended-hours marker: a dashed amber price line on the candle series
// that tracks the current ext-hours trade. Cleared once regular hours
// start (so the daily candle becomes authoritative again).
const EXT_HOURS_COLOR = "#e69138";
let _extPriceLine = null;

function _clearExtPriceLine() {
  if (_extPriceLine) {
    try { candleSeries.removePriceLine(_extPriceLine); } catch {}
    _extPriceLine = null;
  }
}

function _setExtPriceLine(price, session) {
  if (typeof price !== "number") return;
  _clearExtPriceLine();
  _extPriceLine = candleSeries.createPriceLine({
    price,
    color: EXT_HOURS_COLOR,
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dashed,
    axisLabelVisible: true,
    title: session === "pre" ? "Pre-mkt" : "After-hrs",
  });
}

function _setLiveBadge(state) {
  const el = document.getElementById("live-badge");
  if (!el) return;
  if (state === "off") {
    el.hidden = true;
    el.className = "live-badge";
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.className = `live-badge ${state}`;
  el.textContent = state === "live" ? "● LIVE" : "● DELAYED";
}

async function _pollOnce() {
  if (!activeTicker) return;
  if (document.visibilityState !== "visible") return;
  const session = _marketSession();
  if (session === "closed") {
    _clearExtPriceLine();
    _setLiveBadge("delayed");
    return;
  }
  try {
    const resp = await fetch(`/api/quote/${encodeURIComponent(activeTicker)}`);
    if (!resp.ok) {
      // 503 = no Alpaca creds; 404 = no trade; 502 = upstream — all soft-fail.
      _setLiveBadge("delayed");
      return;
    }
    const data = await resp.json();
    _quoteLastTick = Date.now();
    const priceEl = document.getElementById("summary-price");
    if (priceEl && typeof data.price === "number") {
      priceEl.textContent = `$${data.price.toFixed(2)}`;
    }
    if (session === "regular") {
      _clearExtPriceLine();
      _applyLivePriceToCandle(data.price);
    } else {
      _setExtPriceLine(data.price, session);
    }
    _setLiveBadge("live");
  } catch {
    _setLiveBadge("delayed");
  }
}

function startQuotePolling() {
  stopQuotePolling();
  // Fire one immediately, then on interval. The first call will populate
  // the badge state before the user has to wait 30s.
  _pollOnce();
  _quoteTimer = setInterval(_pollOnce, QUOTE_POLL_MS);
}

function stopQuotePolling() {
  if (_quoteTimer) clearInterval(_quoteTimer);
  _quoteTimer = null;
  _clearExtPriceLine();
  _setLiveBadge("off");
}

// Restart polling when the page becomes visible again (tab switch / phone
// wake). When hidden, stop the interval entirely to save Vercel function
// invocations.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && activeTicker) {
    startQuotePolling();
  } else {
    stopQuotePolling();
  }
});

function fmt(v, digits) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

// --- Signal config -------------------------------------------------------
// Stored under swingtrader.signal and shared with /backtest.html, which
// reads the same key to inherit the user's signal definition.
const SIGNAL_KEY = "swingtrader.signal";
const SIGNAL_VERSION = 1;
const SIGNAL_DEFAULTS = {
  sigma_threshold: 1.0,
  require_trend_alignment: true,
  require_stoch_extreme: true,
  stoch_oversold: 35.0,
  stoch_overbought: 65.0,
  trend_direction: "any",
  min_avg_volume_m: 50.0,
};

function loadSignal() {
  try {
    const raw = localStorage.getItem(SIGNAL_KEY);
    if (!raw) return { ...SIGNAL_DEFAULTS };
    const parsed = JSON.parse(raw);
    if (parsed._v !== SIGNAL_VERSION) return { ...SIGNAL_DEFAULTS };
    return { ...SIGNAL_DEFAULTS, ...parsed };
  } catch { return { ...SIGNAL_DEFAULTS }; }
}

function saveSignal(sig) {
  localStorage.setItem(SIGNAL_KEY, JSON.stringify({ ...sig, _v: SIGNAL_VERSION }));
}

let signal = loadSignal();
// Signal config is edited on /settings.html now. chart.js just reads the
// localStorage that page writes — there's no sidebar form to wire up here.

// --- Canonical alert-rule sync ------------------------------------------
// The server-side scanner reads from KV. We push a single rule (id "ui-signal")
// derived from Signal + Watchlist + notify email, debounced so rapid edits
// coalesce into one POST.
const NOTIFY_EMAIL_KEY = "swingtrader.notifyEmail";
const SIGNAL_RULE_ID = "ui-signal";

function loadNotifyEmail() {
  return localStorage.getItem(NOTIFY_EMAIL_KEY) || "";
}
function saveNotifyEmail(v) {
  localStorage.setItem(NOTIFY_EMAIL_KEY, v || "");
}

function buildSignalRule() {
  const thr = Math.abs(Number(signal.sigma_threshold) || 1.0);
  return {
    id: SIGNAL_RULE_ID,
    name: "Signal alerts (auto-synced)",
    tickers: loadWatchlist(),
    side: "both",
    entry_sigma: -thr,
    require_trend: !!signal.require_trend_alignment,
    min_trend_pct: 0.0,
    exit_target_pct: 20.0,
    exit_stop_pct: 10.0,
    leverage: 5.0,
    enabled: true,
    notify_email: loadNotifyEmail(),
    trend_direction: signal.trend_direction || "any",
    require_stoch_extreme: !!signal.require_stoch_extreme,
    stoch_oversold: Number(signal.stoch_oversold) || 35,
    stoch_overbought: Number(signal.stoch_overbought) || 65,
    min_avg_volume_m: Number(signal.min_avg_volume_m) || 0,
  };
}

let _syncTimer = null;
function scheduleSignalRuleSync() {
  if (_syncTimer) clearTimeout(_syncTimer);
  _syncTimer = setTimeout(syncSignalRule, 500);
}

async function syncSignalRule() {
  try {
    await fetch("/api/rules", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildSignalRule()),
    });
  } catch (err) { /* non-fatal */ }
}

// Notify email is now edited on /settings.html; chart.js only reads it
// from localStorage when building the canonical AlertRule.

// Backtest markers state — kept so applyAllMarkers stays uniform with the
// manual-trade marker pipeline. The main page no longer fires a backtest,
// so backtestMarkers stays []. The /backtest page has its own markers UI.
let backtestMarkers = [];

function applyAllMarkers() {
  const manual = (currentDrawings?.trades || []).map(manualTradeToMarker);
  const combined = [...manual, ...backtestMarkers];
  combined.sort((a, b) => String(a.time).localeCompare(String(b.time)));
  try { candleSeries.setMarkers(combined); } catch {}
}

// --- Drawing tools -------------------------------------------------------
const DRAWINGS_KEY = "swingtrader.drawings";
const DRAW_COLOR = "#58a6ff";

// Fib levels (with 127.2/161.8 extensions). Color = visual emphasis;
// 50 and 61.8 (golden ratio) get the warmer tones since they're most-watched.
const FIB_LEVELS = [
  { ratio: 0,     color: "#7d8590" },
  { ratio: 0.236, color: "#3498db" },
  { ratio: 0.382, color: "#1abc9c" },
  { ratio: 0.5,   color: "#f1c40f" },
  { ratio: 0.618, color: "#e67e22" },
  { ratio: 0.786, color: "#e74c3c" },
  { ratio: 1.0,   color: "#7d8590" },
  { ratio: 1.272, color: "#c0392b" },
  { ratio: 1.618, color: "#8e44ad" },
];

function loadAllDrawings() {
  try { return JSON.parse(localStorage.getItem(DRAWINGS_KEY) || "{}"); }
  catch { return {}; }
}

function saveAllDrawings(d) {
  localStorage.setItem(DRAWINGS_KEY, JSON.stringify(d));
}

function getDrawings(ticker) {
  const all = loadAllDrawings();
  return all[ticker] || { hlines: [], trendlines: [], fibs: [], trades: [] };
}

function setDrawingsFor(ticker, drawings, opts = {}) {
  const all = loadAllDrawings();
  all[ticker] = drawings;
  saveAllDrawings(all);
  if (!opts.skipServerSync) schedulePushDrawings(ticker);
}

// --- Server-side drawings sync ------------------------------------------
const _drawingPushTimers = new Map();  // ticker -> timeout handle

function schedulePushDrawings(ticker) {
  if (!ticker) return;
  const prev = _drawingPushTimers.get(ticker);
  if (prev) clearTimeout(prev);
  _drawingPushTimers.set(ticker, setTimeout(() => pushDrawings(ticker), 500));
}

async function pushDrawings(ticker) {
  const blob = getDrawings(ticker);
  try {
    await fetch(`/api/drawings/${encodeURIComponent(ticker)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        hlines: blob.hlines || [],
        trendlines: blob.trendlines || [],
        fibs: blob.fibs || [],
        trades: blob.trades || [],
      }),
    });
  } catch (err) { /* non-fatal, localStorage still has the data */ }
}

async function fetchAndMergeDrawings(ticker) {
  let server;
  try {
    const resp = await fetch(`/api/drawings/${encodeURIComponent(ticker)}`);
    if (!resp.ok) return;
    server = await resp.json();
  } catch { return; }
  const local = getDrawings(ticker);
  const serverHas = _drawingsNonEmpty(server);
  const localHas = _drawingsNonEmpty(local);
  if (serverHas) {
    // Server wins. Replace local cache without re-pushing.
    setDrawingsFor(ticker, server, { skipServerSync: true });
  } else if (localHas) {
    // First-time migration: push local to server.
    schedulePushDrawings(ticker);
  }
}

function _drawingsNonEmpty(d) {
  if (!d) return false;
  return (d.hlines?.length || 0) + (d.trendlines?.length || 0)
       + (d.fibs?.length || 0) + (d.trades?.length || 0) > 0;
}

let drawingMode = "cursor";
let trendlinePending = null;          // null or { time, value }
let fibPending = null;                // null or { time, value }
let currentDrawings = { hlines: [], trendlines: [], fibs: [], trades: [] };
let drawnPriceLines = [];              // IPriceLine handles (hlines)
let drawnTrendlineSeries = [];         // ISeriesApi handles (trendlines)
let drawnFibSeries = [];               // array of ISeriesApi[] (one inner array per fib)
let lastChartRightTime = null;         // rightmost time on the chart, for fib endpoint

const drawingHintEl = document.getElementById("drawing-hint");

function setDrawingMode(mode) {
  drawingMode = mode;
  trendlinePending = null;
  fibPending = null;
  for (const btn of document.querySelectorAll(".drawing-toolbar button[data-tool]")) {
    btn.classList.toggle("active", btn.dataset.tool === mode);
  }
  updateDrawingHint();
}

function updateDrawingHint(text) {
  if (text) {
    drawingHintEl.textContent = text;
    drawingHintEl.classList.add("visible");
  } else if (drawingMode === "hline") {
    drawingHintEl.textContent = "Click chart to place horizontal line · Esc to cancel";
    drawingHintEl.classList.add("visible");
  } else if (drawingMode === "trend") {
    drawingHintEl.textContent = trendlinePending
      ? "Click second point · Esc to cancel"
      : "Click first point of trend line · Esc to cancel";
    drawingHintEl.classList.add("visible");
  } else if (drawingMode === "fib") {
    drawingHintEl.textContent = fibPending
      ? "Click opposite swing point (high or low) · Esc to cancel"
      : "Click first swing point (high or low) · Esc to cancel";
    drawingHintEl.classList.add("visible");
  } else if (drawingMode === "entry") {
    drawingHintEl.textContent = "Click chart to mark trade entry · Esc to cancel";
    drawingHintEl.classList.add("visible");
  } else if (drawingMode === "exit") {
    drawingHintEl.textContent = "Click chart to mark trade exit · Esc to cancel";
    drawingHintEl.classList.add("visible");
  } else {
    drawingHintEl.classList.remove("visible");
  }
}

function clearDrawnHandles() {
  for (const handle of drawnPriceLines) {
    try { candleSeries.removePriceLine(handle); } catch {}
  }
  for (const series of drawnTrendlineSeries) {
    try { priceChart.removeSeries(series); } catch {}
  }
  for (const seriesArr of drawnFibSeries) {
    for (const s of seriesArr) {
      try { priceChart.removeSeries(s); } catch {}
    }
  }
  drawnPriceLines = [];
  drawnTrendlineSeries = [];
  drawnFibSeries = [];
  // Trade level lines + connectors belong to the price chart too — wipe on ticker change.
  if (typeof _clearTradeLevelLines === "function") _clearTradeLevelLines();
  if (typeof _clearTradeConnectors === "function") _clearTradeConnectors();
}

function clearDrawingsForCurrentTicker() {
  clearDrawnHandles();
  const preservedTrades = currentDrawings.trades || [];
  currentDrawings = { hlines: [], trendlines: [], fibs: [], trades: preservedTrades };
  if (activeTicker) setDrawingsFor(activeTicker, currentDrawings);
}

function renderDrawings(ticker) {
  clearDrawnHandles();
  const raw = getDrawings(ticker);
  currentDrawings = {
    hlines: raw.hlines || [],
    trendlines: raw.trendlines || [],
    fibs: raw.fibs || [],
    trades: raw.trades || [],
  };
  for (const hl of currentDrawings.hlines) {
    drawnPriceLines.push(candleSeries.createPriceLine({
      price: hl.price,
      color: hl.color || DRAW_COLOR,
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Solid,
      axisLabelVisible: true,
      title: hl.title || "",
    }));
  }
  for (const tl of currentDrawings.trendlines) {
    drawnTrendlineSeries.push(_makeTrendSeries(tl));
  }
  for (const fib of currentDrawings.fibs) {
    drawnFibSeries.push(_makeFibLines(fib));
  }
  applyAllMarkers();
  renderTradesList();
  renderTradeLevelLines();
  renderTradeConnectors();
}

function _makeFibLines(fib) {
  // 0% anchored at first click, 100% at second click, extensions past second.
  // Lines render only from first click forward to the right edge of the chart.
  const p1 = fib.p1 || fib.high;
  const p2 = fib.p2 || fib.low;
  const start = p1.value;
  const end = p2.value;
  const range = end - start;
  const startTime = p1.time;
  const endTime = lastChartRightTime && String(lastChartRightTime) > String(startTime)
    ? lastChartRightTime
    : p2.time;

  const seriesArr = [];
  for (const lvl of FIB_LEVELS) {
    const price = start + range * lvl.ratio;
    const s = priceChart.addLineSeries({
      color: lvl.color,
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: false,
      title: `${(lvl.ratio * 100).toFixed(1)}%`,
    });
    s.setData([
      { time: startTime, value: price },
      { time: endTime, value: price },
    ]);
    seriesArr.push(s);
  }
  return seriesArr;
}

function addFib(p1, p2) {
  const fib = { p1, p2 };
  currentDrawings.fibs.push(fib);
  setDrawingsFor(activeTicker, currentDrawings);
  drawnFibSeries.push(_makeFibLines(fib));
}

function _makeTrendSeries(tl) {
  const series = priceChart.addLineSeries({
    color: tl.color || DRAW_COLOR,
    lineWidth: 1.5,
    lineStyle: LightweightCharts.LineStyle.Solid,
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
  });
  const points = [tl.p1, tl.p2].slice().sort((a, b) =>
    String(a.time).localeCompare(String(b.time))
  );
  series.setData(points);
  return series;
}

function addHLine(price) {
  currentDrawings.hlines.push({ price, color: DRAW_COLOR });
  setDrawingsFor(activeTicker, currentDrawings);
  drawnPriceLines.push(candleSeries.createPriceLine({
    price,
    color: DRAW_COLOR,
    lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Solid,
    axisLabelVisible: true,
  }));
}

function addTrendline(p1, p2) {
  const tl = { p1, p2, color: DRAW_COLOR };
  currentDrawings.trendlines.push(tl);
  setDrawingsFor(activeTicker, currentDrawings);
  drawnTrendlineSeries.push(_makeTrendSeries(tl));
}

document.querySelectorAll("#drawing-toolbar button").forEach(btn => {
  btn.addEventListener("click", () => {
    if (btn.dataset.tool === "clear") {
      clearDrawingsForCurrentTicker();
      closeDrawingRailOnMobile();
      return;
    }
    setDrawingMode(btn.dataset.tool);
    closeDrawingRailOnMobile();
  });
});

// Mobile-only: floating button toggles the drawing rail in/out from the left.
const _drawingRailEl = document.getElementById("drawing-rail");
const _drawingRailToggle = document.getElementById("drawing-rail-toggle");
function _isMobile() { return window.matchMedia("(max-width: 720px)").matches; }
function closeDrawingRailOnMobile() {
  if (!_isMobile() || !_drawingRailEl) return;
  _drawingRailEl.classList.remove("open");
  _drawingRailToggle?.classList.remove("active");
}
_drawingRailToggle?.addEventListener("click", () => {
  if (!_drawingRailEl) return;
  const opened = _drawingRailEl.classList.toggle("open");
  _drawingRailToggle.classList.toggle("active", opened);
});

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  const modal = document.getElementById("trade-modal");
  if (modal && !modal.hidden) {
    closeTradeModal();
    return;
  }
  if (drawingMode !== "cursor") setDrawingMode("cursor");
});

priceChart.subscribeClick(param => {
  if (drawingMode === "cursor") return;
  if (!param.point || !param.time) return;
  const price = candleSeries.coordinateToPrice(param.point.y);
  if (price == null) return;
  if (drawingMode === "hline") {
    addHLine(price);
    setDrawingMode("cursor");
    return;
  }
  if (drawingMode === "trend") {
    if (!trendlinePending) {
      trendlinePending = { time: param.time, value: price };
      updateDrawingHint();
    } else {
      addTrendline(trendlinePending, { time: param.time, value: price });
      trendlinePending = null;
      setDrawingMode("cursor");
    }
    return;
  }
  if (drawingMode === "fib") {
    if (!fibPending) {
      fibPending = { time: param.time, value: price };
      updateDrawingHint();
    } else {
      addFib(fibPending, { time: param.time, value: price });
      fibPending = null;
      setDrawingMode("cursor");
    }
    return;
  }
  if (drawingMode === "entry" || drawingMode === "exit") {
    openTradeModal(drawingMode, param.time, price);
    setDrawingMode("cursor");
  }
});

// --- Display settings ----------------------------------------------------
const DISPLAY_KEY = "swingtrader.display";
const DISPLAY_DEFAULTS = {
  ichimoku: { tenkan: false, kijun: false, cloud: true, chikou: false },
  regression: { line: true, sd1: true, sd2: true, sd3: true },
  volume: { bars: true },
};

function loadDisplay() {
  try {
    const raw = localStorage.getItem(DISPLAY_KEY);
    if (!raw) return structuredClone(DISPLAY_DEFAULTS);
    const parsed = JSON.parse(raw);
    return mergeDeep(structuredClone(DISPLAY_DEFAULTS), parsed);
  } catch { return structuredClone(DISPLAY_DEFAULTS); }
}

function saveDisplay(d) {
  localStorage.setItem(DISPLAY_KEY, JSON.stringify(d));
}

function mergeDeep(base, over) {
  for (const k of Object.keys(over)) {
    if (over[k] && typeof over[k] === "object" && !Array.isArray(over[k])) {
      base[k] = mergeDeep(base[k] || {}, over[k]);
    } else {
      base[k] = over[k];
    }
  }
  return base;
}

let display = loadDisplay();

function applyDisplay() {
  // Ichimoku
  ichimokuSeries.tenkan.applyOptions({ visible: display.ichimoku.tenkan });
  ichimokuSeries.kijun.applyOptions({ visible: display.ichimoku.kijun });
  ichimokuSeries.senkou_a.applyOptions({ visible: display.ichimoku.cloud });
  ichimokuSeries.senkou_b.applyOptions({ visible: display.ichimoku.cloud });
  ichimokuSeries.chikou.applyOptions({ visible: display.ichimoku.chikou });
  // Regression
  overlaySeries.regression_line.applyOptions({ visible: display.regression.line });
  overlaySeries.upper_1sd.applyOptions({ visible: display.regression.sd1 });
  overlaySeries.lower_1sd.applyOptions({ visible: display.regression.sd1 });
  overlaySeries.upper_2sd.applyOptions({ visible: display.regression.sd2 });
  overlaySeries.lower_2sd.applyOptions({ visible: display.regression.sd2 });
  overlaySeries.upper_3sd.applyOptions({ visible: display.regression.sd3 });
  overlaySeries.lower_3sd.applyOptions({ visible: display.regression.sd3 });
  // Volume
  volumeTotalSeries.applyOptions({ visible: display.volume.bars });
  volumeBuySeries.applyOptions({ visible: display.volume.bars });
  updateLegend();
}

function updateLegend() {
  const map = {
    "sw-reg": display.regression.line,
    "sw-1": display.regression.sd1,
    "sw-2": display.regression.sd2,
    "sw-3": display.regression.sd3,
    "sw-tenkan": display.ichimoku.tenkan,
    "sw-kijun": display.ichimoku.kijun,
    "sw-senkou-a": display.ichimoku.cloud,
    "sw-senkou-b": display.ichimoku.cloud,
  };
  for (const [cls, vis] of Object.entries(map)) {
    const swatch = document.querySelector(`.legend i.${cls}`);
    if (swatch) swatch.parentElement.style.display = vis ? "" : "none";
  }
}

function initDisplayControls() {
  for (const cb of document.querySelectorAll("input[data-display]")) {
    const path = cb.dataset.display.split(".");
    cb.checked = path.reduce((o, k) => o[k], display);
    cb.addEventListener("change", () => {
      let target = display;
      for (let i = 0; i < path.length - 1; i++) target = target[path[i]];
      target[path[path.length - 1]] = cb.checked;
      saveDisplay(display);
      applyDisplay();
    });
  }
}

// --- Manual trade marks --------------------------------------------------
// Real-world trades the user marks on the chart. Stored per-ticker in the
// same localStorage blob as drawings (currentDrawings.trades). Exits are
// auto-paired LIFO to the most-recent unpaired entry on the same ticker.

const tradesListEl = document.getElementById("my-trades-list");
const tradesEmptyEl = document.getElementById("my-trades-empty");
const tradeModalEl = document.getElementById("trade-modal");
const tradeModalForm = document.getElementById("trade-modal-form");
const tradeModalTitle = document.getElementById("trade-modal-title");
const tradeModalMeta = document.getElementById("trade-modal-meta");
const tradeModalDirField = document.getElementById("trade-modal-direction-field");
const tradeSizeInput = document.getElementById("trade-size");
const tradeNoteInput = document.getElementById("trade-note");
const tradeStopInput = document.getElementById("trade-stop-price");
const tradeTargetInput = document.getElementById("trade-target-price");
const tradePricesGroup = document.getElementById("trade-modal-prices");

// Defaults derived from the locked 2% underlying stop / 4% underlying target frame.
const ENTRY_STOP_FRAC = 0.02;
const ENTRY_TARGET_FRAC = 0.04;

function _defaultStopTarget(direction, price) {
  if (direction === "short") {
    return { stop: price * (1 + ENTRY_STOP_FRAC), target: price * (1 - ENTRY_TARGET_FRAC) };
  }
  return { stop: price * (1 - ENTRY_STOP_FRAC), target: price * (1 + ENTRY_TARGET_FRAC) };
}

let pendingTrade = null;  // { kind: "entry"|"exit", time, price }

function newTradeId() {
  return Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8);
}

function persistTrades() {
  if (activeTicker) setDrawingsFor(activeTicker, currentDrawings);
}

function addEntry({ direction, size, note, time, price, stopPrice, targetPrice }) {
  const defaults = _defaultStopTarget(direction, price);
  currentDrawings.trades.push({
    id: newTradeId(),
    kind: "entry",
    direction, size, note,
    time, price,
    stopPrice: stopPrice != null ? stopPrice : defaults.stop,
    targetPrice: targetPrice != null ? targetPrice : defaults.target,
    pairedExitId: null,
  });
  persistTrades();
  applyAllMarkers();
  renderTradesList();
  renderTradeLevelLines();
  renderTradeConnectors();
}

function addExit({ size, note, time, price }) {
  const trade = {
    id: newTradeId(),
    kind: "exit",
    size, note,
    time, price,
    pairedEntryId: null,
    pairedDirection: null,
    realizedPct: null,
  };
  // LIFO: pair with the most-recent unpaired entry on this ticker.
  for (let i = currentDrawings.trades.length - 1; i >= 0; i--) {
    const t = currentDrawings.trades[i];
    if (t.kind === "entry" && !t.pairedExitId) {
      t.pairedExitId = trade.id;
      trade.pairedEntryId = t.id;
      trade.pairedDirection = t.direction;
      const sign = t.direction === "long" ? 1 : -1;
      trade.realizedPct = sign * (price - t.price) / t.price * 100;
      break;
    }
  }
  currentDrawings.trades.push(trade);
  persistTrades();
  applyAllMarkers();
  renderTradesList();
  renderTradeLevelLines();
  renderTradeConnectors();
}

function deleteTrade(id) {
  const trades = currentDrawings.trades;
  const idx = trades.findIndex(t => t.id === id);
  if (idx < 0) return;
  const removed = trades[idx];
  // Unpair partner so the surviving side becomes "open" again.
  if (removed.kind === "entry" && removed.pairedExitId) {
    const partner = trades.find(t => t.id === removed.pairedExitId);
    if (partner) {
      partner.pairedEntryId = null;
      partner.pairedDirection = null;
      partner.realizedPct = null;
    }
  } else if (removed.kind === "exit" && removed.pairedEntryId) {
    const partner = trades.find(t => t.id === removed.pairedEntryId);
    if (partner) partner.pairedExitId = null;
  }
  trades.splice(idx, 1);
  persistTrades();
  applyAllMarkers();
  renderTradesList();
  renderTradeLevelLines();
  renderTradeConnectors();
}

// --- Open-trade level lines (entry / stop / target) ---------------------
// Each open entry gets 3 horizontal priceLines on the candle series. The
// lines are draggable: mousedown/touchstart inside a small price tolerance
// around an active line starts a drag; mousemove updates the line live;
// release saves the new price to the trade record.

// Map of trade.id -> { entry: lineSeriesHandle, stop: ..., target: ... }
// Each line is a 2-point LineSeries from (entry_time, price) to the right
// edge of the chart, so it doesn't bleed back into pre-entry history.
const _tradeLevelLines = new Map();

const TRADE_COLOR = "#e69138";
const LEVEL_LINE_COLORS = {
  entry: TRADE_COLOR,
  stop: TRADE_COLOR,
  target: TRADE_COLOR,
};

function _clearTradeLevelLines() {
  for (const handles of _tradeLevelLines.values()) {
    for (const s of Object.values(handles)) {
      try { priceChart.removeSeries(s); } catch {}
    }
  }
  _tradeLevelLines.clear();
}

function _makeTradeLevelSeries(t, kind, price) {
  const dash = kind === "entry";
  const series = priceChart.addLineSeries({
    color: LEVEL_LINE_COLORS[kind],
    lineWidth: 1.5,
    lineStyle: dash ? LightweightCharts.LineStyle.Dashed : LightweightCharts.LineStyle.Solid,
    priceLineVisible: false,
    lastValueVisible: true,
    crosshairMarkerVisible: false,
    title: `${t.direction === "long" ? "L" : "S"} ${kind}`,
  });
  const endTime = (lastChartRightTime && String(lastChartRightTime) > String(t.time))
    ? lastChartRightTime : t.time;
  series.setData([
    { time: t.time, value: price },
    { time: endTime, value: price },
  ]);
  return series;
}

function renderTradeLevelLines() {
  _clearTradeLevelLines();
  if (!lastChartRightTime) return;
  const trades = currentDrawings?.trades || [];
  for (const t of trades) {
    if (t.kind !== "entry") continue;
    if (t.pairedExitId) continue;  // closed — drop the lines
    if (t.stopPrice == null || t.targetPrice == null) continue;
    _tradeLevelLines.set(t.id, {
      entry: _makeTradeLevelSeries(t, "entry", t.price),
      stop:  _makeTradeLevelSeries(t, "stop",  t.stopPrice),
      target:_makeTradeLevelSeries(t, "target",t.targetPrice),
    });
  }
}

// --- Closed-trade connectors --------------------------------------------
// For each paired entry+exit, a dotted amber segment from entry bar to
// exit bar. Realized P&L is shown via the exit marker's text label.
const _tradeConnectors = new Map();  // exit.id -> series handle

function _clearTradeConnectors() {
  for (const s of _tradeConnectors.values()) {
    try { priceChart.removeSeries(s); } catch {}
  }
  _tradeConnectors.clear();
}

function renderTradeConnectors() {
  _clearTradeConnectors();
  const trades = currentDrawings?.trades || [];
  const entryById = new Map();
  for (const t of trades) if (t.kind === "entry") entryById.set(t.id, t);

  for (const t of trades) {
    if (t.kind !== "exit" || !t.pairedEntryId) continue;
    const entry = entryById.get(t.pairedEntryId);
    if (!entry) continue;
    const series = priceChart.addLineSeries({
      color: TRADE_COLOR,
      lineWidth: 1.5,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    const points = [
      { time: entry.time, value: entry.price },
      { time: t.time, value: t.price },
    ].sort((a, b) => String(a.time).localeCompare(String(b.time)));
    series.setData(points);
    _tradeConnectors.set(t.id, series);
  }
}

// Drag interaction. Hit-test by y-coordinate → price → check each open
// trade's stored prices within a tolerance proportional to current chart
// price range. ns-resize cursor when hovering an active line.
let _activeDrag = null;  // { tradeId, kind, lineHandle }
const _priceProximityFrac = 0.005;  // 0.5% of last close, rough hit area

function _hitTestLevelLine(y) {
  const price = candleSeries.coordinateToPrice(y);
  if (price == null) return null;
  // Compute tolerance from the visible price range so it works across
  // very different price scales (penny stocks vs $500 names).
  const visible = candleSeries.priceScale().getVisibleRange?.();
  let tolerance;
  if (visible && visible.maxValue && visible.minValue) {
    tolerance = (visible.maxValue - visible.minValue) * 0.012;
  } else {
    tolerance = Math.abs(price) * _priceProximityFrac;
  }
  let best = null;
  let bestDist = Infinity;
  const trades = currentDrawings?.trades || [];
  for (const t of trades) {
    if (t.kind !== "entry" || t.pairedExitId) continue;
    if (t.stopPrice == null || t.targetPrice == null) continue;
    for (const kind of ["entry", "stop", "target"]) {
      const lp = kind === "entry" ? t.price : (kind === "stop" ? t.stopPrice : t.targetPrice);
      const d = Math.abs(price - lp);
      if (d < tolerance && d < bestDist) {
        bestDist = d;
        best = { tradeId: t.id, kind };
      }
    }
  }
  return best;
}

function _onChartMouseMoveForDrag(ev) {
  const rect = containers.price.getBoundingClientRect();
  const y = ev.clientY - rect.top;
  if (_activeDrag) {
    const newPrice = candleSeries.coordinateToPrice(y);
    if (newPrice == null || newPrice <= 0) return;
    const t = currentDrawings.trades.find(x => x.id === _activeDrag.tradeId);
    if (!t) return;
    if (_activeDrag.kind === "entry") t.price = newPrice;
    else if (_activeDrag.kind === "stop") t.stopPrice = newPrice;
    else if (_activeDrag.kind === "target") t.targetPrice = newPrice;
    try {
      const endTime = (lastChartRightTime && String(lastChartRightTime) > String(t.time))
        ? lastChartRightTime : t.time;
      _activeDrag.lineHandle.setData([
        { time: t.time, value: newPrice },
        { time: endTime, value: newPrice },
      ]);
    } catch {}
    ev.preventDefault?.();
  } else {
    // Hover-only cursor feedback.
    containers.price.style.cursor = _hitTestLevelLine(y) ? "ns-resize" : "";
  }
}

function _onChartMouseDownForDrag(ev) {
  if (ev.button !== undefined && ev.button !== 0) return;
  const rect = containers.price.getBoundingClientRect();
  const y = (ev.touches ? ev.touches[0].clientY : ev.clientY) - rect.top;
  const hit = _hitTestLevelLine(y);
  if (!hit) return;
  const handles = _tradeLevelLines.get(hit.tradeId);
  if (!handles) return;
  _activeDrag = { tradeId: hit.tradeId, kind: hit.kind, lineHandle: handles[hit.kind] };
  containers.price.style.cursor = "ns-resize";
  ev.preventDefault?.();
}

function _onChartMouseUpForDrag() {
  if (!_activeDrag) return;
  _activeDrag = null;
  persistTrades();
  renderTradesList();
}

containers.price.addEventListener("mousedown", _onChartMouseDownForDrag);
containers.price.addEventListener("mousemove", _onChartMouseMoveForDrag);
window.addEventListener("mouseup", _onChartMouseUpForDrag);

// Touch: same logic, slightly different event sources.
containers.price.addEventListener("touchstart", (ev) => {
  if (!ev.touches || ev.touches.length !== 1) return;
  _onChartMouseDownForDrag(ev);
}, { passive: false });
containers.price.addEventListener("touchmove", (ev) => {
  if (!_activeDrag || !ev.touches || ev.touches.length !== 1) return;
  const fake = { clientX: ev.touches[0].clientX, clientY: ev.touches[0].clientY, preventDefault: () => ev.preventDefault() };
  _onChartMouseMoveForDrag(fake);
}, { passive: false });
window.addEventListener("touchend", _onChartMouseUpForDrag);

function manualTradeToMarker(t) {
  if (t.kind === "entry") {
    const isLong = t.direction === "long";
    return {
      time: t.time,
      position: isLong ? "belowBar" : "aboveBar",
      color: TRADE_COLOR,
      shape: "circle",
      text: `${isLong ? "BUY" : "SHORT"} ${t.size}`,
    };
  }
  const dir = t.pairedDirection;
  const v = t.realizedPct;
  const text = v == null
    ? `EXIT ${t.size}`
    : `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
  // Place exit opposite to its paired entry (or above by default).
  const position = dir === "short" ? "belowBar" : "aboveBar";
  return { time: t.time, position, color: TRADE_COLOR, shape: "circle", text };
}

function openTradeModal(kind, time, price) {
  pendingTrade = { kind, time, price };
  tradeModalTitle.textContent = kind === "entry" ? "Mark Entry" : "Mark Exit";
  tradeModalMeta.textContent =
    `${activeTicker || ""} · ${formatTradeDate(time)} · $${price.toFixed(2)}`;
  tradeModalDirField.style.display = kind === "entry" ? "" : "none";
  if (tradePricesGroup) tradePricesGroup.style.display = kind === "entry" ? "" : "none";
  const longRadio = tradeModalForm.querySelector('input[name="trade-direction"][value="long"]');
  if (longRadio) longRadio.checked = true;
  tradeSizeInput.value = "";
  tradeNoteInput.value = "";
  if (kind === "entry") {
    // Prefill defaults assuming "long" since that's the initial radio state.
    const d = _defaultStopTarget("long", price);
    if (tradeStopInput) tradeStopInput.value = d.stop.toFixed(2);
    if (tradeTargetInput) tradeTargetInput.value = d.target.toFixed(2);
  }
  tradeModalEl.hidden = false;
  setTimeout(() => tradeSizeInput.focus(), 0);
}

// When user flips Long ↔ Short in the modal, re-prefill stop/target so the
// defaults reflect the chosen side. Only fires if the user hasn't manually
// edited those fields yet.
tradeModalForm?.querySelectorAll('input[name="trade-direction"]').forEach(radio => {
  radio.addEventListener("change", () => {
    if (!pendingTrade || pendingTrade.kind !== "entry") return;
    const dir = tradeModalForm.querySelector('input[name="trade-direction"]:checked')?.value || "long";
    const d = _defaultStopTarget(dir, pendingTrade.price);
    if (tradeStopInput) tradeStopInput.value = d.stop.toFixed(2);
    if (tradeTargetInput) tradeTargetInput.value = d.target.toFixed(2);
  });
});

function closeTradeModal() {
  pendingTrade = null;
  tradeModalEl.hidden = true;
}

function formatTradeDate(time) {
  if (time == null) return "";
  if (typeof time === "string") return time;
  if (typeof time === "number") {
    return new Date(time * 1000).toISOString().slice(0, 10);
  }
  if (typeof time === "object" && time.year != null) {
    const m = String(time.month).padStart(2, "0");
    const d = String(time.day).padStart(2, "0");
    return `${time.year}-${m}-${d}`;
  }
  return String(time);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderTradesList() {
  if (!tradesListEl) return;
  tradesListEl.innerHTML = "";
  const trades = currentDrawings.trades || [];
  if (!trades.length) {
    tradesEmptyEl.style.display = "";
    return;
  }
  tradesEmptyEl.style.display = "none";
  const sorted = [...trades].sort((a, b) =>
    String(b.time).localeCompare(String(a.time))
  );
  for (const t of sorted) {
    const li = document.createElement("li");
    li.className = "my-trade-item";

    let kindClass, kindLabel, plHtml;
    if (t.kind === "entry") {
      kindClass = t.direction === "long" ? "entry-long" : "entry-short";
      kindLabel = t.direction === "long" ? "BUY" : "SHORT";
      plHtml = t.pairedExitId
        ? `<span class="mt-pl">closed</span>`
        : `<span class="mt-pl open">open</span>`;
    } else {
      const open = !t.pairedEntryId;
      kindClass = open ? "exit-open" : "exit";
      kindLabel = "EXIT";
      if (open) {
        plHtml = `<span class="mt-pl open">unpaired</span>`;
      } else {
        const v = t.realizedPct ?? 0;
        const cls = v > 0 ? "pos" : v < 0 ? "neg" : "";
        plHtml = `<span class="mt-pl ${cls}">${v >= 0 ? "+" : ""}${v.toFixed(2)}%</span>`;
      }
    }

    const detail = `${t.size} @ $${Number(t.price).toFixed(2)}`;
    const sub = `${formatTradeDate(t.time)}${t.note ? " · " + escapeHtml(t.note) : ""}`;
    li.innerHTML = `
      <span class="mt-kind ${kindClass}">${kindLabel}</span>
      <span class="mt-detail">${detail}</span>
      ${plHtml}
      <button class="mt-delete" type="button" title="Delete" aria-label="Delete">${ICON_X}</button>
      <span class="mt-sub">${sub}</span>
    `;
    li.querySelector(".mt-delete").addEventListener("click", () => deleteTrade(t.id));
    tradesListEl.appendChild(li);
  }
}

tradeModalForm?.addEventListener("submit", (e) => {
  e.preventDefault();
  if (!pendingTrade) return;
  const size = Number(tradeSizeInput.value);
  if (!(size > 0)) return;
  const note = tradeNoteInput.value.trim();
  const direction = tradeModalForm.querySelector('input[name="trade-direction"]:checked')?.value || "long";
  const { kind, time, price } = pendingTrade;
  if (kind === "entry") {
    const stopRaw = tradeStopInput?.value;
    const targetRaw = tradeTargetInput?.value;
    const stopPrice = stopRaw ? Number(stopRaw) : null;
    const targetPrice = targetRaw ? Number(targetRaw) : null;
    addEntry({ direction, size, note, time, price, stopPrice, targetPrice });
  }
  else addExit({ size, note, time, price });
  closeTradeModal();
});

document.getElementById("trade-cancel")?.addEventListener("click", closeTradeModal);
document.querySelector(".trade-modal-backdrop")?.addEventListener("click", closeTradeModal);

// --- Watchlist -----------------------------------------------------------
const WATCHLIST_KEY = "swingtrader.watchlist";
const DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "PLTR", "AMD"];

const watchlistEl = document.getElementById("watchlist");
const wlForm = document.getElementById("watchlist-add-form");
const wlInput = document.getElementById("watchlist-add-input");

function loadWatchlist() {
  try {
    const raw = localStorage.getItem(WATCHLIST_KEY);
    if (raw == null) return [...DEFAULT_WATCHLIST];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch { return []; }
}

function saveWatchlist(items) {
  localStorage.setItem(WATCHLIST_KEY, JSON.stringify(items));
}

function addWatchlistTicker(raw) {
  const ticker = String(raw || "").toUpperCase().trim();
  if (!ticker) return;
  const items = loadWatchlist();
  if (items.includes(ticker)) return;
  items.push(ticker);
  saveWatchlist(items);
  renderWatchlist();
}

function removeWatchlistTicker(ticker) {
  saveWatchlist(loadWatchlist().filter(t => t !== ticker));
  renderWatchlist();
}

async function fetchSummary(ticker) {
  const resp = await fetch(`/api/summary/${encodeURIComponent(ticker)}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return await resp.json();
}

let activeTicker = null;

function setActiveTicker(ticker) {
  activeTicker = ticker;
  for (const li of watchlistEl.querySelectorAll(".watchlist-item")) {
    li.classList.toggle("active", li.dataset.ticker === ticker);
  }
}

function renderWatchlist() {
  const items = loadWatchlist();
  watchlistEl.innerHTML = "";
  for (const ticker of items) {
    const li = document.createElement("li");
    li.className = "watchlist-item loading";
    if (ticker === activeTicker) li.classList.add("active");
    li.dataset.ticker = ticker;
    li.innerHTML = `
      <div class="wl-row1">
        <span class="wl-ticker">${ticker}</span>
        <button class="wl-remove" type="button" title="Remove" aria-label="Remove">${ICON_X}</button>
      </div>
      <div class="wl-row2">
        <span class="wl-price">…</span>
        <span class="wl-sd">…</span>
      </div>
    `;
    li.addEventListener("click", () => {
      input.value = ticker;
      loadTicker(ticker);
    });
    li.querySelector(".wl-remove").addEventListener("click", (e) => {
      e.stopPropagation();
      removeWatchlistTicker(ticker);
    });
    watchlistEl.appendChild(li);
  }
  refreshWatchlistSummaries();
}

function refreshWatchlistSummaries() {
  for (const li of watchlistEl.querySelectorAll(".watchlist-item")) {
    const ticker = li.dataset.ticker;
    fetchSummary(ticker).then(s => {
      li.classList.remove("loading", "error");
      li.querySelector(".wl-price").textContent =
        s.current_price != null ? `$${s.current_price.toFixed(2)}` : "—";
      const sdEl = li.querySelector(".wl-sd");
      if (s.sd_position == null || Number.isNaN(s.sd_position)) {
        sdEl.textContent = "—";
        sdEl.className = "wl-sd";
      } else {
        const sd = s.sd_position;
        sdEl.textContent = `${sd >= 0 ? "+" : ""}${sd.toFixed(2)}σ`;
        sdEl.className = "wl-sd" + (sd > 1 ? " neg" : sd < -1 ? " pos" : "");
      }
    }).catch(() => {
      li.classList.remove("loading");
      li.classList.add("error");
      li.querySelector(".wl-price").textContent = "error";
      li.querySelector(".wl-sd").textContent = "";
    });
  }
}

wlForm.addEventListener("submit", (e) => {
  e.preventDefault();
  addWatchlistTicker(wlInput.value);
  wlInput.value = "";
});

// Drag-to-reorder via SortableJS (CDN-loaded). On drag end, read the new
// DOM order, save it (which also syncs to server + re-pushes the rule).
function initWatchlistSortable() {
  if (typeof Sortable === "undefined") return;  // CDN failed; ignore
  if (!watchlistEl || watchlistEl.dataset.sortableInit) return;
  Sortable.create(watchlistEl, {
    animation: 150,
    delay: 180,
    delayOnTouchOnly: true,
    touchStartThreshold: 6,
    filter: "button, input, a",
    preventOnFilter: false,
    ghostClass: "wl-drag-ghost",
    dragClass: "wl-drag-active",
    chosenClass: "wl-drag-chosen",
    onEnd: () => {
      const newOrder = Array.from(watchlistEl.querySelectorAll(".watchlist-item"))
        .map(li => li.dataset.ticker)
        .filter(Boolean);
      if (newOrder.length === 0) return;
      saveWatchlist(newOrder);   // patched wrapper also calls syncWatchlistToServer + scheduleSignalRuleSync
    },
  });
  watchlistEl.dataset.sortableInit = "1";
}

// --- Server sync: push watchlist to backend so the scheduler sees it -----
async function syncWatchlistToServer() {
  try {
    await fetch("/api/watchlist", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tickers: loadWatchlist() }),
    });
  } catch (err) { /* non-fatal */ }
}

// Fetch the watchlist from the server on boot. Server wins if it has data
// (lets a phone edit show up on the desktop). If server is empty but local
// has tickers, push local up.
async function fetchAndMergeWatchlist() {
  let serverTickers;
  try {
    const resp = await fetch("/api/watchlist");
    if (!resp.ok) return;
    const data = await resp.json();
    serverTickers = Array.isArray(data.tickers) ? data.tickers : [];
  } catch { return; }
  const local = loadWatchlist();
  if (serverTickers.length > 0) {
    // Replace local cache without re-pushing (we just got this from server).
    localStorage.setItem(WATCHLIST_KEY, JSON.stringify(serverTickers));
  } else if (local.length > 0) {
    await syncWatchlistToServer();
  }
}

// Patch save to also sync.
const _originalSave = saveWatchlist;
saveWatchlist = function (items) {
  _originalSave(items);
  syncWatchlistToServer();
  scheduleSignalRuleSync();
};

// --- Alerts panel --------------------------------------------------------
const alertsListEl = document.getElementById("alerts-list");
const scanBtn = document.getElementById("alerts-scan-btn");

async function refreshAlerts() {
  try {
    const resp = await fetch("/api/alerts?limit=25");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    renderAlerts(data.alerts || []);
  } catch (err) { /* keep previous render */ }
}

function renderAlerts(alerts) {
  alertsListEl.innerHTML = "";
  if (alerts.length === 0) {
    const empty = document.createElement("li");
    empty.className = "alerts-empty";
    empty.textContent = "No alerts yet";
    alertsListEl.appendChild(empty);
    return;
  }
  for (const a of alerts) {
    const li = document.createElement("li");
    li.className = "alert-item";
    const stars = "★".repeat(a.confidence) + "☆".repeat(3 - a.confidence);
    const firedAt = a.fired_at ? new Date(a.fired_at).toLocaleString() : a.bar_date;
    li.innerHTML = `
      <div class="alert-row1">
        <span class="alert-ticker">${a.ticker}</span>
        <span class="alert-dir ${a.direction}">${a.direction}</span>
        <span class="alert-stars">${stars}</span>
      </div>
      <div class="alert-row2">
        <span>${a.bar_date} · ${a.sd_position >= 0 ? "+" : ""}${a.sd_position.toFixed(2)}σ</span>
        <span>$${a.price.toFixed(2)}</span>
      </div>
    `;
    li.addEventListener("click", () => {
      input.value = a.ticker;
      loadTicker(a.ticker);
    });
    alertsListEl.appendChild(li);
  }
}

scanBtn.addEventListener("click", async () => {
  scanBtn.classList.add("scanning");
  try {
    await fetch("/api/alerts/scan", { method: "POST" });
    await refreshAlerts();
  } catch (err) { /* ignore */ }
  finally {
    scanBtn.classList.remove("scanning");
  }
});

// Every time slot the price chart renders (candles ∪ ichimoku future).
// Populated by renderChart(); consumed by _padDynamicPaneTimeAxes() to keep
// dynamic-pane logical indexing in lock-step with the price chart's.
let _priceChartAllTimes = [];

// --- Dynamic indicator panes ---------------------------------------------
// One entry per active indicator with `has_own_pane: true`. Each entry owns
// a DOM container, a Lightweight Charts instance, and lists of series it
// created (so we can wipe everything cleanly when the chart reloads).
//
// Items with `pane: "price"` get rendered onto priceChart/candleSeries
// instead; we track those series/price-lines under the same indicator
// entry so a reload still wipes them.
const _dynamicPanes = new Map();  // indicator_id -> { chart, chartEl, container, series[], priceSeries[], priceLines[] }
const mainEl = document.querySelector("main");

function _lwLineStyle(s) {
  const LS = LightweightCharts.LineStyle;
  switch (String(s || "").toLowerCase()) {
    case "dashed": return LS.Dashed;
    case "dotted": return LS.Dotted;
    case "large_dashed": return LS.LargeDashed;
    case "sparse_dotted": return LS.SparseDotted;
    default: return LS.Solid;
  }
}

// Returns the input color as rgba() with the given alpha. Accepts hex
// (#rgb / #rrggbb) or pass-through rgba/rgb strings.
function _withAlpha(color, alpha) {
  if (!color) return `rgba(128,128,128,${alpha})`;
  if (color.startsWith("rgba") || color.startsWith("rgb")) return color;
  let hex = color.replace("#", "");
  if (hex.length === 3) hex = hex.split("").map(c => c + c).join("");
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// Per-indicator pane heights (px), persisted to localStorage so the user's
// resize stays sticky across reloads.
const PANE_HEIGHTS_KEY = "swingtrader.paneHeights";
const DEFAULT_PANE_HEIGHT = 140;
const MIN_PANE_HEIGHT = 60;
let _paneHeights = (function _loadPaneHeights() {
  try {
    const raw = localStorage.getItem(PANE_HEIGHTS_KEY);
    const obj = raw ? JSON.parse(raw) : {};
    const m = new Map();
    for (const [k, v] of Object.entries(obj)) {
      const n = Number(v);
      if (Number.isFinite(n) && n >= MIN_PANE_HEIGHT) m.set(k, n);
    }
    return m;
  } catch { return new Map(); }
})();
function _savePaneHeights() {
  try {
    localStorage.setItem(PANE_HEIGHTS_KEY, JSON.stringify(Object.fromEntries(_paneHeights)));
  } catch {}
}

function _updateMainGridRows(ownPaneIds) {
  // The price pane (minmax(0,1fr)) absorbs remaining space; each dynamic
  // own-pane indicator adds an explicit row using its persisted height
  // (or DEFAULT_PANE_HEIGHT). Caller can pass an explicit list of indicator
  // ids (used during renderCustomIndicators before _dynamicPanes is
  // populated); otherwise we derive the list from current _dynamicPanes.
  let ids;
  if (Array.isArray(ownPaneIds)) {
    ids = ownPaneIds;
  } else {
    ids = [];
    for (const [id, p] of _dynamicPanes) if (p.container) ids.push(id);
  }
  let rows = "minmax(0, 1fr)";
  for (const id of ids) {
    const h = _paneHeights.get(id) || DEFAULT_PANE_HEIGHT;
    rows += ` ${h}px`;
  }
  mainEl.style.gridTemplateRows = rows;
}

function _tearDownDynamicPanes() {
  for (const pane of _dynamicPanes.values()) {
    for (const handle of pane.priceLines) {
      try { candleSeries.removePriceLine(handle); } catch {}
    }
    for (const s of pane.priceSeries) {
      try { priceChart.removeSeries(s); } catch {}
    }
    // Only dedicated-pane entries own their own chart/container — overlay-only
    // entries share priceChart and have no DOM container.
    if (pane.container) {
      try { pane.chart.remove(); } catch {}
      try { pane.container.remove(); } catch {}
      const idx = allCharts.indexOf(pane.chart);
      if (idx >= 0) allCharts.splice(idx, 1);
    }
  }
  _dynamicPanes.clear();
  _updateMainGridRows();
}

function _attachPaneResize(handle, container, indicatorId) {
  // Pointer-based row resize. Drag up = pane taller; drag down = pane
  // shorter. Persisted per indicator_id.
  let startY = 0;
  let startHeight = 0;
  const onMove = (e) => {
    const dy = e.clientY - startY;
    const maxH = Math.max(MIN_PANE_HEIGHT * 2, window.innerHeight - 220);
    const h = Math.max(MIN_PANE_HEIGHT, Math.min(maxH, startHeight - dy));
    _paneHeights.set(indicatorId, h);
    _updateMainGridRows();
  };
  const onUp = () => {
    handle.classList.remove("dragging");
    document.removeEventListener("pointermove", onMove);
    document.removeEventListener("pointerup", onUp);
    document.body.style.userSelect = "";
    _savePaneHeights();
  };
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.classList.add("dragging");
    startY = e.clientY;
    startHeight = container.clientHeight;
    document.body.style.userSelect = "none";
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  });
}

function _createDynamicPane(indicator) {
  const container = document.createElement("div");
  container.className = "pane pane-dynamic";
  container.dataset.indicatorId = indicator.indicator_id;
  const handle = document.createElement("div");
  handle.className = "pane-resize-handle";
  handle.title = "Drag to resize";
  _attachPaneResize(handle, container, indicator.indicator_id);
  const label = document.createElement("div");
  label.className = "pane-label";
  label.textContent = indicator.pane_title || indicator.name;
  const chartEl = document.createElement("div");
  chartEl.className = "chart";
  container.appendChild(handle);
  container.appendChild(label);
  container.appendChild(chartEl);
  mainEl.insertBefore(container, document.getElementById("status"));

  const chart = LightweightCharts.createChart(chartEl, indicatorOptions(false));
  // Make the newly-created chart participate in time-scale + crosshair sync.
  allCharts.push(chart);
  syncTimeRange(chart);
  syncCrosshair(chart);

  if (indicator.pane_y_range && indicator.pane_y_range.length === 2) {
    const [lo, hi] = indicator.pane_y_range;
    chart.priceScale("right").applyOptions({
      autoScale: false,
      // Lightweight Charts has no direct fixed-range API; we approximate by
      // disabling autoscale and trusting the indicator to keep data inside.
    });
    void lo; void hi;
  }

  return { chart, chartEl, container, series: [], priceSeries: [], priceLines: [] };
}

function _renderPlotItem(item, paneEntry) {
  const style = item.style || {};
  const onPrice = item.pane === "price";
  const targetChart = onPrice ? priceChart : paneEntry.chart;

  switch (item.kind) {
    case "line": {
      const s = targetChart.addLineSeries({
        color: style.color || "#cccccc",
        lineWidth: style.lineWidth ?? 1.5,
        lineStyle: _lwLineStyle(style.lineStyle),
        priceLineVisible: false,
        lastValueVisible: !!style.lastValueVisible,
        crosshairMarkerVisible: !onPrice,
        title: style.title || "",
      });
      s.setData(item.data || []);
      (onPrice ? paneEntry.priceSeries : paneEntry.series).push(s);
      // Markers piggyback on the host series; record it for later setMarkers calls.
      paneEntry._lastSeries = s;
      break;
    }
    case "candle": {
      const s = targetChart.addCandlestickSeries({
        upColor: style.upColor || "#26a69a",
        downColor: style.downColor || "#ef5350",
        borderUpColor: style.borderUpColor || style.upColor || "#26a69a",
        borderDownColor: style.borderDownColor || style.downColor || "#ef5350",
        wickUpColor: style.wickUpColor || style.upColor || "#26a69a",
        wickDownColor: style.wickDownColor || style.downColor || "#ef5350",
      });
      s.setData(item.data || []);
      (onPrice ? paneEntry.priceSeries : paneEntry.series).push(s);
      paneEntry._lastSeries = s;
      break;
    }
    case "area": {
      // Filled area between line and a baseline value (default 0). Used by
      // oscillators like VuManChu WT that want a wave-shaped fill above
      // and below zero. Supports asymmetric colors (topColor / bottomColor)
      // so an indicator like MFI can fill green-above-zero / red-below.
      const baseValue = style.baseValue ?? 0;
      const topColor = style.topColor || style.color || "#cccccc";
      const bottomColor = style.bottomColor || style.color || "#cccccc";
      const opacity = style.fillOpacity ?? 0.32;
      const s = targetChart.addBaselineSeries({
        baseValue: { type: "price", price: baseValue },
        topLineColor: topColor,
        topFillColor1: _withAlpha(topColor, opacity),
        topFillColor2: _withAlpha(topColor, opacity * 0.1),
        bottomLineColor: bottomColor,
        bottomFillColor1: _withAlpha(bottomColor, opacity * 0.1),
        bottomFillColor2: _withAlpha(bottomColor, opacity),
        lineWidth: style.lineWidth ?? 1,
        priceLineVisible: false,
        lastValueVisible: !!style.lastValueVisible,
        crosshairMarkerVisible: !onPrice,
      });
      s.setData(item.data || []);
      (onPrice ? paneEntry.priceSeries : paneEntry.series).push(s);
      paneEntry._lastSeries = s;
      break;
    }
    case "histogram": {
      const s = targetChart.addHistogramSeries({
        color: style.color || "#888888",
        priceLineVisible: false,
        lastValueVisible: false,
        base: style.base ?? 0,
      });
      s.setData(item.data || []);
      (onPrice ? paneEntry.priceSeries : paneEntry.series).push(s);
      paneEntry._lastSeries = s;
      break;
    }
    case "marker": {
      // Lightweight Charts markers must attach to an existing series. Use
      // the most-recently-created series in the same pane.
      const host = paneEntry._lastSeries || paneEntry.series[0] || paneEntry.priceSeries[0];
      if (host && Array.isArray(item.data)) {
        try { host.setMarkers(item.data.slice()); } catch {}
      }
      break;
    }
    case "price_line": {
      // Horizontal level. If pane="price", attaches to candleSeries on the
      // price pane (and is tracked for explicit cleanup). If pane="own", it
      // attaches to the most-recent series in this indicator's pane and
      // gets cleaned up automatically when the pane chart is removed.
      const host = item.pane === "own"
        ? (paneEntry._lastSeries || paneEntry.series[0])
        : candleSeries;
      if (!host) break;
      for (const entry of item.data || []) {
        const handle = host.createPriceLine({
          price: entry.price,
          color: entry.color || style.color || "#cccccc",
          lineWidth: entry.lineWidth ?? style.lineWidth ?? 1,
          lineStyle: _lwLineStyle(entry.lineStyle || style.lineStyle),
          axisLabelVisible: entry.axisLabelVisible !== false,
          title: entry.title || "",
        });
        // Only track candleSeries handles for explicit teardown; own-pane
        // handles are destroyed with the chart.
        if (item.pane !== "own") paneEntry.priceLines.push(handle);
      }
      break;
    }
    case "fill": {
      // Not yet implemented — Lightweight Charts needs paired AreaSeries
      // or a baseline trick. Skipped for now; emit two line items instead.
      break;
    }
  }
}

// Add a hidden whitespace series to each dynamic pane covering every time
// slot the price chart renders. This makes their logical indexing match the
// price chart so the time-scale sync doesn't shift bars left within the
// pane.
function _padDynamicPaneTimeAxes() {
  if (_priceChartAllTimes.length === 0) return;
  const whitespace = _priceChartAllTimes.map(t => ({ time: t }));
  for (const p of _dynamicPanes.values()) {
    if (!p.container) continue;
    const s = p.chart.addLineSeries({
      visible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    s.setData(whitespace);
    p.series.push(s);
  }
}

// After dynamic panes are populated, equalize all right-price-scale widths so
// the time axes line up horizontally. Without this, a pane whose axis labels
// happen to be wider (e.g. "-100.00" vs "100") shifts its time axis left by
// the difference, making bars no longer align with the price chart above.
function _alignPriceScaleWidths() {
  // Wait one frame so each chart has actually laid out its axis text before
  // we read .width(). priceChart.priceScale("right").width() is computed
  // from the longest label currently shown.
  requestAnimationFrame(() => {
    const charts = [priceChart];
    for (const p of _dynamicPanes.values()) {
      if (p.container) charts.push(p.chart);
    }
    if (charts.length < 2) {
      priceChart.priceScale("right").applyOptions({ minimumWidth: 0 });
      return;
    }
    let maxWidth = 0;
    for (const c of charts) {
      try {
        const w = c.priceScale("right").width();
        if (w > maxWidth) maxWidth = w;
      } catch {}
    }
    if (maxWidth <= 0) return;
    for (const c of charts) {
      try {
        c.priceScale("right").applyOptions({ minimumWidth: maxWidth });
      } catch {}
    }
  });
}

function renderCustomIndicators(indicators) {
  _tearDownDynamicPanes();
  // Reserve grid rows BEFORE creating chart instances so their containers
  // have nonzero height at construction time. (autoSize handles late
  // resizing too, but starting with the right layout avoids the brief
  // 0-height flash and a missing-data fallback.) Pass the id list so
  // persisted heights apply immediately on first render.
  const ownPaneIds = indicators.filter(i => i.has_own_pane).map(i => i.indicator_id);
  _updateMainGridRows(ownPaneIds);
  for (const ind of indicators) {
    const paneEntry = ind.has_own_pane ? _createDynamicPane(ind) : {
      chart: priceChart, chartEl: containers.price, container: null,
      series: [], priceSeries: [], priceLines: [],
    };
    for (const item of ind.items || []) {
      _renderPlotItem(item, paneEntry);
    }
    _dynamicPanes.set(ind.indicator_id, paneEntry);
  }
  _padDynamicPaneTimeAxes();
  _alignPriceScaleWidths();
}

// --- Indicators picker ---------------------------------------------------
// Sidebar section lists active indicators with edit/remove; "+" opens a
// modal showing the full catalog and per-indicator parameter form. Saves
// go to /api/indicators/active and trigger a chart reload so the backend
// re-computes the new selection.

let _indicatorsCatalog = [];   // [{id, name, category, description, has_own_pane, params: [...]}]
let _activeIndicators = [];    // [{indicator_id, params}]
let _pickerCurrent = null;     // {id, params, isEdit} — what's being configured in the modal right now

const _indActiveListEl = document.getElementById("indicators-active-list");
const _indEmptyEl = document.getElementById("indicators-empty");
const _indModalEl = document.getElementById("indicators-modal");
const _indSearchEl = document.getElementById("indicators-search");
const _indCatalogEl = document.getElementById("indicators-catalog");
const _indParamsEl = document.getElementById("indicators-params");
const _indParamsTitleEl = document.getElementById("indicators-params-title");
const _indParamsDescEl = document.getElementById("indicators-params-desc");
const _indParamsFormEl = document.getElementById("indicators-params-form");
const _indParamsBackBtn = document.getElementById("indicators-params-back");
const _indSaveBtn = document.getElementById("indicators-modal-save");
const _indCancelBtn = document.getElementById("indicators-modal-cancel");
const _indAddBtn = document.getElementById("indicators-add-btn");

async function _fetchIndicatorsCatalog() {
  try {
    const resp = await fetch("/api/indicators");
    if (!resp.ok) return;
    const data = await resp.json();
    _indicatorsCatalog = data.indicators || [];
  } catch { /* keep empty */ }
}

async function _fetchActiveIndicators() {
  try {
    const resp = await fetch("/api/indicators/active");
    if (!resp.ok) return;
    const data = await resp.json();
    _activeIndicators = data.active || [];
  } catch { /* keep empty */ }
}

async function _saveActiveIndicators() {
  try {
    await fetch("/api/indicators/active", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ active: _activeIndicators }),
    });
  } catch { /* non-fatal */ }
}

function _findIndicatorSpec(id) {
  return _indicatorsCatalog.find(i => i.id === id);
}

function _paramSummary(spec, params) {
  if (!spec || !spec.params) return "";
  // Render numeric/select params in declaration order; skip colors.
  const bits = [];
  for (const p of spec.params) {
    if (p.type === "color") continue;
    const v = params[p.id] ?? p.default;
    bits.push(String(v));
  }
  return bits.length ? `(${bits.join(", ")})` : "";
}

function _renderActiveIndicatorsList() {
  if (!_indActiveListEl) return;
  _indActiveListEl.innerHTML = "";
  if (_activeIndicators.length === 0) {
    if (_indEmptyEl) _indEmptyEl.hidden = false;
    return;
  }
  if (_indEmptyEl) _indEmptyEl.hidden = true;
  for (const entry of _activeIndicators) {
    const spec = _findIndicatorSpec(entry.indicator_id);
    if (!spec) continue;
    const li = document.createElement("li");
    li.className = "indicator-active-item";
    li.innerHTML = `
      <div class="indicator-active-row">
        <span class="name">${spec.name}</span>
        <span class="params">${_paramSummary(spec, entry.params || {})}</span>
      </div>
      <div class="indicator-active-actions">
        <button type="button" class="edit" title="Edit parameters" aria-label="Edit">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
        </button>
        <button type="button" class="remove" title="Remove" aria-label="Remove">${ICON_X}</button>
      </div>
    `;
    li.querySelector(".edit").addEventListener("click", () => _openPicker(entry.indicator_id));
    li.querySelector(".remove").addEventListener("click", async () => {
      _activeIndicators = _activeIndicators.filter(e => e.indicator_id !== entry.indicator_id);
      await _saveActiveIndicators();
      _renderActiveIndicatorsList();
      if (activeTicker) loadTicker(activeTicker);
    });
    _indActiveListEl.appendChild(li);
  }
}

function _renderCatalog(filter) {
  _indCatalogEl.innerHTML = "";
  const q = (filter || "").toLowerCase().trim();
  const matches = _indicatorsCatalog.filter(i =>
    !q || i.name.toLowerCase().includes(q) || i.category.toLowerCase().includes(q) || i.description.toLowerCase().includes(q)
  );
  for (const spec of matches) {
    const li = document.createElement("li");
    li.className = "indicator-catalog-item";
    li.innerHTML = `
      <div><span class="name">${spec.name}</span><span class="category">${spec.category}</span></div>
      <div class="desc">${spec.description || ""}</div>
    `;
    li.addEventListener("click", () => _showParamsForm(spec.id, /*existing*/ null));
    _indCatalogEl.appendChild(li);
  }
}

function _showParamsForm(indicatorId, existingParams) {
  const spec = _findIndicatorSpec(indicatorId);
  if (!spec) return;
  _pickerCurrent = {
    id: spec.id,
    params: { ...Object.fromEntries(spec.params.map(p => [p.id, p.default])), ...(existingParams || {}) },
    isEdit: !!existingParams,
  };
  _indParamsTitleEl.textContent = spec.name;
  _indParamsDescEl.textContent = spec.description || "";
  _indParamsFormEl.innerHTML = "";
  for (const p of spec.params) {
    const wrap = document.createElement("label");
    if (p.type === "select" && Array.isArray(p.options)) wrap.classList.add("full");
    if (p.help) wrap.classList.add("full");
    const labelText = document.createElement("span");
    labelText.textContent = p.label;
    wrap.appendChild(labelText);
    let inputEl;
    if (p.type === "bool") {
      inputEl = document.createElement("input");
      inputEl.type = "checkbox";
      inputEl.checked = !!_pickerCurrent.params[p.id];
      inputEl.addEventListener("change", () => { _pickerCurrent.params[p.id] = inputEl.checked; });
    } else if (p.type === "select") {
      inputEl = document.createElement("select");
      for (const opt of p.options || []) {
        const o = document.createElement("option");
        o.value = opt; o.textContent = opt;
        if (_pickerCurrent.params[p.id] === opt) o.selected = true;
        inputEl.appendChild(o);
      }
      inputEl.addEventListener("change", () => { _pickerCurrent.params[p.id] = inputEl.value; });
    } else if (p.type === "color") {
      inputEl = document.createElement("input");
      inputEl.type = "color";
      inputEl.value = _pickerCurrent.params[p.id];
      inputEl.addEventListener("change", () => { _pickerCurrent.params[p.id] = inputEl.value; });
    } else {
      inputEl = document.createElement("input");
      inputEl.type = "number";
      if (p.min != null) inputEl.min = String(p.min);
      if (p.max != null) inputEl.max = String(p.max);
      if (p.step != null) inputEl.step = String(p.step);
      inputEl.value = String(_pickerCurrent.params[p.id]);
      inputEl.addEventListener("input", () => {
        const v = p.type === "int" ? parseInt(inputEl.value, 10) : parseFloat(inputEl.value);
        if (!Number.isNaN(v)) _pickerCurrent.params[p.id] = v;
      });
    }
    wrap.appendChild(inputEl);
    if (p.help) {
      const help = document.createElement("span");
      help.style.fontSize = "9px";
      help.style.opacity = "0.7";
      help.textContent = p.help;
      wrap.appendChild(help);
    }
    _indParamsFormEl.appendChild(wrap);
  }
  _indCatalogEl.hidden = true;
  _indSearchEl.hidden = true;
  _indParamsEl.hidden = false;
  _indSaveBtn.disabled = false;
  _indSaveBtn.textContent = _pickerCurrent.isEdit ? "Save" : "Add";
}

function _openPicker(prefilledId /*optional*/) {
  _indModalEl.hidden = false;
  _indSearchEl.value = "";
  _indSearchEl.hidden = false;
  _indCatalogEl.hidden = false;
  _indParamsEl.hidden = true;
  _indSaveBtn.disabled = true;
  _renderCatalog("");
  if (prefilledId) {
    const existing = _activeIndicators.find(e => e.indicator_id === prefilledId);
    _showParamsForm(prefilledId, existing ? existing.params : null);
  }
}

function _closePicker() {
  _indModalEl.hidden = true;
  _pickerCurrent = null;
}

async function _saveFromPicker() {
  if (!_pickerCurrent) return;
  _activeIndicators = _activeIndicators.filter(e => e.indicator_id !== _pickerCurrent.id);
  _activeIndicators.push({ indicator_id: _pickerCurrent.id, params: _pickerCurrent.params });
  await _saveActiveIndicators();
  _renderActiveIndicatorsList();
  _closePicker();
  if (activeTicker) loadTicker(activeTicker);
}

_indAddBtn?.addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  _openPicker(null);
});
_indCancelBtn?.addEventListener("click", _closePicker);
_indSaveBtn?.addEventListener("click", _saveFromPicker);
_indParamsBackBtn?.addEventListener("click", () => {
  _indParamsEl.hidden = true;
  _indCatalogEl.hidden = false;
  _indSearchEl.hidden = false;
  _indSaveBtn.disabled = true;
  _pickerCurrent = null;
});
_indSearchEl?.addEventListener("input", () => _renderCatalog(_indSearchEl.value));
_indModalEl?.querySelector(".indicators-modal-backdrop")?.addEventListener("click", _closePicker);

// --- Boot ----------------------------------------------------------------
initDisplayControls();
applyDisplay();
refreshAlerts();
setInterval(refreshAlerts, 60 * 1000);

// Deep-link support: /?ticker=XYZ (used by the Scanner page to hand off a
// ticker to the chart). Falls back to AAPL.
function _bootTicker() {
  try {
    const q = new URLSearchParams(window.location.search).get("ticker");
    if (q) return q.toUpperCase().trim();
  } catch {}
  return "AAPL";
}

// Watchlist boots async: fetch server first (so phone edits show up here),
// then render. Once that's settled, push a fresh signal-rule (with the
// up-to-date ticker list) and load the initial chart.
(async () => {
  await Promise.all([
    fetchAndMergeWatchlist(),
    _fetchIndicatorsCatalog(),
    _fetchActiveIndicators(),
  ]);
  renderWatchlist();
  initWatchlistSortable();
  scheduleSignalRuleSync();
  _renderActiveIndicatorsList();
  const t = _bootTicker();
  input.value = t;
  loadTicker(t);
})();
