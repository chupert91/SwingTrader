// Options payoff visualizer. Fetches an expiration list + the call/put chain
// for one expiration from the Alpaca-backed /api/options endpoints, renders a
// clickable two-sided chain table, and draws a P/L vs. stock-price diagram for
// the picked single long option. All pricing math runs here (client-side) so
// the time slider can redraw without a round-trip.

const R = 0.043;        // risk-free rate assumption (annualized)
const MULT = 100;       // shares per option contract
const DAY_MS = 86400000;

const el = (id) => document.getElementById(id);

// ---- Black-Scholes (European) -------------------------------------------

// Standard normal CDF — Abramowitz & Stegun 7.1.26, accurate to ~7 digits.
function normCdf(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp(-x * x / 2);
  const p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 +
    t * (-1.821255978 + t * 1.330274429))));
  return x >= 0 ? 1 - p : p;
}

// Theoretical option value. At/after expiry (t<=0) or zero vol it collapses
// to intrinsic value, which is exactly the expiration payoff curve.
function bsPrice(S, K, t, vol, r, isCall) {
  if (t <= 0 || vol <= 0) {
    return isCall ? Math.max(0, S - K) : Math.max(0, K - S);
  }
  const sqrtT = Math.sqrt(t);
  const d1 = (Math.log(S / K) + (r + vol * vol / 2) * t) / (vol * sqrtT);
  const d2 = d1 - vol * sqrtT;
  return isCall
    ? S * normCdf(d1) - K * Math.exp(-r * t) * normCdf(d2)
    : K * Math.exp(-r * t) * normCdf(-d2) - S * normCdf(-d1);
}

// Back the implied vol out of a market price by bisection. Returns null when
// the price sits outside the no-arbitrage band (can't be matched by any vol).
function impliedVol(price, S, K, t, r, isCall) {
  if (!(price > 0) || t <= 0) return null;
  let lo = 1e-4, hi = 5.0;
  if (price <= bsPrice(S, K, t, lo, r, isCall)) return null;
  if (price >= bsPrice(S, K, t, hi, r, isCall)) return null;
  for (let i = 0; i < 64; i++) {
    const mid = (lo + hi) / 2;
    if (bsPrice(S, K, t, mid, r, isCall) < price) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

// ---- formatting ----------------------------------------------------------

function fmtUsd(v, signed = false) {
  if (v == null || Number.isNaN(v)) return "—";
  const sign = signed && v > 0 ? "+" : "";
  return sign + v.toLocaleString("en-US", {
    style: "currency", currency: "USD", maximumFractionDigits: 0,
  });
}
function fmtPrice(v) {
  if (v == null || Number.isNaN(v)) return "—";
  const dp = Math.abs(v) >= 100 ? 0 : 2;
  return "$" + v.toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp });
}
// Compact axis label: -$1.2k, +$340.
function fmtPLshort(v) {
  if (v == null || Number.isNaN(v)) return "—";
  const sign = v < 0 ? "-" : "";
  const a = Math.abs(v);
  if (a >= 1000) return `${sign}$${(a / 1000).toFixed(a >= 10000 ? 0 : 1)}k`;
  return `${sign}$${a.toFixed(0)}`;
}
function fmtPct(v, signed = false) {
  if (v == null || Number.isNaN(v)) return "—";
  return `${signed && v > 0 ? "+" : ""}${v.toFixed(1)}%`;
}
function plClass(v) {
  if (v == null || Number.isNaN(v) || v === 0) return "";
  return v > 0 ? "ai-pos" : "ai-neg";
}

// "nice" axis ticks spanning [min,max].
function niceTicks(min, max, count) {
  if (!(max > min)) return [min];
  const raw = (max - min) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-6; t += step) {
    ticks.push(t);
  }
  return ticks;
}

// ---- payoff chart --------------------------------------------------------

const COL = {
  bg: "#0e1117", grid: "#21262d", text: "#8b949e", bright: "#e6edf3",
  now: "#58a6ff", exp: "#e67e22", spot: "rgba(230,237,243,0.5)",
  be: "#f1c40f", zero: "rgba(230,237,243,0.4)",
  profit: "rgba(38,166,154,0.16)", loss: "rgba(239,83,80,0.15)",
};
const PAD = { l: 66, r: 18, t: 16, b: 38 };

class PayoffChart {
  constructor(host) {
    this.host = host;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "opt-canvas";
    host.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.model = null;
    this.daysElapsed = 0;
    this.mouse = null;
    this.canvas.addEventListener("mousemove", (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
      this.render();
    });
    this.canvas.addEventListener("mouseleave", () => {
      this.mouse = null;
      this.render();
    });
    new ResizeObserver(() => this.render()).observe(host);
  }

  setModel(model) { this.model = model; this.daysElapsed = 0; this.render(); }
  setTime(days) { this.daysElapsed = days; this.render(); }

  // P/L of the position at underlying price S, with `tYears` left to expiry.
  plAt(S, tYears) {
    const m = this.model;
    const val = bsPrice(S, m.strike, tYears, m.iv, R, m.isCall);
    return (val - m.premium) * MULT * m.qty;
  }

  render() {
    const m = this.model;
    const cssW = this.host.clientWidth;
    const cssH = this.host.clientHeight;
    if (cssW < 2 || cssH < 2) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = cssW * dpr;
    this.canvas.height = cssH * dpr;
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = COL.bg;
    ctx.fillRect(0, 0, cssW, cssH);
    if (!m) return;

    const plotW = cssW - PAD.l - PAD.r;
    const plotH = cssH - PAD.t - PAD.b;
    if (plotW < 10 || plotH < 10) return;

    // ---- domains --------------------------------------------------------
    const be = m.isCall ? m.strike + m.premium : m.strike - m.premium;
    const anchors = [m.spot, m.strike, be].filter((v) => v > 0);
    const xLo = Math.max(0.01, Math.min(...anchors) * 0.72);
    const xHi = Math.max(...anchors) * 1.3;

    const tRem = Math.max(0, (m.dte - this.daysElapsed) / 365);
    const N = 220;
    const curve = [];
    let yLo = 0, yHi = 0;
    for (let i = 0; i <= N; i++) {
      const S = xLo + (xHi - xLo) * (i / N);
      const expPL = this.plAt(S, 0);
      const nowPL = this.plAt(S, tRem);
      curve.push({ S, expPL, nowPL });
      yLo = Math.min(yLo, expPL, nowPL);
      yHi = Math.max(yHi, expPL, nowPL);
    }
    const yPad = (yHi - yLo) * 0.08 || 1;
    yLo -= yPad; yHi += yPad;

    const sx = (S) => PAD.l + ((S - xLo) / (xHi - xLo)) * plotW;
    const sy = (pl) => PAD.t + (1 - (pl - yLo) / (yHi - yLo)) * plotH;
    const zeroY = sy(0);

    // ---- grid + axes ----------------------------------------------------
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    ctx.lineWidth = 1;
    ctx.strokeStyle = COL.grid;
    ctx.fillStyle = COL.text;

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const pl of niceTicks(yLo, yHi, 6)) {
      const y = sy(pl);
      if (y < PAD.t - 1 || y > PAD.t + plotH + 1) continue;
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + plotW, y); ctx.stroke();
      ctx.fillText(fmtPLshort(pl), PAD.l - 8, y);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const px of niceTicks(xLo, xHi, 7)) {
      const x = sx(px);
      if (x < PAD.l - 1 || x > PAD.l + plotW + 1) continue;
      ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + plotH); ctx.stroke();
      ctx.fillText(fmtPrice(px), x, PAD.t + plotH + 8);
    }

    // zero P/L line
    ctx.strokeStyle = COL.zero;
    ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(PAD.l, zeroY); ctx.lineTo(PAD.l + plotW, zeroY); ctx.stroke();

    // ---- profit / loss fill under the slider-date curve ----------------
    const fillNow = () => {
      ctx.beginPath();
      ctx.moveTo(sx(curve[0].S), zeroY);
      for (const p of curve) ctx.lineTo(sx(p.S), sy(p.nowPL));
      ctx.lineTo(sx(curve[curve.length - 1].S), zeroY);
      ctx.closePath();
    };
    const clipBand = (top, bottom) => {
      ctx.beginPath();
      ctx.rect(PAD.l, top, plotW, Math.max(0, bottom - top));
      ctx.clip();
    };
    ctx.save(); clipBand(PAD.t, zeroY);
    ctx.fillStyle = COL.profit; fillNow(); ctx.fill(); ctx.restore();
    ctx.save(); clipBand(zeroY, PAD.t + plotH);
    ctx.fillStyle = COL.loss; fillNow(); ctx.fill(); ctx.restore();

    // ---- curves ---------------------------------------------------------
    const stroke = (key, color, dash) => {
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash(dash || []);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      curve.forEach((p, i) => {
        const x = sx(p.S), y = sy(p[key]);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    };
    stroke("expPL", COL.exp, [5, 4]);
    stroke("nowPL", COL.now, null);

    // ---- spot + breakeven markers ---------------------------------------
    const vline = (price, color, label, labelY) => {
      if (price < xLo || price > xHi) return;
      const x = sx(price);
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + plotH); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(label, x, labelY);
      ctx.restore();
    };
    vline(m.spot, COL.spot, "spot " + fmtPrice(m.spot), PAD.t + 2);
    if (be > 0) {
      ctx.save();
      ctx.fillStyle = COL.be;
      ctx.beginPath(); ctx.arc(sx(be), zeroY, 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText("B/E " + fmtPrice(be), sx(be), zeroY - 7);
      ctx.restore();
    }

    // ---- crosshair tooltip ---------------------------------------------
    if (this.mouse && this.mouse.x >= PAD.l && this.mouse.x <= PAD.l + plotW) {
      const S = xLo + ((this.mouse.x - PAD.l) / plotW) * (xHi - xLo);
      const idx = Math.round((S - xLo) / (xHi - xLo) * N);
      const p = curve[Math.max(0, Math.min(N, idx))];
      const x = sx(p.S);
      ctx.save();
      ctx.setLineDash([2, 3]);
      ctx.strokeStyle = "rgba(230,237,243,0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + plotH); ctx.stroke();
      ctx.setLineDash([]);
      for (const [key, color] of [["expPL", COL.exp], ["nowPL", COL.now]]) {
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(x, sy(p[key]), 3.5, 0, Math.PI * 2); ctx.fill();
      }
      this._tooltip(ctx, x, p, cssW);
      ctx.restore();
    }
  }

  _tooltip(ctx, x, p, cssW) {
    const lines = [
      ["Price", fmtPrice(p.S)],
      ["At slider", fmtUsd(p.nowPL, true), p.nowPL],
      ["At expiry", fmtUsd(p.expPL, true), p.expPL],
    ];
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    let w = 0;
    for (const [k, v] of lines) w = Math.max(w, ctx.measureText(`${k}   ${v}`).width);
    const boxW = w + 20, boxH = lines.length * 16 + 10;
    let bx = x + 14;
    if (bx + boxW > cssW - 4) bx = x - 14 - boxW;
    const by = PAD.t + 8;
    ctx.fillStyle = "rgba(22,27,34,0.96)";
    ctx.strokeStyle = COL.grid;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(bx, by, boxW, boxH, 5); ctx.fill(); ctx.stroke();
    ctx.textBaseline = "middle";
    lines.forEach(([k, v, val], i) => {
      const y = by + 13 + i * 16;
      ctx.textAlign = "left";
      ctx.fillStyle = COL.text;
      ctx.fillText(k, bx + 10, y);
      ctx.textAlign = "right";
      ctx.fillStyle = val == null ? COL.bright : (val >= 0 ? "#26a69a" : "#ef5350");
      ctx.fillText(v, bx + boxW - 10, y);
    });
  }
}

// ---- state + data --------------------------------------------------------

const state = {
  ticker: "", name: "", spot: null, asof: null,
  expirations: [], expiration: "", dte: 0,
  rows: [],            // [{strike, call: contract|null, put: contract|null}]
  type: "call", selectedStrike: null, contract: null,
  qty: 1, iv: null, ivSource: "",
};

let chart = null;

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function setMeta(text, isError = false) {
  const m = el("opt-meta");
  m.textContent = text;
  m.classList.toggle("opt-meta-err", isError);
}

async function loadChain(ticker) {
  ticker = ticker.trim().toUpperCase();
  if (!ticker) return;
  setMeta(`Loading ${ticker} option chain…`);
  el("opt-controls").hidden = true;
  el("opt-model-panel").hidden = true;
  try {
    const data = await fetchJson(`/api/options/expirations/${encodeURIComponent(ticker)}`);
    if (!data.expirations.length) {
      setMeta(`No option expirations found for ${ticker}.`, true);
      return;
    }
    state.ticker = data.ticker;
    state.name = data.name || "";
    state.spot = data.spot;
    state.asof = data.asof;
    state.expirations = data.expirations;

    const sel = el("opt-expiration");
    sel.innerHTML = "";
    for (const e of data.expirations) {
      const o = document.createElement("option");
      o.value = e.date;
      o.textContent = `${e.date}  ·  ${e.dte}d`;
      sel.appendChild(o);
    }
    // Default to the expiration closest to 60 DTE (the swing-trade horizon).
    const best = data.expirations.reduce((a, b) =>
      Math.abs(b.dte - 60) < Math.abs(a.dte - 60) ? b : a);
    sel.value = best.date;

    el("opt-controls").hidden = false;
    await loadExpiration();
  } catch (err) {
    setMeta(`Could not load ${ticker}: ${err.message}`, true);
  }
}

async function loadExpiration() {
  const expiration = el("opt-expiration").value;
  const enc = encodeURIComponent(state.ticker);
  setMeta(`Loading ${state.ticker} ${expiration} chain…`);
  try {
    const [calls, puts] = await Promise.all([
      fetchJson(`/api/options/chain/${enc}?expiration=${expiration}&type=call`),
      fetchJson(`/api/options/chain/${enc}?expiration=${expiration}&type=put`),
    ]);
    state.spot = calls.spot;
    state.expiration = calls.expiration;
    state.dte = calls.dte;

    // Merge call + put legs by strike. Only strikes with a usable mid on at
    // least one side make it into the table.
    const byStrike = new Map();
    const add = (contract, side) => {
      if (!(contract.mid > 0)) return;
      let row = byStrike.get(contract.strike);
      if (!row) { row = { strike: contract.strike, call: null, put: null }; byStrike.set(contract.strike, row); }
      row[side] = contract;
    };
    calls.contracts.forEach((c) => add(c, "call"));
    puts.contracts.forEach((p) => add(p, "put"));
    state.rows = [...byStrike.values()].sort((a, b) => a.strike - b.strike);

    if (!state.rows.length) {
      setMeta(`${state.ticker} ${expiration}: no quoted strikes.`, true);
      el("opt-model-panel").hidden = true;
      return;
    }
    el("opt-model-panel").hidden = false;
    renderChain();

    // Default: at-the-money strike, call side (put if that strike has no call).
    const atm = state.rows.reduce((a, b) =>
      Math.abs(b.strike - state.spot) < Math.abs(a.strike - state.spot) ? b : a);
    selectContract(atm.strike, atm.call ? "call" : "put");
    scrollChainToSelected();
  } catch (err) {
    setMeta(`Could not load chain: ${err.message}`, true);
    el("opt-model-panel").hidden = true;
  }
}

function renderChain() {
  const fmtD = (d) => (d == null ? "—" : d.toFixed(2));
  // Strike nearest spot — gets a divider so the eye finds the money fast.
  const nearStrike = state.rows.reduce((a, b) =>
    Math.abs(b.strike - state.spot) < Math.abs(a.strike - state.spot) ? b : a).strike;

  const side = (row, type) => {
    const c = row[type];
    const itm = type === "call" ? row.strike < state.spot : row.strike > state.spot;
    const cls = `opt-side opt-side-${type}${itm ? " opt-itm" : ""}`;
    if (!c) return `<span class="${cls} opt-empty">—</span>`;
    const title = `bid ${fmtPrice(c.bid)} / ask ${fmtPrice(c.ask)} · IV ${fmtPct((c.iv || 0) * 100)}`;
    const cells = type === "call"
      ? `<span class="opt-d">${fmtD(c.delta)}</span><span class="opt-p">${fmtPrice(c.mid)}</span>`
      : `<span class="opt-p">${fmtPrice(c.mid)}</span><span class="opt-d">${fmtD(c.delta)}</span>`;
    return `<button type="button" class="${cls}" data-strike="${row.strike}" ` +
      `data-type="${type}" title="${title}">${cells}</button>`;
  };

  el("opt-chain").innerHTML = state.rows.map((row) => {
    const moneyPct = ((row.strike - state.spot) / state.spot) * 100;
    const near = row.strike === nearStrike ? " opt-row-near" : "";
    return `<div class="opt-row${near}">
      ${side(row, "call")}
      <span class="opt-strike-cell">${fmtPrice(row.strike)}<i>${fmtPct(moneyPct, true)}</i></span>
      ${side(row, "put")}
    </div>`;
  }).join("");
  markSelected();
}

function markSelected() {
  for (const btn of el("opt-chain").querySelectorAll("button.opt-side")) {
    const on = parseFloat(btn.dataset.strike) === state.selectedStrike &&
               btn.dataset.type === state.type;
    btn.classList.toggle("opt-sel", on);
  }
}

function scrollChainToSelected() {
  const sel = el("opt-chain").querySelector("button.opt-sel");
  if (!sel) return;
  const chain = el("opt-chain");
  chain.scrollTop = sel.offsetTop - chain.clientHeight / 2 + sel.offsetHeight / 2;
}

function selectContract(strike, type) {
  const row = state.rows.find((r) => r.strike === strike);
  if (!row || !row[type]) return;
  state.type = type;
  state.selectedStrike = strike;
  state.contract = row[type];
  state.qty = Math.max(1, parseInt(el("opt-qty").value, 10) || 1);
  markSelected();
  buildModel();
}

function buildModel() {
  const contract = state.contract;
  const isCall = state.type === "call";
  const strike = contract.strike;
  const tFull = state.dte / 365;
  // Prefer IV solved from the current mid so the slider-date curve passes
  // cleanly through zero P/L at the current price. Fall back to the feed's
  // own IV, then a generic guess, if the mark can't be inverted.
  let iv = impliedVol(contract.mid, state.spot, strike, tFull, R, isCall);
  let ivSource = "mark";
  if (iv == null && contract.iv) { iv = contract.iv; ivSource = "feed"; }
  if (iv == null) { iv = 0.6; ivSource = "default"; }
  state.iv = iv;
  state.ivSource = ivSource;

  el("opt-model-panel").hidden = false;
  const slider = el("opt-time");
  slider.max = String(Math.max(1, state.dte));
  slider.value = "0";

  setMeta(metaText());
  el("opt-contract-label").textContent =
    `${state.ticker} ${fmtPrice(strike)} ${isCall ? "Call" : "Put"} · ${state.expiration}`;
  chart.setModel({
    isCall, strike, dte: state.dte, spot: state.spot,
    premium: contract.mid, iv, qty: state.qty,
  });
  updateTime();
}

function metaText() {
  const parts = [state.ticker];
  if (state.name) parts.push(state.name);
  parts.push(`spot ${fmtPrice(state.spot)}`);
  if (state.asof) {
    const d = new Date(state.asof);
    if (!Number.isNaN(d.getTime())) {
      parts.push(`quotes ${d.toLocaleString([], {
        month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
      })}`);
    }
  }
  return parts.join("  ·  ");
}

function updateTime() {
  if (!state.contract) return;
  const elapsed = parseInt(el("opt-time").value, 10) || 0;
  chart.setTime(elapsed);

  const tRem = Math.max(0, state.dte - elapsed);
  const sliderDate = new Date(Date.now() + elapsed * DAY_MS);
  const dateStr = sliderDate.toLocaleDateString([], { month: "short", day: "numeric" });
  el("opt-time-label").textContent = tRem <= 0
    ? `Expiration · ${state.expiration} · 0 days left`
    : `${dateStr} · T+${elapsed}d · ${tRem} days to expiry`;

  renderTiles(elapsed, tRem);
}

function renderTiles(elapsed, tRem) {
  const m = state;
  const isCall = m.type === "call";
  const premTotal = m.contract.mid * MULT * m.qty;
  const be = isCall ? m.contract.strike + m.contract.mid
                    : m.contract.strike - m.contract.mid;
  const bePct = ((be - m.spot) / m.spot) * 100;
  // P/L if the underlying is unchanged at the slider date — pure theta view.
  const flatVal = bsPrice(m.spot, m.contract.strike, tRem / 365, m.iv, R, isCall);
  const flatPL = (flatVal - m.contract.mid) * MULT * m.qty;

  const tiles = [
    ["Premium / contract", fmtPrice(m.contract.mid),
      `${fmtUsd(premTotal)} for ${m.qty} contract${m.qty > 1 ? "s" : ""}`, ""],
    ["Implied vol", fmtPct(m.iv * 100), `from ${m.ivSource}`, ""],
    ["Delta", m.contract.delta != null ? m.contract.delta.toFixed(3) : "—",
      "per share, feed", ""],
    ["Breakeven (expiry)", fmtPrice(be), `${fmtPct(bePct, true)} vs spot`, ""],
    ["Max loss", fmtUsd(-premTotal, true), "premium paid", "ai-neg"],
    [`P/L at T+${elapsed}d if flat`, fmtUsd(flatPL, true),
      "underlying unchanged", plClass(flatPL)],
  ];
  el("opt-tiles").innerHTML = tiles.map(([k, v, sub, cls]) => `
    <div class="ai-tile">
      <div class="ai-tile-k">${k}</div>
      <div class="ai-tile-v ${cls}">${v}</div>
      <div class="opt-tile-sub">${sub}</div>
    </div>`).join("");
}

// ---- wiring --------------------------------------------------------------

function init() {
  chart = new PayoffChart(el("opt-chart"));

  el("opt-ticker-form").addEventListener("submit", (e) => {
    e.preventDefault();
    loadChain(el("opt-ticker").value);
  });
  el("opt-expiration").addEventListener("change", loadExpiration);
  el("opt-qty").addEventListener("change", () => {
    if (state.contract) selectContract(state.selectedStrike, state.type);
  });
  el("opt-time").addEventListener("input", updateTime);
  el("opt-chain").addEventListener("click", (e) => {
    const btn = e.target.closest("button.opt-side");
    if (!btn) return;
    selectContract(parseFloat(btn.dataset.strike), btn.dataset.type);
  });

  // Surface the missing-credentials case up front rather than on first fetch.
  fetch("/api/options/expirations/SPY")
    .then((r) => { if (r.status === 503) el("opt-config-warn").hidden = false; })
    .catch(() => {});

  loadChain(el("opt-ticker").value);
}

document.addEventListener("DOMContentLoaded", init);
