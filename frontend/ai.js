// AI Trader page. Reads /api/ai/state and renders status, settings, open
// positions, an equity / cumulative-realized chart, closed trades, and the
// raw cron run log. Writes go through /api/ai/settings and /api/ai/run.

const REFRESH_MS = 30_000;

const el = (id) => document.getElementById(id);

function fmtUsd(v, signed = false) {
  if (v == null || Number.isNaN(v)) return "—";
  const s = (signed && v > 0 ? "+" : "") +
    v.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
  return s;
}
function fmtPct(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return `${v > 0 ? "+" : ""}${Number(v).toFixed(1)}%`;
}
function pnlClass(v) {
  if (v == null || Number.isNaN(v) || v === 0) return "";
  return v > 0 ? "ai-pos" : "ai-neg";
}
function shortTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function tierBadge(t) {
  // BOUNCE = chart-eye 5-feature pattern promoted a WEAK breadth candidate.
  // See backend/ai_strategy._bounce_features + research/bounce_confirm_sweep.txt.
  const cls = { PRIME: "t-prime", OK: "t-ok", PANIC: "t-panic",
                BOUNCE: "t-bounce", WEAK: "t-weak" }[t] || "t-weak";
  const title = t === "BOUNCE"
    ? "Bounce-confirm: in band + reclaim>=0.20s + higher-low (252d) + stoch turning + support confluence"
    : "";
  return `<span class="ai-badge ${cls}" title="${title}">${esc(t || "?")}</span>`;
}
// Signal source: which band fired (primary [-2.5,-2.0] vs deep z<=-3.5).
// Old pre-deploy trades have no source field — default to primary so the
// row never renders blank.
function sourceBadge(src) {
  const s = (src === "deep") ? "deep" : "primary";
  const cls = (s === "deep") ? "src-deep" : "src-primary";
  const label = (s === "deep") ? "DEEP" : "primary";
  return `<span class="ai-badge ${cls}" title="${s === 'deep' ? 'Deep capitulation cross (z<=-3.5)' : 'Primary band [-2.5,-2.0]'}">${label}</span>`;
}
function reasonBadge(r) {
  const map = {
    take_profit: ["Take profit", "t-prime"],
    time_stop: ["Time stop", "t-ok"],
    disaster_stop: ["Disaster stop", "t-panic"],
    entry_unfilled: ["Unfilled", "t-weak"],
    external: ["External", "t-weak"],
  };
  const [label, cls] = map[r] || [r || "—", "t-weak"];
  return `<span class="ai-badge ${cls}">${esc(label)}</span>`;
}
function contractLabel(c) {
  if (!c) return "—";
  return `${esc(c.strike)}C ${esc(c.expiration)}`;
}

// ---- chart ---------------------------------------------------------------

let chart = null, equitySeries = null, realizedSeries = null;

function ensureChart() {
  if (chart) return;
  const host = el("ai-pnl-chart");
  chart = LightweightCharts.createChart(host, {
    autoSize: true,
    layout: { background: { type: "solid", color: "#161b22" }, textColor: "#8b949e", fontSize: 11 },
    grid: { vertLines: { color: "#21262d" }, horzLines: { color: "#21262d" } },
    rightPriceScale: { borderColor: "#21262d" },
    leftPriceScale: { borderColor: "#21262d", visible: true },
    timeScale: { borderColor: "#21262d", timeVisible: true },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });
  equitySeries = chart.addLineSeries({
    color: "#58a6ff", lineWidth: 2, priceScaleId: "right",
    priceLineVisible: false, title: "Equity",
  });
  realizedSeries = chart.addLineSeries({
    color: "#26a69a", lineWidth: 2, priceScaleId: "left",
    priceLineVisible: false, title: "Cum. realized",
  });
}

function renderChart(series) {
  const host = el("ai-pnl-chart");
  if (!series || series.length === 0) {
    el("ai-pnl-empty").hidden = false;
    host.style.display = "none";
    return;
  }
  el("ai-pnl-empty").hidden = true;
  host.style.display = "block";
  ensureChart();
  // LWC needs strictly-ascending unique times; dedupe by epoch-second.
  const byT = new Map();
  for (const p of series) {
    const sec = Math.floor(new Date(p.t).getTime() / 1000);
    if (!Number.isNaN(sec)) byT.set(sec, p);
  }
  const pts = [...byT.entries()].sort((a, b) => a[0] - b[0]);
  equitySeries.setData(pts.map(([sec, p]) => ({ time: sec, value: Number(p.equity) || 0 })));
  realizedSeries.setData(pts.map(([sec, p]) => ({ time: sec, value: Number(p.realized_cum) || 0 })));
  chart.timeScale().fitContent();
}

// ---- render --------------------------------------------------------------

let settingsCache = {};

function renderState(d) {
  // Kill switch + note
  const enabled = !!d.settings.enabled;
  el("ai-enabled").checked = enabled;
  el("ai-enabled-note").textContent = enabled
    ? "Bot is ON — entries evaluated once per trading day during market hours. Existing positions managed every run."
    : "Bot is OFF — no new entries. Existing positions are still managed.";
  el("ai-config-warn").hidden = !!d.configured;

  // Settings inputs — null/undefined renders as blank (the "disabled" sentinel
  // for take_profit_pct / disaster_stop_pct / disaster_stop_usd / sigma_target).
  settingsCache = d.settings || {};
  for (const inp of document.querySelectorAll("#ai-settings [data-set]")) {
    const k = inp.dataset.set;
    if (k in settingsCache) {
      const v = settingsCache[k];
      if (inp.type === "checkbox") inp.checked = !!v;
      else inp.value = (v === null || v === undefined) ? "" : v;
    }
  }

  // Tiles
  const a = d.account || {};
  const lastRun = (d.run_log && d.run_log[0]) || null;
  const tiles = [
    ["Equity", a.equity != null ? fmtUsd(Number(a.equity)) : "—"],
    ["Options BP", a.options_buying_power != null ? fmtUsd(Number(a.options_buying_power)) : "—"],
    ["Cash", a.cash != null ? fmtUsd(Number(a.cash)) : "—"],
    ["Open", String(d.open_trades.length)],
    ["Closed", String(d.stats.closed_count ?? 0)],
    ["Total realized", `<span class="${pnlClass(d.stats.total_realized)}">${fmtUsd(d.stats.total_realized, true)}</span>`],
    ["Last run", lastRun ? `${shortTime(lastRun.t)}<br><span class="ai-sub">${lastRun.market_open ? "market open" : "market closed"}</span>` : "—"],
  ];
  el("ai-tiles").innerHTML = tiles.map(([k, v]) =>
    `<div class="ai-tile"><div class="ai-tile-k">${k}</div><div class="ai-tile-v">${v}</div></div>`).join("");

  // Open positions
  const ob = el("ai-open-table").querySelector("tbody");
  ob.innerHTML = "";
  el("ai-open-empty").hidden = d.open_trades.length > 0;
  for (const t of d.open_trades) {
    const m = t.mark || {};
    const entry = t.entry || {};
    const upc = m.unrealized_plpc != null ? m.unrealized_plpc * 100 : null;
    const held = t.trading_days_held != null
      ? `${t.trading_days_held}/${settingsCache.time_stop_days}d` : "—";
    const tpStop = `${entry.fill_price ? (t.tp?.limit_price ?? "—") : "—"} / ${t.disaster_stop?.stop_price ?? "—"}`;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${esc(t.ticker)}</td>
      <td>${tierBadge(t.signal?.tier)} ${sourceBadge(t.signal?.source)}</td>
      <td>${contractLabel(t.contract)}</td>
      <td><span class="ai-badge t-weak">${esc(t.status)}</span></td>
      <td>${entry.fill_price != null ? "$" + entry.fill_price : (entry.limit_price != null ? "lmt $" + entry.limit_price : "—")}</td>
      <td>${m.current_price != null ? "$" + Number(m.current_price).toFixed(2) : "—"}</td>
      <td class="${pnlClass(upc)}">${upc != null ? fmtPct(upc) : "—"}</td>
      <td>${held}</td>
      <td class="ai-sub">${tpStop}</td>`;
    ob.appendChild(tr);
  }

  // Stats line
  const s = d.stats;
  el("ai-stats").innerHTML = s.closed_count
    ? `<span>Win rate <b>${s.win_rate ?? "—"}%</b></span>
       <span>Avg win <b class="ai-pos">${fmtUsd(s.avg_win, true)}</b></span>
       <span>Avg loss <b class="ai-neg">${fmtUsd(s.avg_loss, true)}</b></span>
       <span>Total <b class="${pnlClass(s.total_realized)}">${fmtUsd(s.total_realized, true)}</b></span>`
    : "";

  // Closed trades
  const cb = el("ai-closed-table").querySelector("tbody");
  cb.innerHTML = "";
  el("ai-closed-empty").hidden = d.closed_trades.length > 0;
  for (const t of d.closed_trades) {
    const entry = t.entry || {}, exit = t.exit || {}, pnl = t.pnl || {};
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${esc(t.ticker)}</td>
      <td>${tierBadge(t.signal?.tier)} ${sourceBadge(t.signal?.source)}</td>
      <td>${contractLabel(t.contract)}</td>
      <td>${entry.fill_price != null ? "$" + entry.fill_price : "—"}</td>
      <td>${exit.fill_price != null ? "$" + exit.fill_price : "—"}</td>
      <td class="${pnlClass(pnl.realized_usd)}">${pnl.realized_usd != null ? fmtUsd(pnl.realized_usd, true) : "—"}</td>
      <td class="${pnlClass(pnl.realized_pct)}">${pnl.realized_pct != null ? fmtPct(pnl.realized_pct) : "—"}</td>
      <td>${reasonBadge(exit.reason)}</td>
      <td class="ai-sub">${shortTime(exit.filled_at || t.updated_at)}</td>`;
    cb.appendChild(tr);
  }

  renderChart(d.equity);

  // Run log
  el("ai-runlog").innerHTML = (d.run_log || []).map((r) => {
    const lines = [];
    if (r.reason) lines.push(`<div class="ai-rl-skip">${esc(r.reason)}</div>`);
    for (const ac of r.actions || []) lines.push(`<div class="ai-rl-act">${esc(ac)}</div>`);
    for (const sk of r.skips || []) lines.push(`<div class="ai-rl-skip">skip: ${esc(sk)}</div>`);
    for (const er of r.errors || []) lines.push(`<div class="ai-rl-err">error: ${esc(er)}</div>`);
    if (!lines.length) lines.push(`<div class="ai-rl-skip">no actions</div>`);
    return `<div class="ai-rl">
      <div class="ai-rl-head">
        <span class="ai-rl-time">${shortTime(r.t)}</span>
        <span class="ai-badge ${r.market_open ? "t-prime" : "t-weak"}">${r.market_open ? "open" : "closed"}</span>
        <span class="ai-sub">${esc(r.phase || "")}${r.entries_placed != null ? ` · ${r.entries_placed} placed` : ""}</span>
      </div>${lines.join("")}</div>`;
  }).join("") || `<div class="ai-empty">No runs logged yet.</div>`;
}

// ---- io ------------------------------------------------------------------

async function load() {
  try {
    const r = await fetch("/api/ai/state");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    renderState(await r.json());
  } catch (e) {
    el("ai-runlog").innerHTML = `<div class="ai-rl-err">Failed to load state: ${esc(e.message)}</div>`;
  }
}

async function postSettings(patch) {
  const r = await fetch("/api/ai/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

el("ai-enabled").addEventListener("change", async (e) => {
  try {
    await postSettings({ enabled: e.target.checked });
    await load();
  } catch (err) {
    el("ai-save-status").textContent = `Failed: ${err.message}`;
    e.target.checked = !e.target.checked;
  }
});

el("ai-save-settings").addEventListener("click", async () => {
  // Blank numeric input → send explicit null so the backend can DISABLE
  // a previously-set field (e.g. unset take_profit_pct to enable the
  // sigma-target exit path). The list of fields that accept null is
  // mirrored in DEFAULT_SETTINGS (take_profit_pct, sigma_target,
  // disaster_stop_pct, disaster_stop_usd, time_stop_days).
  const NULLABLE = new Set([
    "take_profit_pct", "sigma_target",
    "disaster_stop_pct", "disaster_stop_usd",
    "time_stop_days",
  ]);
  const patch = {};
  for (const inp of document.querySelectorAll("#ai-settings [data-set]")) {
    const k = inp.dataset.set;
    if (inp.type === "checkbox") patch[k] = inp.checked;
    else if (inp.tagName === "SELECT") patch[k] = inp.value;
    else if (inp.value === "") {
      if (NULLABLE.has(k)) patch[k] = null;
    } else {
      patch[k] = Number(inp.value);
    }
  }
  const status = el("ai-save-status");
  status.textContent = "Saving…";
  try {
    await postSettings(patch);
    status.textContent = "Saved.";
    await load();
  } catch (err) {
    status.textContent = `Failed: ${err.message}`;
  }
});

el("ai-run-now").addEventListener("click", async () => {
  const btn = el("ai-run-now");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const r = await fetch("/api/ai/run", { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    await load();
  } catch (err) {
    el("ai-save-status").textContent = `Run failed: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run now";
  }
});

load();
setInterval(() => {
  if (document.visibilityState === "visible") load();
}, REFRESH_MS);
