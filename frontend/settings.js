// Settings page: edits the canonical signal config + notify email and
// pushes the synced AlertRule to /api/rules on every change. The chart
// page reads the same localStorage keys, so changes propagate when the
// user navigates back.

const SIGNAL_KEY = "swingtrader.signal";
const NOTIFY_EMAIL_KEY = "swingtrader.notifyEmail";

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
    return { ...SIGNAL_DEFAULTS, ...parsed };
  } catch { return { ...SIGNAL_DEFAULTS }; }
}
function saveSignal(s) {
  localStorage.setItem(SIGNAL_KEY, JSON.stringify({ ...s, _v: 1 }));
}
function loadNotifyEmail() { return localStorage.getItem(NOTIFY_EMAIL_KEY) || ""; }
function saveNotifyEmail(v) { localStorage.setItem(NOTIFY_EMAIL_KEY, v || ""); }

let signal = loadSignal();
let notifyEmail = loadNotifyEmail();

const statusEl = document.getElementById("set-status");
let _statusTimer = null;
function showStatus(text, kind = "ok") {
  statusEl.hidden = false;
  statusEl.textContent = text;
  statusEl.dataset.kind = kind;
  if (_statusTimer) clearTimeout(_statusTimer);
  if (kind === "ok") {
    _statusTimer = setTimeout(() => { statusEl.hidden = true; }, 1500);
  }
}

function bindForm() {
  for (const el of document.querySelectorAll("[data-set]")) {
    const key = el.dataset.set;

    if (key === "notify_email") {
      el.value = notifyEmail;
      el.addEventListener("input", () => {
        notifyEmail = el.value.trim();
        saveNotifyEmail(notifyEmail);
        scheduleSync();
      });
      continue;
    }

    const val = signal[key];
    if (el.type === "checkbox") {
      el.checked = !!val;
    } else {
      el.value = val == null ? "" : val;
    }

    el.addEventListener("change", () => {
      if (el.type === "checkbox") {
        signal[key] = el.checked;
      } else if (el.tagName === "SELECT") {
        signal[key] = el.value;
      } else if (el.value === "") {
        signal[key] = null;
      } else {
        signal[key] = Number(el.value);
      }
      saveSignal(signal);
      scheduleSync();
    });
  }
}

async function loadUniverse() {
  const el = document.getElementById("set-universe-list");
  try {
    const resp = await fetch("/api/watchlist");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const tickers = data.tickers || [];
    if (tickers.length === 0) {
      el.innerHTML = '<span class="bt-hint">(no tickers — add some on the chart page)</span>';
      return;
    }
    el.innerHTML = tickers
      .map(t => `<span class="set-ticker-chip">${escapeHtml(t)}</span>`)
      .join("");
  } catch (err) {
    el.innerHTML = `<span class="bt-hint">Failed to load watchlist: ${escapeHtml(err.message)}</span>`;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

let _syncTimer = null;
function scheduleSync() {
  if (_syncTimer) clearTimeout(_syncTimer);
  _syncTimer = setTimeout(pushRule, 400);
}

async function pushRule() {
  showStatus("Saving…", "ok");
  try {
    const watchResp = await fetch("/api/watchlist");
    if (!watchResp.ok) throw new Error(`watchlist HTTP ${watchResp.status}`);
    const watchData = await watchResp.json();
    const tickers = watchData.tickers || [];

    const thr = Math.abs(Number(signal.sigma_threshold) || 1.0);
    const rule = {
      id: "ui-signal",
      name: "Signal alerts (auto-synced)",
      tickers,
      side: "both",
      entry_sigma: -thr,
      require_trend: !!signal.require_trend_alignment,
      min_trend_pct: 0.0,
      exit_target_pct: 20.0,
      exit_stop_pct: 10.0,
      leverage: 5.0,
      enabled: true,
      notify_email: notifyEmail,
      trend_direction: signal.trend_direction || "any",
      require_stoch_extreme: !!signal.require_stoch_extreme,
      stoch_oversold: Number(signal.stoch_oversold) || 35,
      stoch_overbought: Number(signal.stoch_overbought) || 65,
      min_avg_volume_m: Number(signal.min_avg_volume_m) || 0,
    };
    const r = await fetch("/api/rules", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(rule),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    showStatus("Saved", "ok");
  } catch (err) {
    showStatus(`Save failed: ${err.message}`, "error");
  }
}

bindForm();
loadUniverse();
