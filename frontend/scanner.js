const ICON_PLUS = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>`;

const listEl = document.getElementById("scanner-list");
const metaEl = document.getElementById("scanner-meta");
const refreshBtn = document.getElementById("scanner-refresh");

async function loadResults() {
  try {
    const resp = await fetch("/api/discoveries");
    if (!resp.ok) {
      metaEl.textContent = `Failed to load: HTTP ${resp.status}`;
      return;
    }
    render(await resp.json());
  } catch (err) {
    metaEl.textContent = `Failed to load: ${err.message}`;
  }
}

function render(data) {
  listEl.innerHTML = "";
  const matches = (data && data.matches) || [];
  if (matches.length === 0) {
    metaEl.textContent = data && data.as_of_date
      ? `As of ${data.as_of_date}: no matches. Loosen Signal or wait for next scan.`
      : "No scan results yet. Hit refresh to run one now.";
    return;
  }
  metaEl.textContent = `As of ${data.as_of_date} · ${matches.length} match${matches.length === 1 ? "" : "es"} (of ${data.universe_size || "?"} scanned)`;
  for (const m of matches) {
    const li = document.createElement("li");
    li.className = "discovery-item";
    const sd = m.sd_position?.toFixed(2) ?? "—";
    const slope = m.slope_annual_pct != null
      ? `${m.slope_annual_pct >= 0 ? "+" : ""}${m.slope_annual_pct.toFixed(0)}%`
      : "—";
    const k = m.stoch_rsi_k != null ? m.stoch_rsi_k.toFixed(0) : "—";
    const vol = m.avg_dollar_volume_m != null
      ? `${m.avg_dollar_volume_m >= 1000 ? (m.avg_dollar_volume_m / 1000).toFixed(1) + "B" : m.avg_dollar_volume_m.toFixed(0) + "M"}`
      : "—";
    li.innerHTML = `
      <span class="disc-ticker">${m.ticker}</span>
      <span class="disc-dir ${m.direction}">${m.direction}</span>
      <button class="disc-add" type="button" title="Add to watchlist" aria-label="Add to watchlist">${ICON_PLUS}</button>
      <span class="disc-stats">σ ${sd} · K ${k} · slope ${slope} · $${(m.current_price ?? 0).toFixed(2)} · vol $${vol}</span>
    `;
    li.querySelector(".disc-add").addEventListener("click", async (e) => {
      e.stopPropagation();
      const btn = e.currentTarget;
      btn.disabled = true;
      const ok = await addToWatchlist(m.ticker);
      if (ok) li.remove();
      else btn.disabled = false;
    });
    li.addEventListener("click", (e) => {
      if (e.target.closest(".disc-add")) return;
      window.location.href = `/?ticker=${encodeURIComponent(m.ticker)}`;
    });
    listEl.appendChild(li);
  }
}

// Server is the source of truth for the watchlist; the chart page boots by
// fetching it from /api/watchlist, so we just GET-modify-PUT here.
async function addToWatchlist(rawTicker) {
  const ticker = String(rawTicker || "").toUpperCase().trim();
  if (!ticker) return false;
  try {
    const resp = await fetch("/api/watchlist");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const tickers = Array.isArray(data.tickers) ? data.tickers : [];
    if (!tickers.includes(ticker)) tickers.push(ticker);
    const put = await fetch("/api/watchlist", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tickers }),
    });
    return put.ok;
  } catch { return false; }
}

refreshBtn.addEventListener("click", async () => {
  refreshBtn.classList.add("scanning");
  metaEl.textContent = "Scanning S&P 500… this can take 30-60s";
  try {
    const resp = await fetch("/api/discover", { method: "POST" });
    if (resp.ok) render(await resp.json());
    else metaEl.textContent = `Scan failed: HTTP ${resp.status}`;
  } catch (err) {
    metaEl.textContent = `Scan failed: ${err.message}`;
  } finally {
    refreshBtn.classList.remove("scanning");
  }
});

loadResults();
