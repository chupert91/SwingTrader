// Alert Rules UI: list, create, edit, toggle, delete.
// Talks to /api/rules (Vercel) or /api/rules (local FastAPI — same path).

(function () {
  const rulesList = document.getElementById("rules-list");
  const statusEl = document.getElementById("rules-status");
  const addDefaultBtn = document.getElementById("rules-add-default");
  const addBlankBtn = document.getElementById("rules-add-blank");

  if (!rulesList) return; // page may not include the rules section

  const OPTIMIZED_DEFAULT = {
    id: "",
    name: "σ + trend + stoch confluence (both sides)",
    tickers: ["TSLA", "NVDA", "PLTR", "MP"],
    side: "both",
    entry_sigma: -1.0,
    require_trend: false,
    min_trend_pct: 0.0,
    exit_target_pct: 20.0,
    exit_stop_pct: 10.0,
    leverage: 5.0,
    enabled: true,
    notify_email: "",
    trend_direction: "any",
    require_stoch_extreme: true,
    stoch_oversold: 35.0,
    stoch_overbought: 65.0,
  };

  const BLANK = {
    id: "",
    name: "Custom rule",
    tickers: ["TSLA"],
    side: "both",
    entry_sigma: -1.0,
    require_trend: false,
    min_trend_pct: 0.0,
    exit_target_pct: 20.0,
    exit_stop_pct: 10.0,
    leverage: 5.0,
    enabled: true,
    notify_email: "",
    trend_direction: "any",
    require_stoch_extreme: true,
    stoch_oversold: 35.0,
    stoch_overbought: 65.0,
  };

  function setStatus(msg, isError) {
    statusEl.textContent = msg || "";
    statusEl.classList.toggle("error", !!isError);
  }

  async function fetchRules() {
    const resp = await fetch("/api/rules");
    if (!resp.ok) throw new Error(`GET /api/rules → ${resp.status}`);
    const data = await resp.json();
    return data.rules || [];
  }

  async function saveRule(rule) {
    const resp = await fetch("/api/rules", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(rule),
    });
    if (!resp.ok) throw new Error(`POST /api/rules → ${resp.status}`);
    return await resp.json();
  }

  async function deleteRule(id) {
    const resp = await fetch(`/api/rules?id=${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (!resp.ok) throw new Error(`DELETE /api/rules → ${resp.status}`);
    return await resp.json();
  }

  function uuid() {
    return crypto.randomUUID ? crypto.randomUUID() :
      `r-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }

  function renderRule(rule) {
    const li = document.createElement("li");
    li.className = "rule-item" + (rule.enabled ? "" : " disabled");
    li.dataset.id = rule.id;

    const sigSide = rule.side === "short" ? `+${Math.abs(rule.entry_sigma).toFixed(1)}σ`
                  : rule.side === "both"  ? `±${Math.abs(rule.entry_sigma).toFixed(1)}σ`
                  : `${rule.entry_sigma.toFixed(1)}σ`;
    const trendStr = rule.require_trend ? `, trend≥${rule.min_trend_pct}%/yr` : "";
    const tickerStr = (rule.tickers || []).join(", ") || "(no tickers)";

    li.innerHTML = `
      <div class="rule-row1">
        <label class="rule-toggle" title="Enable / disable this rule">
          <input type="checkbox" data-action="toggle" ${rule.enabled ? "checked" : ""} />
        </label>
        <input class="rule-name" data-action="rename" value="${escapeHtml(rule.name)}" />
        <button class="rule-delete" data-action="delete" title="Delete rule" aria-label="Delete rule"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>
      </div>
      <div class="rule-row2">
        <span class="rule-side rule-side-${rule.side}">${rule.side}</span>
        <span class="rule-sig">${sigSide}${trendStr}</span>
        <button class="rule-edit" data-action="edit">edit</button>
      </div>
      <div class="rule-row3">${escapeHtml(tickerStr)}</div>
      <div class="rule-edit-panel" data-edit-panel hidden>
        <label>Tickers (comma-separated)
          <input type="text" data-field="tickers" value="${escapeHtml((rule.tickers||[]).join(","))}" />
        </label>
        <label>Side
          <select data-field="side">
            <option value="long" ${rule.side==="long"?"selected":""}>Long only</option>
            <option value="short" ${rule.side==="short"?"selected":""}>Short only</option>
            <option value="both" ${rule.side==="both"?"selected":""}>Both</option>
          </select>
        </label>
        <label>Entry σ <input type="number" step="0.1" data-field="entry_sigma" value="${rule.entry_sigma}" /></label>
        <label class="row-check"><input type="checkbox" data-field="require_trend" ${rule.require_trend?"checked":""} /> Require trend alignment</label>
        <label>Min trend %/yr <input type="number" step="1" data-field="min_trend_pct" value="${rule.min_trend_pct}" /></label>
        <label>Slope direction
          <select data-field="trend_direction">
            <option value="any" ${(rule.trend_direction||"any")==="any"?"selected":""}>Any</option>
            <option value="up" ${rule.trend_direction==="up"?"selected":""}>Up only (slope &gt; 0)</option>
            <option value="down" ${rule.trend_direction==="down"?"selected":""}>Down only (slope &lt; 0)</option>
          </select>
        </label>
        <label class="row-check"><input type="checkbox" data-field="require_stoch_extreme" ${rule.require_stoch_extreme?"checked":""} /> Require stoch RSI extreme</label>
        <label>Oversold (long) <input type="number" step="1" data-field="stoch_oversold" value="${rule.stoch_oversold ?? 25}" /></label>
        <label>Overbought (short) <input type="number" step="1" data-field="stoch_overbought" value="${rule.stoch_overbought ?? 75}" /></label>
        <label>Exit target % (option) <input type="number" step="1" data-field="exit_target_pct" value="${rule.exit_target_pct}" /></label>
        <label>Exit stop % (option) <input type="number" step="1" data-field="exit_stop_pct" value="${rule.exit_stop_pct}" /></label>
        <label>Leverage <input type="number" step="0.5" data-field="leverage" value="${rule.leverage}" /></label>
        <label>Notify email <input type="email" data-field="notify_email" value="${escapeHtml(rule.notify_email||"")}" /></label>
        <div class="rule-edit-buttons">
          <button data-action="save">Save</button>
          <button data-action="cancel">Cancel</button>
        </div>
      </div>
    `;

    li.addEventListener("click", (e) => onRuleClick(e, rule));
    li.addEventListener("change", (e) => onRuleChange(e, rule));
    return li;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"
    }[c]));
  }

  async function onRuleChange(e, rule) {
    const action = e.target.dataset.action;
    if (action === "toggle") {
      rule.enabled = e.target.checked;
      try {
        await saveRule(rule);
        e.target.closest(".rule-item").classList.toggle("disabled", !rule.enabled);
        setStatus(`${rule.name}: ${rule.enabled?"enabled":"disabled"}`);
      } catch (err) {
        setStatus("Save failed: " + err.message, true);
        e.target.checked = !rule.enabled; // revert
        rule.enabled = !rule.enabled;
      }
    } else if (action === "rename") {
      rule.name = e.target.value || "Unnamed";
      try { await saveRule(rule); setStatus("Renamed."); }
      catch (err) { setStatus("Save failed: " + err.message, true); }
    }
  }

  async function onRuleClick(e, rule) {
    const action = e.target.dataset.action;
    const li = e.target.closest(".rule-item");
    if (!action) return;

    if (action === "delete") {
      if (!confirm(`Delete rule "${rule.name}"?`)) return;
      try {
        await deleteRule(rule.id);
        li.remove();
        setStatus("Deleted.");
      } catch (err) { setStatus("Delete failed: " + err.message, true); }
    } else if (action === "edit") {
      const panel = li.querySelector("[data-edit-panel]");
      panel.hidden = !panel.hidden;
    } else if (action === "cancel") {
      li.querySelector("[data-edit-panel]").hidden = true;
    } else if (action === "save") {
      const panel = li.querySelector("[data-edit-panel]");
      const updated = {
        ...rule,
        tickers: panel.querySelector('[data-field="tickers"]').value
                  .split(",").map(s => s.trim().toUpperCase()).filter(Boolean),
        side: panel.querySelector('[data-field="side"]').value,
        entry_sigma: parseFloat(panel.querySelector('[data-field="entry_sigma"]').value) || -2.0,
        require_trend: panel.querySelector('[data-field="require_trend"]').checked,
        min_trend_pct: parseFloat(panel.querySelector('[data-field="min_trend_pct"]').value) || 0,
        exit_target_pct: parseFloat(panel.querySelector('[data-field="exit_target_pct"]').value) || 20,
        exit_stop_pct: parseFloat(panel.querySelector('[data-field="exit_stop_pct"]').value) || 10,
        leverage: parseFloat(panel.querySelector('[data-field="leverage"]').value) || 5,
        notify_email: panel.querySelector('[data-field="notify_email"]').value.trim(),
        trend_direction: panel.querySelector('[data-field="trend_direction"]').value,
        require_stoch_extreme: panel.querySelector('[data-field="require_stoch_extreme"]').checked,
        stoch_oversold: parseFloat(panel.querySelector('[data-field="stoch_oversold"]').value) || 25,
        stoch_overbought: parseFloat(panel.querySelector('[data-field="stoch_overbought"]').value) || 75,
      };
      try {
        await saveRule(updated);
        setStatus("Saved.");
        load();  // re-render to pick up changes
      } catch (err) { setStatus("Save failed: " + err.message, true); }
    }
  }

  async function load() {
    try {
      const rules = await fetchRules();
      rulesList.innerHTML = "";
      if (rules.length === 0) {
        const li = document.createElement("li");
        li.className = "rules-empty";
        li.textContent = "No rules yet. Use the buttons above to create one.";
        rulesList.appendChild(li);
        return;
      }
      for (const r of rules) rulesList.appendChild(renderRule(r));
      setStatus("");
    } catch (err) {
      setStatus("Load failed: " + err.message, true);
    }
  }

  addDefaultBtn?.addEventListener("click", async () => {
    const rule = { ...OPTIMIZED_DEFAULT, id: uuid() };
    try { await saveRule(rule); setStatus("Created optimized rule."); load(); }
    catch (err) { setStatus("Create failed: " + err.message, true); }
  });

  addBlankBtn?.addEventListener("click", async () => {
    const rule = { ...BLANK, id: uuid() };
    try { await saveRule(rule); setStatus("Created blank rule."); load(); }
    catch (err) { setStatus("Create failed: " + err.message, true); }
  });

  load();
})();
