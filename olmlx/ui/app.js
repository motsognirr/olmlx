"use strict";

// Minimal operator dashboard for olmlx (#373). No framework, no build step:
// polls the existing JSON + Prometheus endpoints and renders plain DOM.

const POLL_MS = 3000;

const $ = (id) => document.getElementById(id);

function fmtBytes(n) {
  if (!n || n <= 0) return "–";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function fmtExpires(iso) {
  if (!iso) return "–";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "–";
  const secs = Math.round((t - Date.now()) / 1000);
  if (secs <= 0) return "expired";
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.round(secs / 60)}m`;
  return `${Math.round(secs / 3600)}h`;
}

function setConn(ok) {
  const el = $("conn");
  el.textContent = ok ? "connected" : "disconnected";
  el.className = `badge ${ok ? "ok" : "down"}`;
}

// Parse Prometheus text exposition into {name: [{labels, value}]}.
function parseMetrics(text) {
  const out = {};
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const sp = line.lastIndexOf(" ");
    if (sp < 0) continue;
    const left = line.slice(0, sp);
    const value = parseFloat(line.slice(sp + 1));
    if (Number.isNaN(value)) continue;
    const br = left.indexOf("{");
    const name = br < 0 ? left : left.slice(0, br);
    const labels = {};
    if (br >= 0) {
      const inner = left.slice(br + 1, left.lastIndexOf("}"));
      for (const m of inner.matchAll(/(\w+)="([^"]*)"/g)) labels[m[1]] = m[2];
    }
    (out[name] ||= []).push({ labels, value });
  }
  return out;
}

function sumSeries(series) {
  return (series || []).reduce((a, s) => a + s.value, 0);
}

function histogramAvg(metrics, base) {
  const sum = sumSeries(metrics[`${base}_sum`]);
  const count = sumSeries(metrics[`${base}_count`]);
  return count > 0 ? sum / count : null;
}

function renderStats(metrics, runningCount) {
  $("stat-loaded").textContent =
    runningCount != null ? runningCount : sumSeries(metrics.olmlx_loaded_models) || "0";

  const tps =
    histogramAvg(metrics, "olmlx_inference_decode_tokens_per_second") ??
    sumSeries(metrics.olmlx_inference_decode_tokens_per_second);
  $("stat-tps").textContent = tps ? tps.toFixed(1) : "–";

  const inflight = sumSeries(metrics.olmlx_http_requests_in_flight);
  $("stat-inflight").textContent = Number.isFinite(inflight) ? inflight : "–";

  const lookups = metrics.olmlx_prompt_cache_lookups_total || [];
  let hits = 0;
  let total = 0;
  for (const s of lookups) {
    total += s.value;
    const r = (s.labels.result || s.labels.outcome || "").toLowerCase();
    if (r === "hit") hits += s.value;
  }
  $("stat-cache").textContent = total > 0 ? `${Math.round((hits / total) * 100)}%` : "–";
}

async function unload(name, btn) {
  btn.disabled = true;
  try {
    await fetch("/api/unload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: name }),
    });
  } finally {
    refresh();
  }
}

async function warmup(name, btn) {
  btn.disabled = true;
  btn.textContent = "loading…";
  try {
    await fetch("/api/warmup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: name }),
    });
  } finally {
    refresh();
  }
}

function renderRunning(models) {
  const body = $("running-body");
  body.innerHTML = "";
  if (!models.length) {
    body.innerHTML = '<tr><td colspan="6" class="muted">no models loaded</td></tr>';
    return;
  }
  for (const m of models) {
    const tr = document.createElement("tr");
    const quant = m.details?.quantization_level || "–";
    tr.innerHTML =
      `<td>${m.name}</td><td>${fmtBytes(m.size)}</td><td>${quant}</td>` +
      `<td>${m.active_refs ?? 0}</td><td>${fmtExpires(m.expires_at)}</td>`;
    const td = document.createElement("td");
    const btn = document.createElement("button");
    btn.textContent = "Unload";
    btn.className = "danger";
    btn.disabled = (m.active_refs ?? 0) > 0;
    btn.onclick = () => unload(m.name, btn);
    td.appendChild(btn);
    tr.appendChild(td);
    body.appendChild(tr);
  }
}

function renderAvailable(models, runningNames) {
  const body = $("available-body");
  body.innerHTML = "";
  if (!models.length) {
    body.innerHTML = '<tr><td colspan="3" class="muted">no models installed</td></tr>';
    return;
  }
  for (const m of models) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${m.name}</td><td>${fmtBytes(m.size)}</td>`;
    const td = document.createElement("td");
    if (!runningNames.has(m.name)) {
      const btn = document.createElement("button");
      btn.textContent = "Load";
      btn.onclick = () => warmup(m.name, btn);
      td.appendChild(btn);
    } else {
      td.innerHTML = '<span class="muted">loaded</span>';
    }
    tr.appendChild(td);
    body.appendChild(tr);
  }
}

async function refresh() {
  try {
    const [psRes, tagsRes, metricsRes, verRes] = await Promise.all([
      fetch("/api/ps"),
      fetch("/api/tags"),
      fetch("/metrics"),
      fetch("/api/version"),
    ]);
    const ps = await psRes.json();
    const tags = await tagsRes.json();
    const metrics = parseMetrics(await metricsRes.text());
    const ver = await verRes.json().catch(() => ({}));

    const running = ps.models || [];
    const runningNames = new Set(running.map((m) => m.name));
    renderRunning(running);
    renderAvailable(tags.models || [], runningNames);
    renderStats(metrics, running.length);
    if (ver.version) $("version").textContent = `v${ver.version}`;
    setConn(true);
  } catch (e) {
    setConn(false);
  }
}

async function pull(model) {
  const log = $("pull-log");
  log.hidden = false;
  log.textContent = `Pulling ${model}…\n`;
  try {
    const res = await fetch("/api/pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, stream: true }),
    });
    if (!res.body) {
      log.textContent += await res.text();
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const obj = JSON.parse(line);
          log.textContent += `${obj.status || JSON.stringify(obj)}\n`;
        } catch {
          log.textContent += `${line}\n`;
        }
        log.scrollTop = log.scrollHeight;
      }
    }
    log.textContent += "done.\n";
  } catch (e) {
    log.textContent += `error: ${e}\n`;
  } finally {
    refresh();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("pull-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const v = $("pull-input").value.trim();
    if (v) pull(v);
  });
  refresh();
  setInterval(refresh, POLL_MS);
});
