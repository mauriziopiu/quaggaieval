"use strict";

/*
 * eval_viewer — reads a comparison_manifest.json (output of
 * visualize_iterative_refinement.py) directly and computes everything
 * shown here client-side: no dependency on analyze_finetuned_metrics.py
 * or analyze_comparison_manifest.py having been run first.
 *
 * Load with: python3 -m http.server, then open this page with
 * ?data=<path-to-comparison_manifest.json> relative to this page.
 */

const STATE = {
  dataUrl: null,
  baseDir: "",
  manifest: null,
  rows: [],
  rowsByStem: {},
  sortKey: "stem",
  sortDir: "asc",
  search: "",
};

const METRICS = [
  { key: "iou", label: "IoU", get: (r) => r.iou },
  { key: "precision", label: "Precision", get: (r) => r.metrics.precision },
  { key: "recall", label: "Recall", get: (r) => r.metrics.recall },
  { key: "accuracy", label: "Accuracy", get: (r) => r.metrics.accuracy },
  { key: "f1", label: "F1", get: (r) => r.metrics.f1 },
];

const ROW_ACCESSORS = {
  stem: (r) => r.stem,
  coverage: (r) => r.coveragePct,
  iou: (r) => r.iou,
  precision: (r) => r.metrics.precision,
  recall: (r) => r.metrics.recall,
  f1: (r) => r.metrics.f1,
  accuracy: (r) => r.metrics.accuracy,
  rounds: (r) => r.numRounds,
  bestRound: (r) => r.bestRound,
  stopReason: (r) => r.stopReason || "",
};

const TABLE_COLUMNS = [
  ["stem", "Stem"],
  ["coverage", "Coverage"],
  ["iou", "IoU"],
  ["precision", "Precision"],
  ["recall", "Recall"],
  ["f1", "F1"],
  ["accuracy", "Accuracy"],
  ["rounds", "Rounds"],
  ["bestRound", "Best Round"],
  ["stopReason", "Stop Reason"],
];

// --------------------------------------------------------------- helpers

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function computeMetrics(counts) {
  const c = counts || { tp: 0, tn: 0, fp: 0, fn: 0 };
  const tp = c.tp || 0, tn = c.tn || 0, fp = c.fp || 0, fn = c.fn || 0;
  const total = tp + tn + fp + fn;
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : (fn === 0 ? 1.0 : 0.0);
  const recall    = (tp + fn) > 0 ? tp / (tp + fn) : (fp === 0 ? 1.0 : 0.0);
  const f1Denom   = 2 * tp + fp + fn;
  const f1        = f1Denom > 0 ? (2 * tp) / f1Denom : 1.0;
  const accuracy  = total > 0 ? (tp + tn) / total : 1.0;
  return {
    tp_pct: total > 0 ? (100 * tp) / total : 0,
    tn_pct: total > 0 ? (100 * tn) / total : 0,
    fp_pct: total > 0 ? (100 * fp) / total : 0,
    fn_pct: total > 0 ? (100 * fn) / total : 0,
    precision, recall, accuracy, f1,
  };
}

function computeRanking(rows, getter) {
  const sorted = [...rows].sort((a, b) => getter(a) - getter(b));
  const n = sorted.length;
  return {
    worst:  sorted[0],
    median: sorted[Math.floor((n - 1) / 2)],
    best:   sorted[n - 1],
  };
}

function baseDirFromUrl(url) {
  const idx = url.lastIndexOf("/");
  return idx >= 0 ? url.slice(0, idx + 1) : "";
}

function imgSrc(relPath) {
  if (!relPath) return null;
  return STATE.baseDir + relPath;
}

function stopReasonTag(reason) {
  if (reason === "iou_threshold_met")   return '<span class="tag good">threshold met</span>';
  if (reason === "max_points_reached")  return '<span class="tag warn">max points</span>';
  if (reason === "perfect_after_round1") return '<span class="tag good">perfect@1</span>';
  return `<span class="tag muted">${escapeHtml(reason || "—")}</span>`;
}

function pointAddedLabel(pointAdded) {
  if (!pointAdded) return "";
  const cls  = pointAdded.label === 1 ? "positive" : "negative";
  const sign = pointAdded.label === 1 ? "+" : "−";
  return `<span class="point-tag ${cls}">(${sign})</span>`;
}

// --------------------------------------------------------------- loading

function buildRows(manifest) {
  const processed = (manifest.predictions || []).filter((e) => e.status === "processed");
  const rows = processed.map((e) => {
    const bestCounts  = e.pixel_counts_finetuned;
    const bestMetrics = computeMetrics(bestCounts);
    const c = bestCounts || { tp: 0, tn: 0, fp: 0, fn: 0 };
    const total = (c.tp || 0) + (c.tn || 0) + (c.fp || 0) + (c.fn || 0);
    const coveragePct = total > 0 ? (100 * ((c.tp || 0) + (c.fn || 0))) / total : 0;
    return {
      stem: e.stem,
      entry: e,
      coveragePct,
      iou: typeof e.iou_finetuned === "number" ? e.iou_finetuned : 0,
      metrics: bestMetrics,
      numRounds: (e.rounds || []).length,
      bestRound: e.best_round ? e.best_round.round : null,
      isLastRound: e.best_round ? e.best_round.is_last_round : null,
      stopReason: e.stop_reason || null,
    };
  });
  rows.sort((a, b) => a.stem.localeCompare(b.stem));
  return rows;
}

async function loadAndRender(dataUrl) {
  const statusEl = document.getElementById("manifest-status");
  statusEl.className = "manifest-status";
  statusEl.textContent = `Loading ${dataUrl} ...`;
  document.getElementById("app").innerHTML = "";

  try {
    const res = await fetch(dataUrl, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    const manifest = await res.json();

    if (!manifest.predictions) {
      throw new Error("This doesn't look like a comparison_manifest.json (no 'predictions' key).");
    }

    STATE.dataUrl = dataUrl;
    STATE.baseDir = baseDirFromUrl(dataUrl);
    STATE.manifest = manifest;
    STATE.rows = buildRows(manifest);
    STATE.rowsByStem = {};
    STATE.rows.forEach((r) => { STATE.rowsByStem[r.stem] = r; });

    const hasRounds = manifest.predictions.some((e) => e.rounds);
    const processedCount = STATE.rows.length;
    const totalCount = manifest.predictions.length;

    statusEl.textContent = `Loaded ${processedCount} of ${totalCount} entries from ${dataUrl}`;

    if (!hasRounds) {
      document.getElementById("app").innerHTML = `
        <div class="notice">
          This manifest doesn't have per-round data (<code>rounds</code>), so it doesn't look
          like it came from <code>visualize_iterative_refinement.py</code>. This viewer is built
          around that schema (all rounds + best-round highlighting) and may not render correctly.
        </div>`;
      return;
    }

    router();
  } catch (err) {
    statusEl.className = "manifest-status error";
    statusEl.textContent =
      `Failed to load: ${err.message}. Make sure you're serving this over http:// ` +
      `(not file://) and the path is correct relative to this page.`;
    document.getElementById("app").innerHTML = '<div class="empty-state">No data loaded.</div>';
  }
}

// ---------------------------------------------------------------- router

function router() {
  const hash = location.hash.replace(/^#\/?/, "");
  if (hash.startsWith("sample/")) {
    renderDetail(decodeURIComponent(hash.slice("sample/".length)));
  } else {
    renderOverview();
  }
}

// -------------------------------------------------------- overview view

function renderOverview() {
  const app = document.getElementById("app");
  const rows = STATE.rows;

  if (!rows.length) {
    app.innerHTML = '<div class="empty-state">No processed entries in this manifest.</div>';
    return;
  }

  const macroHtml = METRICS.map((m) => {
    const vals = rows.map(m.get);
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    return `<div class="macro-card"><div class="label">${m.label}</div><div class="value">${avg.toFixed(3)}</div></div>`;
  }).join("");

  const rankingHtml = METRICS.map((m) => {
    const { worst, median, best } = computeRanking(rows, m.get);
    const statRow = (tag, r) => `
      <div class="metric-stat-row">
        <span class="stat-tag">${tag}</span>
        <span class="stat-value">${m.get(r).toFixed(3)}</span>
        <a class="stat-stem" href="#/sample/${encodeURIComponent(r.stem)}">${escapeHtml(r.stem)}</a>
      </div>`;
    return `
      <div class="metric-group">
        <div class="metric-name">${m.label}</div>
        ${statRow("Worst", worst)}
        ${statRow("Median", median)}
        ${statRow("Best", best)}
      </div>`;
  }).join("");

  app.innerHTML = `
    <div class="summary-section">
      <div class="section-title">Overall &mdash; macro average (best round per image)</div>
      <div class="macro-grid">${macroHtml}</div>
      <div class="metric-groups">${rankingHtml}</div>
    </div>
    <div class="summary-section">
      <div class="section-title">All Samples</div>
      <div class="table-controls">
        <input type="search" id="search-input" placeholder="Search stem..." value="${escapeHtml(STATE.search)}" />
        <span class="count" id="row-count"></span>
      </div>
      <div class="table-wrap">
        <table class="results" id="results-table">
          <thead><tr>${tableHeaderHtml()}</tr></thead>
          <tbody id="results-tbody"></tbody>
        </table>
      </div>
    </div>
  `;

  document.getElementById("search-input").addEventListener("input", (e) => {
    STATE.search = e.target.value;
    renderTableBody();
  });

  document.querySelectorAll("#results-table thead th").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (STATE.sortKey === key) {
        STATE.sortDir = STATE.sortDir === "asc" ? "desc" : "asc";
      } else {
        STATE.sortKey = key;
        STATE.sortDir = key === "stem" ? "asc" : "desc";
      }
      updateHeaderSortedClasses();
      renderTableBody();
    });
  });

  renderTableBody();
}

function tableHeaderHtml() {
  return TABLE_COLUMNS.map(([key, label]) =>
    `<th data-key="${key}" class="${key === STATE.sortKey ? "sorted" : ""}">${label}</th>`
  ).join("");
}

function updateHeaderSortedClasses() {
  document.querySelectorAll("#results-table thead th").forEach((th) => {
    th.classList.toggle("sorted", th.dataset.key === STATE.sortKey);
  });
}

function getSortedFilteredRows() {
  const term = STATE.search.trim().toLowerCase();
  let rows = STATE.rows;
  if (term) rows = rows.filter((r) => r.stem.toLowerCase().includes(term));

  const getter = ROW_ACCESSORS[STATE.sortKey];
  rows = [...rows].sort((a, b) => {
    const av = getter(a), bv = getter(b);
    if (av < bv) return -1;
    if (av > bv) return 1;
    return 0;
  });
  if (STATE.sortDir === "desc") rows.reverse();
  return rows;
}

function renderTableBody() {
  const rows = getSortedFilteredRows();
  document.getElementById("row-count").textContent = `${rows.length} of ${STATE.rows.length}`;

  const tbody = document.getElementById("results-tbody");
  tbody.innerHTML = rows.map((r) => `
    <tr data-stem="${escapeHtml(r.stem)}">
      <td class="stem-cell">${escapeHtml(r.stem)}</td>
      <td>${r.coveragePct.toFixed(1)}%</td>
      <td>${r.iou.toFixed(3)}</td>
      <td>${r.metrics.precision.toFixed(3)}</td>
      <td>${r.metrics.recall.toFixed(3)}</td>
      <td>${r.metrics.f1.toFixed(3)}</td>
      <td>${r.metrics.accuracy.toFixed(3)}</td>
      <td>${r.numRounds}</td>
      <td>${r.bestRound}${r.isLastRound === false ? ' <span class="tag warn">overshoot</span>' : ""}</td>
      <td>${stopReasonTag(r.stopReason)}</td>
    </tr>
  `).join("");

  tbody.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => {
      location.hash = "#/sample/" + encodeURIComponent(tr.dataset.stem);
    });
  });
}

// ---------------------------------------------------------- detail view

function imgTag(src) {
  if (!src) return '<div class="img-missing">not available</div>';
  return `<img src="${escapeHtml(src)}" loading="lazy" onerror="this.replaceWith(Object.assign(document.createElement('div'), {className:'img-missing', textContent:'failed to load'}))" />`;
}

function imageCard(title, src, opts) {
  opts = opts || {};
  const body = src ? imgTag(src) : `<div class="img-missing">${escapeHtml(opts.missingText || "not available")}</div>`;
  return `
    <div class="image-card">
      <div class="card-title"><span>${escapeHtml(title)}</span></div>
      ${body}
      ${opts.statsHtml || ""}
    </div>`;
}

function statsBoxHtml(iou, metrics) {
  const i = typeof iou === "number" ? iou : 0;
  return `
    <div class="stats-box">
      <div class="stats-grid">
        <div class="stat"><div class="k">TP</div><div class="v">${metrics.tp_pct.toFixed(1)}%</div></div>
        <div class="stat"><div class="k">TN</div><div class="v">${metrics.tn_pct.toFixed(1)}%</div></div>
        <div class="stat"><div class="k">FP</div><div class="v">${metrics.fp_pct.toFixed(1)}%</div></div>
        <div class="stat"><div class="k">FN</div><div class="v">${metrics.fn_pct.toFixed(1)}%</div></div>
      </div>
      <div class="stats-metrics">
        <span class="m"><span class="k">IoU</span> <span class="v">${i.toFixed(3)}</span></span>
        <span class="m"><span class="k">Prec</span> <span class="v">${metrics.precision.toFixed(3)}</span></span>
        <span class="m"><span class="k">Rec</span> <span class="v">${metrics.recall.toFixed(3)}</span></span>
        <span class="m"><span class="k">Acc</span> <span class="v">${metrics.accuracy.toFixed(3)}</span></span>
        <span class="m"><span class="k">F1</span> <span class="v">${metrics.f1.toFixed(3)}</span></span>
      </div>
    </div>`;
}

function renderDetail(stem) {
  const app = document.getElementById("app");
  const row = STATE.rowsByStem[stem];

  if (!row) {
    app.innerHTML = `<div class="empty-state">Sample "${escapeHtml(stem)}" not found in this manifest. <a href="#/">Back to overview</a></div>`;
    return;
  }

  const e = row.entry;
  const paths = e.paths || {};
  const rounds = e.rounds || [];
  const bestRoundNum = e.best_round ? e.best_round.round : null;
  const baselineMetrics = computeMetrics(e.pixel_counts_baseline);

  const idx  = STATE.rows.findIndex((r) => r.stem === stem);
  const prev = idx > 0 ? STATE.rows[idx - 1] : null;
  const next = idx < STATE.rows.length - 1 ? STATE.rows[idx + 1] : null;

  const refCardsHtml = `
    <div class="card-grid">
      ${imageCard("Raw Image", imgSrc(paths.raw_image), { missingText: "Raw image not available — run add_raw_images.py" })}
      ${imageCard("Ground Truth", imgSrc(paths.ground_truth))}
      ${imageCard("Baseline", imgSrc(paths.model_baseline), { statsHtml: statsBoxHtml(e.iou_baseline, baselineMetrics) })}
    </div>`;

  const roundCardsHtml = rounds.map((r) => {
    const isBest  = r.round === bestRoundNum;
    const metrics = computeMetrics(r.pixel_counts_finetuned);
    return `
      <div class="image-card ${isBest ? "best" : ""}">
        <div class="card-title">
          <span>Round ${r.round} — ${r.num_points} point${r.num_points > 1 ? "s" : ""} ${pointAddedLabel(r.point_added)}</span>
          ${isBest ? '<span class="best-badge">BEST</span>' : ""}
        </div>
        <div class="image-pair">
          <div><div class="pair-label">Prediction</div>${imgTag(imgSrc(r.path_model))}</div>
          <div><div class="pair-label">Diff</div>${imgTag(imgSrc(r.path_diff))}</div>
        </div>
        ${statsBoxHtml(r.iou_finetuned, metrics)}
      </div>`;
  }).join("");

  app.innerHTML = `
    <div class="detail-header">
      <div>
        <a class="back-link" href="#/">&larr; Back to overview</a>
        <div class="detail-title">${escapeHtml(stem)}</div>
        <div class="detail-meta">
          <span>Coverage: <b>${row.coveragePct.toFixed(1)}%</b></span>
          <span>Rounds: <b>${row.numRounds}</b></span>
          <span>Best round: <b>${row.bestRound}</b>${row.isLastRound === false ? ' <span class="tag warn">overshoot</span>' : ""}</span>
          <span>Stop reason: ${stopReasonTag(row.stopReason)}</span>
        </div>
      </div>
      <div class="nav-buttons">
        <a href="${prev ? "#/sample/" + encodeURIComponent(prev.stem) : "#"}" class="${prev ? "" : "disabled"}">&larr; Prev</a>
        <a href="${next ? "#/sample/" + encodeURIComponent(next.stem) : "#"}" class="${next ? "" : "disabled"}">Next &rarr;</a>
      </div>
    </div>

    <div class="section-title">Reference</div>
    ${refCardsHtml}

    <div class="section-title">Rounds</div>
    <div class="rounds-legend">
      <span><span class="legend-dot" style="background:#3c78dc"></span>TP</span>
      <span><span class="legend-dot" style="background:#ff8c00"></span>FP</span>
      <span><span class="legend-dot" style="background:#f0dc28"></span>FN</span>
      <span>Points: <span class="point-tag positive">+</span> positive &nbsp; <span class="point-tag negative">−</span> negative</span>
    </div>
    <div class="card-grid">${roundCardsHtml}</div>
  `;

  window.scrollTo(0, 0);
}

// ------------------------------------------------------------------ init

function init() {
  const params = new URLSearchParams(location.search);
  const dataUrl = params.get("data");
  const input = document.getElementById("manifest-input");
  if (dataUrl) input.value = dataUrl;

  document.getElementById("manifest-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const url = input.value.trim();
    if (!url) return;
    const newParams = new URLSearchParams(location.search);
    newParams.set("data", url);
    history.replaceState(null, "", `${location.pathname}?${newParams.toString()}${location.hash}`);
    loadAndRender(url);
  });

  window.addEventListener("hashchange", router);

  if (dataUrl) {
    loadAndRender(dataUrl);
  } else {
    document.getElementById("app").innerHTML = `
      <div class="notice">
        No manifest loaded. Enter a path to a <code>comparison_manifest.json</code> above
        (relative to this page, e.g.
        <code>../../output/output_vis_v4/comparison_manifest.json</code>), or append
        <code>?data=...</code> to the URL.
      </div>`;
  }
}

document.addEventListener("DOMContentLoaded", init);
