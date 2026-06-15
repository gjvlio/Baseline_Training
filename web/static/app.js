"use strict";

// Processing steps — faithful to ACE-Net paper sections 3.2 / 3.3 / 3.4
const STEPS = [
  "Log-Mel spectrogram extraction (80 bands, 16 kHz)",   // §3.2 MDCNN input
  "MDCNN acoustic encoding → acoustic features",          // §3.2.1
  "BERT textual encoding + cross-attention → Z_at",       // §3.2.2, Eq.5-9
  "Facial keyframe selection (Top-K = 8)",                // §3.3.1
  "FVLiteNet visual encoding → Z_v",                      // §3.3.2, Eq.12-13
  "Multi-aspect fusion + MLP discriminator → P(fake)",    // §3.4, Eq.14-17
];

// ── State machine ──
// Pages:  /           → upload
//         /scanning   → analyzing
//         /results    → results
// Data flows via sessionStorage so page refreshes survive.

let selectedFile = null;
let stepTimer    = null;
let currentStep  = 0;

// ── DOM helpers ──
function get(id) { return document.getElementById(id); }

function show(stateId) {
  ["state-idle", "state-selected", "state-analyzing", "state-results", "state-error"]
    .forEach(id => { const el = get(id); if (el) el.hidden = el.id !== stateId; });
}

// ── Route → screen mapping ──
function route() {
  const path = location.pathname;
  if (path === "/scanning") {
    renderScanning();
  } else if (path === "/results") {
    renderResults(JSON.parse(sessionStorage.getItem("acenet_result") || "null"));
  } else {
    show("state-idle");
  }
}

window.addEventListener("popstate", route);
document.addEventListener("DOMContentLoaded", route);

function nav(path) {
  history.pushState({}, "", path);
  route();
}

// ── Upload screen ──
const $fi = get("file-input");

function setupUpload() {
  const $idle = get("state-idle");
  const $sel  = get("state-selected");
  if (!$idle) return;

  $idle.addEventListener("click",    () => $fi.click());
  $idle.addEventListener("dragover", e => { e.preventDefault(); $idle.classList.add("drag-over"); });
  $idle.addEventListener("dragleave",    () => $idle.classList.remove("drag-over"));
  $idle.addEventListener("drop", e => {
    e.preventDefault(); $idle.classList.remove("drag-over");
    if (e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]);
  });
  $fi.addEventListener("change", () => { if ($fi.files[0]) pick($fi.files[0]); });

  if ($sel) {
    $sel.addEventListener("click", e => { if (e.target.id !== "run-btn") $fi.click(); });
  }

  const runBtn = get("run-btn");
  if (runBtn) runBtn.addEventListener("click", e => { e.stopPropagation(); startAnalysis(); });

  const resetBtn = get("reset-btn");
  if (resetBtn) resetBtn.addEventListener("click", () => nav("/"));

  const retryBtn = get("retry-btn");
  if (retryBtn) retryBtn.addEventListener("click", () => nav("/"));
}
setupUpload();

function pick(f) {
  selectedFile = f;
  const fi = get("file-info");
  if (fi) fi.textContent = `${f.name}  ·  ${(f.size / 1e6).toFixed(1)} MB`;
  show("state-selected");
}

// ── Start analysis — fires POST then navigates to /scanning ──
async function startAnalysis() {
  if (!selectedFile) return;

  sessionStorage.setItem("acenet_filename", selectedFile.name);
  sessionStorage.removeItem("acenet_result");
  sessionStorage.removeItem("acenet_error");

  const fd = new FormData();
  fd.append("file", selectedFile);

  // Must be assigned BEFORE nav() — renderScanning() reads it synchronously
  window._aceNetFetch = fetch("/analyze", { method: "POST", body: fd });

  nav("/scanning");
}

// ── Scanning screen ──
function renderScanning() {
  show("state-analyzing");

  const sub = get("analyzing-sub");
  if (sub) sub.textContent = sessionStorage.getItem("acenet_filename") || "";

  // Build step list
  const ol = get("steps-list");
  if (ol) {
    ol.innerHTML = STEPS.map((label, i) =>
      `<li id="s${i}"><span class="step-icon" id="si${i}"></span>${label}</li>`
    ).join("");
  }

  currentStep = 0;
  if (stepTimer) clearInterval(stepTimer);

  // Mark step 0 immediately so user sees instant response
  markStep(0); currentStep = 1;

  // Advance one step every 1.5 s until result arrives or all done
  stepTimer = setInterval(() => {
    if (currentStep < STEPS.length) {
      markStep(currentStep);
      currentStep++;
    } else {
      clearInterval(stepTimer);
      stepTimer = null;
    }
  }, 1500);

  // Wait for the fetch started in startAnalysis()
  const p = window._aceNetFetch;
  if (!p) { nav("/"); return; }   // user landed here directly via URL — go home

  p.then(async resp => {
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(e.detail || `Server error ${resp.status}`);
    }
    return resp.json();
  })
  .then(data => {
    // Snap remaining steps quickly (200 ms each), then navigate
    clearInterval(stepTimer); stepTimer = null;
    const snapRemaining = () => {
      if (currentStep < STEPS.length) {
        markStep(currentStep); currentStep++;
        setTimeout(snapRemaining, 200);
      } else {
        sessionStorage.setItem("acenet_result", JSON.stringify(data));
        setTimeout(() => nav("/results"), 350);
      }
    };
    snapRemaining();
  })
  .catch(err => {
    clearInterval(stepTimer); stepTimer = null;
    show("state-error");
    history.replaceState({}, "", "/");
    const em = get("error-msg");
    if (em) em.textContent = String(err).replace(/^Error:\s*/, "");
  });
}

function markStep(i) {
  const li = get(`s${i}`); if (!li || li.classList.contains("done")) return;
  li.classList.add("done");
  const ic = get(`si${i}`); if (ic) ic.textContent = "✓";
}

// ── Results screen ──
function renderResults(data) {
  if (!data) { nav("/"); return; }
  show("state-results");

  const isGenuine = data.verdict === "GENUINE";
  const pct       = Math.round(data.fake_prob * 100);

  const vc = get("verdict-card");
  if (vc) vc.className = `verdict ${isGenuine ? "genuine" : "fake"}`;

  setText("verdict-badge",  isGenuine ? "Genuine" : "Fake");
  setText("fake-pct",       `${pct}%`);
  setText("verdict-label",  isGenuine ? "Likely authentic" : "Likely deepfake");
  setText("verdict-desc",   isGenuine
    ? "No significant inconsistency detected between audio-text and visual emotion embeddings."
    : "Significant mismatch detected between audio-text and visual emotion embeddings.");

  // Score bar
  const fill = get("score-fill");
  if (fill) {
    fill.style.width = `${pct}%`;
    fill.style.backgroundPosition = `${pct}% 0`;
  }
  setText("score-pct-label", `${pct}%`);
  const tag = get("score-tag");
  if (tag) { tag.textContent = isGenuine ? "Genuine" : "Fake"; tag.className = `score-tag ${isGenuine ? "genuine" : "fake"}`; }

  // L2 discrepancy
  setText("disc-value", data.discrepancy_l2 != null ? data.discrepancy_l2.toFixed(3) : "—");

  // Model stats
  const ms = data.model_stats || {};
  const sg = get("stat-grid");
  if (sg) {
    sg.innerHTML = [
      ["AUC",  fmt(ms.auc)],
      ["F1",   fmt(ms.f1)],
      ["ACC",  pctFmt(ms.accuracy)],
      ["Prec", fmt(ms.precision)],
      ["Rec",  fmt(ms.recall)],
    ].map(([k, v]) =>
      `<div class="stat-item"><span class="stat-val">${v}</span><span class="stat-key">${k}</span></div>`
    ).join("");
  }

  setText("transcript-text", data.transcript || "[not available]");

  const rb = get("reset-btn");
  if (rb) rb.onclick = () => { sessionStorage.clear(); nav("/"); };
}

function setText(id, val) { const el = get(id); if (el) el.textContent = val; }
function fmt(v)    { return v != null ? v.toFixed(3) : "—"; }
function pctFmt(v) { return v != null ? `${(v * 100).toFixed(1)}%` : "—"; }
