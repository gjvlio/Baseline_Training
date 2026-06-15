"use strict";

const STEPS = [
  { label: "Extracting audio waveform" },
  { label: "Running MDCNN + BERT encoder → Z_at" },
  { label: "Extracting video keyframes (Top-8)" },
  { label: "Running FVLiteNet encoder → Z_v" },
  { label: "Computing consistency discriminator" },
  { label: "Bilinear fusion → P(fake)" },
];
const STEP_MS = [900, 3500, 5200, 7000, 8800, 10200];

let selectedFile = null;
let stepTimers   = [];
let allDone      = false;
let pending      = null;

// DOM
const $idle  = get("state-idle");
const $sel   = get("state-selected");
const $ana   = get("state-analyzing");
const $res   = get("state-results");
const $err   = get("state-error");
const $fi    = get("file-input");

function get(id) { return document.getElementById(id); }

function show(stateId) {
  [$idle, $sel, $ana, $res, $err].forEach(el => { el.hidden = el.id !== stateId; });
}

// ── File selection ──
$idle.addEventListener("click",     () => $fi.click());
$sel.addEventListener("click",  e => { if (e.target.id !== "run-btn") $fi.click(); });
$idle.addEventListener("dragover",  e => { e.preventDefault(); $idle.classList.add("drag-over"); });
$idle.addEventListener("dragleave",     () => $idle.classList.remove("drag-over"));
$idle.addEventListener("drop", e => {
  e.preventDefault(); $idle.classList.remove("drag-over");
  const f = e.dataTransfer.files[0]; if (f) pick(f);
});
$fi.addEventListener("change", () => { if ($fi.files[0]) pick($fi.files[0]); });

function pick(f) {
  selectedFile = f;
  get("file-info").textContent = `${f.name}  ·  ${(f.size / 1e6).toFixed(1)} MB`;
  show("state-selected");
}

get("run-btn").addEventListener("click",  e => { e.stopPropagation(); start(); });
get("reset-btn").addEventListener("click",  reset);
get("retry-btn").addEventListener("click",  reset);

function reset() {
  selectedFile = null; $fi.value = "";
  stepTimers.forEach(clearTimeout); stepTimers = [];
  allDone = false; pending = null;
  show("state-idle");
}

// ── Analysis ──
function buildSteps() {
  get("steps-list").innerHTML = STEPS.map((s, i) =>
    `<li id="s${i}"><span class="step-icon" id="si${i}"></span>${s.label}</li>`
  ).join("");
}

function mark(i) {
  const li = get(`s${i}`); if (!li || li.classList.contains("done")) return;
  li.classList.add("done"); get(`si${i}`).textContent = "✓";
}
function markAll() { STEPS.forEach((_, i) => mark(i)); }

async function start() {
  if (!selectedFile) return;
  get("analyzing-sub").textContent = selectedFile.name;
  buildSteps();
  show("state-analyzing");
  allDone = false; pending = null;

  stepTimers = STEP_MS.map((ms, i) =>
    setTimeout(() => {
      mark(i);
      if (i === STEPS.length - 1) {
        allDone = true;
        if (pending) setTimeout(() => render(pending), 350);
      }
    }, ms)
  );

  const fd = new FormData();
  fd.append("file", selectedFile);

  try {
    const resp = await fetch("/analyze", { method: "POST", body: fd });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(e.detail || `Server error ${resp.status}`);
    }
    const data = await resp.json();
    markAll();
    if (allDone) setTimeout(() => render(data), 350);
    else pending = data;
  } catch (e) {
    stepTimers.forEach(clearTimeout);
    get("error-msg").textContent = String(e).replace(/^Error:\s*/, "");
    show("state-error");
  }
}

// ── Results ──
function render(data) {
  const isGenuine = data.verdict === "GENUINE";
  const pct       = Math.round(data.fake_prob * 100);

  // verdict card
  const vc = get("verdict-card");
  vc.className = `verdict ${isGenuine ? "genuine" : "fake"}`;
  get("verdict-badge").textContent  = isGenuine ? "Genuine" : "Fake";
  get("fake-pct").textContent       = `${pct}%`;
  get("verdict-label").textContent  = isGenuine ? "Likely authentic" : "Likely deepfake";
  get("verdict-desc").textContent   = isGenuine
    ? "The model found no significant inconsistency between the audio-text and visual emotion embeddings."
    : "The model detected a significant mismatch between audio-text and visual emotion embeddings.";

  // score bar — fill position maps 0→green end, 100→red end
  get("score-fill").style.width            = `${pct}%`;
  get("score-fill").style.backgroundPosition = `${pct}% 0`;
  get("score-pct-label").textContent = `${pct}%`;
  const tag = get("score-tag");
  tag.textContent = isGenuine ? "Genuine" : "Fake";
  tag.className   = `score-tag ${isGenuine ? "genuine" : "fake"}`;

  // discrepancy L2
  const l2 = data.discrepancy_l2;
  get("disc-value").textContent = l2 != null ? l2.toFixed(3) : "—";

  // model stats
  const ms  = data.model_stats || {};
  const rows = [
    ["AUC",  fmt(ms.auc)],
    ["F1",   fmt(ms.f1)],
    ["ACC",  pctFmt(ms.accuracy)],
    ["Prec", fmt(ms.precision)],
    ["Rec",  fmt(ms.recall)],
  ];
  get("stat-grid").innerHTML = rows.map(([k, v]) =>
    `<div class="stat-item">
       <span class="stat-val">${v}</span>
       <span class="stat-key">${k}</span>
     </div>`
  ).join("");

  // transcript
  get("transcript-text").textContent = data.transcript || "[not available]";

  show("state-results");
}

function fmt(v)    { return v != null ? v.toFixed(3) : "—"; }
function pctFmt(v) { return v != null ? `${(v * 100).toFixed(1)}%` : "—"; }
