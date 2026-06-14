"use strict";

// ── Processing steps (labels match the real backend pipeline) ──
const STEPS = [
  { id: "audio",        label: "Extracting audio waveform" },
  { id: "speech_text",  label: "Running MDCNN + BERT encoder → Z_at" },
  { id: "keyframes",    label: "Extracting video keyframes (Top-8)" },
  { id: "fvlitenet",    label: "Running FVLiteNet encoder → Z_v" },
  { id: "emotion",      label: "Computing emotion heads A + B" },
  { id: "discrepancy",  label: "Computing discrepancy score Δ" },
  { id: "classifier",   label: "Bilinear fusion + classifier → P(fake)" },
];

// Approx wall-clock offset (ms) when each step *appears* to finish during animation.
// Real inference drives the final reveal; these just keep the UI alive while waiting.
const STEP_MS = [900, 3800, 5500, 7200, 9000, 10200, 11500];

// ── State ──
let selectedFile = null;
let stepTimers   = [];
let allStepsDone = false;
let pendingData  = null;   // inference result arrives before animation ends

// ── DOM refs ──
const $idle      = id("state-idle");
const $selected  = id("state-selected");
const $analyzing = id("state-analyzing");
const $results   = id("state-results");
const $error     = id("state-error");
const $fileInput = id("file-input");

function id(x) { return document.getElementById(x); }

function showState(stateId) {
  [$idle, $selected, $analyzing, $results, $error].forEach(el => {
    el.hidden = el.id !== stateId;
  });
}

// ── File selection ──
$idle.addEventListener("click",     () => $fileInput.click());
$selected.addEventListener("click", e => { if (e.target.id !== "run-btn") $fileInput.click(); });

$idle.addEventListener("dragover", e => { e.preventDefault(); $idle.classList.add("drag-over"); });
$idle.addEventListener("dragleave",  () => $idle.classList.remove("drag-over"));
$idle.addEventListener("drop", e => {
  e.preventDefault();
  $idle.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});

$fileInput.addEventListener("change", () => {
  if ($fileInput.files[0]) setFile($fileInput.files[0]);
});

function setFile(f) {
  selectedFile = f;
  id("file-info").textContent = `${f.name} · ${(f.size / 1e6).toFixed(1)} MB`;
  showState("state-selected");
}

id("run-btn").addEventListener("click",  e => { e.stopPropagation(); startAnalysis(); });
id("reset-btn").addEventListener("click",  reset);
id("retry-btn").addEventListener("click",  reset);

function reset() {
  selectedFile = null;
  $fileInput.value = "";
  stepTimers.forEach(clearTimeout);
  stepTimers = [];
  allStepsDone = false;
  pendingData  = null;
  showState("state-idle");
}

// ── Analysis ──
function buildSteps() {
  id("steps-list").innerHTML = STEPS.map((s, i) => `
    <li id="step-${i}">
      <span class="step-icon" id="icon-${i}"></span>${s.label}
    </li>`).join("");
}

function markStep(i) {
  const li = id(`step-${i}`);
  if (!li || li.classList.contains("done")) return;
  li.classList.add("done");
  id(`icon-${i}`).textContent = "✓";
}

function markAllSteps() {
  STEPS.forEach((_, i) => markStep(i));
}

async function startAnalysis() {
  if (!selectedFile) return;

  id("analyzing-sub").textContent = selectedFile.name;
  buildSteps();
  showState("state-analyzing");
  allStepsDone = false;
  pendingData  = null;

  // Animate steps with approximate timings
  stepTimers = STEP_MS.map((ms, i) =>
    setTimeout(() => {
      markStep(i);
      if (i === STEPS.length - 1) {
        allStepsDone = true;
        if (pendingData) setTimeout(() => renderResults(pendingData), 350);
      }
    }, ms)
  );

  // Fire the real POST request
  const fd = new FormData();
  fd.append("file", selectedFile);

  try {
    const resp = await fetch("/analyze", { method: "POST", body: fd });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `Server error ${resp.status}`);
    }
    const data = await resp.json();

    markAllSteps();               // complete remaining steps immediately
    if (allStepsDone) {
      setTimeout(() => renderResults(data), 350);
    } else {
      pendingData = data;         // wait for last step timer
    }
  } catch (err) {
    stepTimers.forEach(clearTimeout);
    id("error-msg").textContent = String(err).replace(/^Error:\s*/, "");
    showState("state-error");
  }
}

// ── Results rendering ──
function renderResults(data) {
  const isGenuine = data.verdict === "GENUINE";
  const pct       = Math.round(data.fake_prob * 100);
  const vc        = id("verdict-card");

  vc.className = `verdict-card ${isGenuine ? "genuine" : "fake"}`;
  id("verdict-badge").textContent  = isGenuine ? "Real" : "Fake";
  id("fake-pct").textContent       = `${pct}%`;
  id("verdict-label").textContent  = isGenuine ? "Likely authentic" : "Likely deepfake";
  id("verdict-desc").textContent   = isGenuine
    ? "Emotions are consistent across audio and visual modalities."
    : "Significant audio-visual emotion discrepancy detected.";

  // Model stats chips
  const ms = data.model_stats || {};
  id("verdict-metrics").innerHTML = [
    ["AUC",  fmt(ms.auc)],
    ["ACC",  pctFmt(ms.accuracy)],
    ["Prec", fmt(ms.precision)],
    ["Rec",  fmt(ms.recall)],
    ["F1",   fmt(ms.f1)],
    ["L2 Δ", data.discrepancy_l2 != null ? data.discrepancy_l2.toFixed(3) : "—"],
  ].map(([k, v]) =>
    `<span class="metric-chip"><strong>${k}</strong> ${v}</span>`
  ).join("");

  // Dominant mismatch
  const dm = data.dominant_mismatch;
  id("dom-label").textContent = `Dominant mismatch · ${dm.emotion}`;
  id("dom-delta").textContent = `Δ = ${dm.delta.toFixed(2)}`;
  const audioVal  = (data.audio_emotion  || {})[dm.emotion] || 0;
  const visualVal = (data.visual_emotion || {})[dm.emotion] || 0;
  id("dom-bar").style.width = `${Math.max(audioVal, visualVal) * 100}%`;
  const sig = dm.delta < 0.05 ? "low" : dm.delta < 0.15 ? "moderate" : "high";
  id("dom-hint").textContent =
    `audio says ${(dm.audio_dominant  || dm.emotion).toLowerCase()} · ` +
    `face says ${(dm.visual_dominant || dm.emotion).toLowerCase()} · ` +
    `${sig} mismatch signal`;

  // Emotion bars
  renderEmoBars("audio-bars",  data.audio_emotion  || {}, "bar-purple");
  renderEmoBars("visual-bars", data.visual_emotion || {}, "bar-green");

  // Per-class delta
  const deltaDiv = id("delta-bars");
  deltaDiv.innerHTML = Object.entries(data.per_class_delta || {})
    .sort((a, b) => b[1] - a[1])
    .map(([emo, d]) => {
      const tier  = d < 0.05 ? "low" : d < 0.15 ? "med" : "high";
      const label = tier === "low" ? "Low" : tier === "med" ? "Med" : "High";
      return `
        <div class="delta-row">
          <span class="delta-label">${emo}</span>
          <div class="delta-track">
            <div class="delta-fill bar-neutral" style="width:${Math.min(d * 200, 100)}%"></div>
          </div>
          <span class="delta-val">${d.toFixed(2)}</span>
          <span class="tag-sm tag-${tier}">${label}</span>
        </div>`;
    }).join("");

  // Transcript
  id("transcript-text").textContent = data.transcript || "[not available]";

  showState("state-results");
}

function renderEmoBars(containerId, emoData, barClass) {
  const container = id(containerId);
  const sorted    = Object.entries(emoData).sort((a, b) => b[1] - a[1]);
  container.innerHTML = sorted.map(([emo, val]) =>
    `<div class="emo-row">
       <span class="emo-label">${emo}</span>
       <div class="emo-track">
         <div class="emo-fill ${barClass}" style="width:${(val * 100).toFixed(1)}%"></div>
       </div>
       <span class="emo-value">${val.toFixed(2)}</span>
     </div>`
  ).join("");
}

function fmt(v)    { return v != null ? v.toFixed(3) : "—"; }
function pctFmt(v) { return v != null ? `${(v * 100).toFixed(1)}%` : "—"; }
