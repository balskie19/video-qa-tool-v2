// ─── State ───────────────────────────────────────────────
let activeTab      = "url";
let _urlRowCount   = 0;
let _fileRowCount  = 0;
let _fileRowData   = {};   // rowId -> File
let _queue         = [];
let _currentReport = null;
let _elapsedInterval = null;
let _elapsedStart    = 0;

// ─── DOM ─────────────────────────────────────────────────
const scriptToggle  = document.getElementById("script-toggle");
const scriptPanel   = document.getElementById("script-panel");
const scriptInput   = document.getElementById("script-input");

const tabs        = document.querySelectorAll(".tab");
const panelUrl    = document.getElementById("panel-url");
const panelUpload = document.getElementById("panel-upload");
const urlList     = document.getElementById("url-list");
const fileList    = document.getElementById("file-list");
const btnAddUrl   = document.getElementById("btn-add-url");
const btnAddFile  = document.getElementById("btn-add-file");
const btnAnalyze  = document.getElementById("btn-analyze");
const btnLabel    = document.getElementById("btn-label");

const inputCard      = document.getElementById("input-card");
const progressCard   = document.getElementById("progress-card");
const resultsList    = document.getElementById("results-list");
const completionBanner = document.getElementById("completion-banner");
const completionText   = document.getElementById("completion-text");
const btnReset       = document.getElementById("btn-reset");

const progressCounter = document.getElementById("progress-counter");
const progressElapsed = document.getElementById("progress-elapsed");
const progressStep    = document.getElementById("progress-step");
const progressPct     = document.getElementById("progress-pct");
const progressBar     = document.getElementById("progress-bar");


// ─── Script Toggle ────────────────────────────────────────
scriptToggle.addEventListener("click", () => {
  const expanded = scriptToggle.getAttribute("aria-expanded") === "true";
  scriptToggle.setAttribute("aria-expanded", !expanded);
  scriptPanel.classList.toggle("hidden", expanded);
});


// ─── Tabs ─────────────────────────────────────────────────
tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    activeTab = tab.dataset.tab;
    tabs.forEach(t => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", t === tab ? "true" : "false");
    });
    panelUrl.classList.toggle("hidden", activeTab !== "url");
    panelUpload.classList.toggle("hidden", activeTab !== "upload");
  });
});


// ─── URL Rows ────────────────────────────────────────────
function addUrlRow(value = "") {
  const id = `url-row-${_urlRowCount++}`;
  const row = document.createElement("div");
  row.className = "input-row";
  row.id = id;
  row.innerHTML = `
    <div class="input-row-main">
      <input
        type="url"
        class="text-input url-single-input"
        placeholder="https://replay.dropbox.com/share/..."
        autocomplete="off"
        spellcheck="false"
      />
      <button class="btn-remove-row" type="button" aria-label="Remove this link">&times;</button>
    </div>
    <div class="row-script-section">
      <button class="row-script-toggle" type="button" aria-expanded="false">
        <span class="row-script-toggle-icon">+</span> Add Script
      </button>
      <textarea
        class="row-script-textarea hidden"
        placeholder="Paste the script for this video — helps catch caption mismatches and sentence splits..."
        rows="4"
        spellcheck="false"
      ></textarea>
    </div>
  `;
  if (value) row.querySelector(".url-single-input").value = value;
  row.querySelector(".btn-remove-row").addEventListener("click", () => {
    removeRow(id, urlList);
  });
  _bindRowScriptToggle(row);
  urlList.appendChild(row);
  updateRemoveVisibility(urlList);
  return row;
}

btnAddUrl.addEventListener("click", () => {
  addUrlRow();
  urlList.lastElementChild.querySelector("input").focus();
});


// ─── File Rows ───────────────────────────────────────────
function addFileRow() {
  const id = `file-row-${_fileRowCount++}`;
  const row = document.createElement("div");
  row.className = "input-row";
  row.id = id;
  row.innerHTML = `
    <div class="input-row-main">
      <div class="file-pick-area" role="button" tabindex="0" aria-label="Choose a video file">
        <span class="file-pick-icon" aria-hidden="true">&#128250;</span>
        <span class="file-pick-label">Choose video file…</span>
        <input type="file" accept="video/*" class="file-pick-input" aria-hidden="true" tabindex="-1" />
      </div>
      <button class="btn-remove-row" type="button" aria-label="Remove this file">&times;</button>
    </div>
    <div class="row-script-section">
      <button class="row-script-toggle" type="button" aria-expanded="false">
        <span class="row-script-toggle-icon">+</span> Add Script
      </button>
      <textarea
        class="row-script-textarea hidden"
        placeholder="Paste the script for this video — helps catch caption mismatches and sentence splits..."
        rows="4"
        spellcheck="false"
      ></textarea>
    </div>
  `;
  const pickArea  = row.querySelector(".file-pick-area");
  const fileInput = row.querySelector(".file-pick-input");
  const pickLabel = row.querySelector(".file-pick-label");

  pickArea.addEventListener("click", () => fileInput.click());
  pickArea.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) {
      _fileRowData[id] = fileInput.files[0];
      pickArea.classList.add("has-file");
      pickLabel.textContent = fileInput.files[0].name;
    }
  });
  row.querySelector(".btn-remove-row").addEventListener("click", () => {
    delete _fileRowData[id];
    removeRow(id, fileList);
  });
  _bindRowScriptToggle(row);
  fileList.appendChild(row);
  updateRemoveVisibility(fileList);
  return row;
}


// ─── Per-row Script Toggle ───────────────────────────────
function _bindRowScriptToggle(row) {
  const toggle   = row.querySelector(".row-script-toggle");
  const textarea = row.querySelector(".row-script-textarea");
  if (!toggle || !textarea) return;
  toggle.addEventListener("click", () => {
    const expanded = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", !expanded);
    textarea.classList.toggle("hidden", expanded);
    toggle.querySelector(".row-script-toggle-icon").textContent = expanded ? "+" : "×";
    if (!expanded) textarea.focus();
  });
}

btnAddFile.addEventListener("click", () => addFileRow());


// ─── Row Helpers ─────────────────────────────────────────
function removeRow(id, list) {
  const row = document.getElementById(id);
  if (row) row.remove();
  updateRemoveVisibility(list);
}

function updateRemoveVisibility(list) {
  const rows = list.querySelectorAll(".input-row");
  rows.forEach(r => {
    const btn = r.querySelector(".btn-remove-row");
    if (btn) btn.classList.toggle("hidden", rows.length === 1);
  });
}

function clearUrlRows() {
  urlList.innerHTML = "";
  _urlRowCount = 0;
  addUrlRow();
}

function clearFileRows() {
  fileList.innerHTML = "";
  _fileRowCount = 0;
  _fileRowData = {};
  addFileRow();
}


// ─── Elapsed Timer ───────────────────────────────────────
function startElapsedTimer() {
  _elapsedStart = Date.now();
  progressElapsed.textContent = "0:00";
  clearInterval(_elapsedInterval);
  _elapsedInterval = setInterval(() => {
    const s = Math.floor((Date.now() - _elapsedStart) / 1000);
    const m = Math.floor(s / 60);
    progressElapsed.textContent = `${m}:${String(s % 60).padStart(2, "0")}`;
  }, 1000);
}
function stopElapsedTimer() {
  clearInterval(_elapsedInterval);
  _elapsedInterval = null;
}


// ─── Analyze ─────────────────────────────────────────────
btnAnalyze.addEventListener("click", startAnalysis);

async function startAnalysis() {
  if (activeTab === "url") {
    const rows = urlList.querySelectorAll(".input-row");
    const items = Array.from(rows).map(row => ({
      url:    (row.querySelector(".url-single-input")?.value || "").trim(),
      script: (row.querySelector(".row-script-textarea")?.value || "").trim(),
    })).filter(i => i.url);
    if (!items.length) {
      const first = urlList.querySelector(".url-single-input");
      if (first) {
        first.focus();
        first.style.borderColor = "var(--color-critical)";
        setTimeout(() => first.style.borderColor = "", 1500);
      }
      return;
    }
    _queue = items.map(i => ({ type: "url", value: i.url, script: i.script }));
  } else {
    const rows = fileList.querySelectorAll(".input-row");
    const rowScripts = {};
    rows.forEach(row => {
      const id = row.id;
      if (_fileRowData[id]) {
        rowScripts[id] = (row.querySelector(".row-script-textarea")?.value || "").trim();
      }
    });
    const files = Object.keys(_fileRowData);
    if (!files.length) {
      const first = fileList.querySelector(".file-pick-area");
      if (first) {
        first.style.borderColor = "var(--color-critical)";
        setTimeout(() => first.style.borderColor = "", 1500);
      }
      return;
    }
    _queue = files.map(id => ({ type: "file", value: _fileRowData[id], script: rowScripts[id] || "" }));
  }

  resultsList.innerHTML = "";
  completionBanner.classList.add("hidden");
  btnAnalyze.disabled = true;
  btnAnalyze.setAttribute("aria-busy", "true");
  progressCard.classList.remove("hidden");

  const total = _queue.length;

  for (let i = 0; i < total; i++) {
    updateCounter(i + 1, total);
    startElapsedTimer();
    setProgress("Preparing video...", 2);
    const videoStart = Date.now();
    await runSingleAnalysis(_queue[i], videoStart);
    stopElapsedTimer();
  }

  showCompletion(total);
}

function updateCounter(current, total) {
  if (total > 1) {
    progressCounter.textContent = `Video ${current} of ${total}`;
    progressCounter.classList.remove("hidden");
    btnLabel.textContent = `Analyzing ${current} of ${total}…`;
  } else {
    progressCounter.classList.add("hidden");
    btnLabel.textContent = "Analyzing…";
  }
}

async function runSingleAnalysis(item, videoStart = Date.now()) {
  const fd = new FormData();
  if (item.type === "url") {
    fd.append("url", item.value);
  } else {
    fd.append("file", item.value);
  }
  // Per-video script takes priority; fall back to global context box
  const contextText = item.script || scriptInput.value.trim();
  if (contextText) fd.append("context", contextText);

  try {
    const resp = await fetch("/analyze", { method: "POST", body: fd });
    if (!resp.ok) throw new Error(`Server error ${resp.status}`);

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const json = line.slice(6).trim();
        if (!json) continue;
        try {
          const evt = JSON.parse(json);
          if (evt.type === "complete") {
            // Use backend name (if clean); then URL-derived fallback
            {
              const backendName = evt.report?.filename || "";
              // Reject if backend returned a raw URL or hash-only fallback
              const isJunk = !backendName
                || backendName.startsWith("http")
                || /^replay_[a-zA-Z0-9]{8,}\.mp4$/.test(backendName);
              if (isJunk) {
                evt.report.filename = item.type === "file"
                  ? item.value.name
                  : _extractUrlLabel(item.value);
              }
            }
            // Attach elapsed time to the report
            const elapsedSec = Math.floor((Date.now() - videoStart) / 1000);
            evt.report._elapsed = elapsedSec;
          }
          handleEvent(evt);
        } catch (_) {}
      }
    }
  } catch (err) {
    appendErrorCard(item, err.message);
  }
}

function handleEvent(evt) {
  if (evt.type === "progress") {
    setProgress(evt.step, evt.percent);
  } else if (evt.type === "complete") {
    console.log("[DEBUG] complete event filename:", evt.report?.filename, "| full report keys:", Object.keys(evt.report || {}));
    setProgress("Done!", 100);
    setTimeout(() => appendReportCard(evt.report), 300);
  } else if (evt.type === "error") {
    appendErrorCard(null, evt.message);
  }
}

function setProgress(step, pct) {
  progressStep.textContent = step;
  progressPct.textContent  = pct + "%";
  progressBar.style.width  = pct + "%";
  const track = document.getElementById("progress-track");
  if (track) track.setAttribute("aria-valuenow", pct);
}


// ─── Render Report ────────────────────────────────────────
function appendReportCard(report) {
  _currentReport = report;

  const section = document.createElement("section");
  section.className = "card results-card";
  section.setAttribute("aria-label", "QA report");

  // Filter Minor issues where the AI itself says it's a Whisper mishear / caption is correct
  const _whisperPhrases = [
    "whisper", "mishear", "caption is correct", "transcript is wrong",
    "caption correctly", "whisper misread", "phonetic", "specialized vocabulary"
  ];
  const issues = (report.issues || []).filter(issue => {
    if ((issue.severity || "").toLowerCase() !== "minor") return true;
    const text = ((issue.description || "") + " " + (issue.issue || "")).toLowerCase();
    return !_whisperPhrases.some(p => text.includes(p));
  });
  const count = issues.length;

  const badgeClass = count === 0 ? "badge-ok" : count <= 3 ? "badge-warn" : "badge-fail";
  const badgeText  = count === 0 ? "No issues" : count === 1 ? "1 issue" : `${count} issues`;

  const elapsedStr = report._elapsed != null
    ? (() => { const m = Math.floor(report._elapsed / 60), s = report._elapsed % 60; return `${m}:${String(s).padStart(2,"0")}`; })()
    : null;

  const durationStr = report.duration != null
    ? (() => { const m = Math.floor(report.duration / 60), s = report.duration % 60; return `${m}:${String(s).padStart(2,"0")}`; })()
    : null;

  section.innerHTML = `
    <div class="results-header">
      <div>
        <p class="report-label">QA Report${durationStr ? ` <span class="report-elapsed">&#127909; ${durationStr}</span>` : ""}${elapsedStr ? ` <span class="report-elapsed">&#9201; ${elapsedStr}</span>` : ""}</p>
        <h2 class="card-title report-title">${report.filename ? `<span class="report-file-icon" aria-hidden="true">&#128250;</span> ${esc(report.filename)}` : "QA Report"}</h2>
        <p class="summary-text">${esc(report.summary || "")}</p>
      </div>
      <div class="issue-badge ${badgeClass}" aria-label="Issue count">${esc(badgeText)}</div>
    </div>
    <div class="issues-grid"></div>
    <div class="results-actions">
      <button class="btn-copy-report" aria-label="Copy report to clipboard">
        <span aria-hidden="true">&#128203;</span>
        <span class="copy-report-label">Copy Report</span>
      </button>
    </div>
  `;

  const grid = section.querySelector(".issues-grid");

  if (count === 0 && !report.raw) {
    grid.innerHTML = `
      <div class="no-issues">
        <div class="no-issues-icon">&#10003;</div>
        <div class="no-issues-title">Looks good!</div>
        <p>No quality issues detected.</p>
      </div>`;
  } else if (report.raw) {
    const pre = document.createElement("pre");
    pre.style.cssText = "white-space:pre-wrap;font-size:13px;color:rgba(0,0,0,0.7)";
    pre.textContent = report.raw;
    grid.appendChild(pre);
  } else {
    const categoryOrder = ["Audio", "Captions", "Video", "Overlays"];
    const grouped = {};
    for (const issue of issues) {
      const cat = issue.category || "Other";
      if (!grouped[cat]) grouped[cat] = [];
      grouped[cat].push(issue);
    }
    const orderedCats = [
      ...categoryOrder.filter(c => grouped[c]),
      ...Object.keys(grouped).filter(c => !categoryOrder.includes(c)),
    ];
    for (const cat of orderedCats) {
      const header = document.createElement("div");
      header.className = "category-header";
      header.textContent = cat;
      grid.appendChild(header);
      for (const issue of grouped[cat]) {
        grid.appendChild(buildIssueCard(issue));
      }
    }
  }

  section.querySelector(".btn-copy-report").addEventListener("click", (e) => {
    const btn = e.currentTarget;
    const label = btn.querySelector(".copy-report-label");
    const text = formatReportText(report);
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(() => {
        label.textContent = "Copied!";
        setTimeout(() => { label.textContent = "Copy Report"; }, 2000);
      });
    } else {
      // HTTP fallback
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      try {
        document.execCommand("copy");
        label.textContent = "Copied!";
        setTimeout(() => { label.textContent = "Copy Report"; }, 2000);
      } catch (err) {
        label.textContent = "Copy failed";
        setTimeout(() => { label.textContent = "Copy Report"; }, 2000);
      }
      document.body.removeChild(ta);
    }
  });

  resultsList.appendChild(section);
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

function appendErrorCard(item, msg) {
  const div = document.createElement("div");
  div.className = "error-banner";
  div.textContent = `Error${item ? ` (${item.type === "url" ? item.value.slice(0, 60) + "…" : item.value.name})` : ""}: ${msg}`;
  resultsList.appendChild(div);
}

function showCompletion(total) {
  progressCard.classList.add("hidden");
  stopElapsedTimer();
  btnAnalyze.disabled = false;
  btnAnalyze.removeAttribute("aria-busy");
  btnLabel.textContent = "Run QA Analysis";

  completionText.textContent = total === 1
    ? "Analysis complete — 1 video processed"
    : `Analysis complete — ${total} videos processed`;
  completionBanner.classList.remove("hidden");
  completionBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
}


// ─── Issue Card ───────────────────────────────────────────
function buildIssueCard(issue) {
  const sev = issue.severity || "Minor";
  const div = document.createElement("div");
  div.className = `issue-item sev-${sev}`;

  div.innerHTML = `
    <div class="issue-top">
      <span class="issue-title">${esc(issue.issue || "Issue")}</span>
      <span class="sev-pill ${sev}">${esc(sev)}</span>
      <span class="cat-pill">${esc(issue.category || "")}</span>
      ${issue.timestamp ? `<span class="issue-ts">${esc(issue.timestamp)}</span>` : ""}
      <button class="btn-copy-issue" aria-label="Copy this issue" title="Copy issue">&#128203;</button>
    </div>
    <div class="issue-desc">${esc(issue.description || "")}</div>
  `;
  div.querySelector(".btn-copy-issue").addEventListener("click", (e) => {
    e.stopPropagation();
    const btn = e.currentTarget;
    copyIssueText(issue);
    btn.textContent = "✓";
    setTimeout(() => { btn.textContent = "📋"; }, 1500);
  });
  return div;
}


// ─── Copy Helpers ─────────────────────────────────────────
function formatReportText(report) {
  const lines = [];
  if (report.filename) lines.push(`FILE: ${report.filename}`);
  lines.push(`QA REPORT — ${report.issue_count ?? 0} issue(s)`);
  lines.push("─".repeat(50));
  if (report.summary) lines.push(report.summary);
  lines.push("");
  for (const issue of (report.issues || [])) {
    lines.push(`[${issue.severity}] [${issue.category}] ${issue.timestamp || ""}`);
    lines.push(issue.issue || "");
    lines.push(issue.description || "");
    lines.push("");
  }
  return lines.join("\n").trim();
}

function copyIssueText(issue) {
  const text = [
    `[${issue.severity}] [${issue.category}] ${issue.timestamp || ""}`,
    issue.issue || "",
    issue.description || "",
  ].join("\n");
  navigator.clipboard.writeText(text).then(() => {});
}


// ─── URL Label Helper ────────────────────────────────────
function _extractUrlLabel(url) {
  try {
    const u = new URL(url);
    const parts = u.pathname.split("/").filter(Boolean);
    const last = parts[parts.length - 1];
    // For Replay links the last path segment is an opaque hash — label it clearly
    if (u.hostname.includes("replay.dropbox.com")) {
      return `Dropbox Replay (${last.slice(0, 8)}...)`;
    }
    // Standard Dropbox share link — last path segment is the filename
    if (u.hostname.includes("dropbox.com")) {
      const name = decodeURIComponent(last);
      if (name && name.length > 3) return name;
    }
    if (last && last.length > 3) return decodeURIComponent(last);
    return u.hostname;
  } catch (_) {
    return url.slice(0, 60);
  }
}


// ─── Escape ───────────────────────────────────────────────
function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}


// ─── Reset ───────────────────────────────────────────────
btnReset.addEventListener("click", () => {
  resultsList.innerHTML = "";
  completionBanner.classList.add("hidden");
  progressCard.classList.add("hidden");
  scriptInput.value = "";
  scriptPanel.classList.add("hidden");
  scriptToggle.setAttribute("aria-expanded", "false");
  clearUrlRows();
  clearFileRows();
  btnAnalyze.disabled = false;
  btnAnalyze.removeAttribute("aria-busy");
  btnLabel.textContent = "Run QA Analysis";
  stopElapsedTimer();
  _currentReport = null;
  _queue = [];
  const old = document.querySelector(".error-banner");
  if (old) old.remove();
  window.scrollTo({ top: 0, behavior: "smooth" });
});


// ─── Init ────────────────────────────────────────────────
addUrlRow();
addFileRow();
