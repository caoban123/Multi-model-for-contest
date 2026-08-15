const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const clipQueryInput = document.querySelector("#clip-query");
const topKInput = document.querySelector("#top-k");
const candidatePoolInput = document.querySelector("#candidate-pool");
const searchButton = document.querySelector("#search-button");
const translateButton = document.querySelector("#translate-button");
const exportButton = document.querySelector("#export-button");
const exportPinsButton = document.querySelector("#export-pins-button");
const clearHistoryButton = document.querySelector("#clear-history-button");
const visualModeButton = document.querySelector("#visual-mode-button");
const metadataModeButton = document.querySelector("#metadata-mode-button");
const statusEl = document.querySelector("#status");
const healthEl = document.querySelector("#health");
const resultsEl = document.querySelector("#results");
const pinsListEl = document.querySelector("#pins-list");
const historyListEl = document.querySelector("#history-list");
const resultTemplate = document.querySelector("#result-template");
let currentPayload = null;
let reviewState = new Map();
let activeMode = "visual";
const STORAGE_KEYS = {
  history: "aic_retrieval_history_v1",
  pins: "aic_retrieval_pins_v1",
};
let searchHistory = loadStoredArray(STORAGE_KEYS.history);
let pinnedResults = loadStoredArray(STORAGE_KEYS.pins);

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed");
  }
  return payload;
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function secondsLabel(value) {
  if (value === "" || value === null || value === undefined) {
    return "-";
  }
  const seconds = Number(value || 0);
  return `${seconds.toFixed(1)}s`;
}

function resultKey(result) {
  return `${result.video_id}:${result.keyframe_id ?? "metadata"}`;
}

function loadStoredArray(key) {
  try {
    const value = JSON.parse(localStorage.getItem(key) || "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

function saveStoredArray(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function renderResults(payload) {
  currentPayload = payload;
  reviewState = new Map();
  resultsEl.replaceChildren();
  exportButton.disabled = true;
  if (!payload.results.length) {
    setStatus("No results.");
    return;
  }
  setStatus(`${payload.results.length} results in ${payload.elapsed_ms.toFixed(1)} ms.`);
  for (const result of payload.results.map((item) => normalizeResult(item, payload.mode))) {
    const node = resultTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.resultKey = resultKey(result);
    node.dataset.result = JSON.stringify(result);
    const thumbWrap = node.querySelector(".thumb-wrap");
    const img = node.querySelector(".thumb");
    node.querySelector(".rank").textContent = `#${result.rank}`;
    node.querySelector(".video-id").textContent = result.video_id;
    node.querySelector(".score").textContent = result.score.toFixed(4);
    node.querySelector(".title").textContent = result.metadata.title || "Untitled";
    node.querySelector(".keyframe").textContent = `${result.keyframe_id ?? "-"}`;
    node.querySelector(".time").textContent = secondsLabel(result.pts_time);
    node.querySelector(".author").textContent = result.metadata.author || "-";
    node.querySelector(".date").textContent = result.metadata.publish_date || "-";

    if (result.image_url) {
      img.src = result.image_url;
      img.alt = `${result.video_id} keyframe ${result.keyframe_id}`;
      img.addEventListener("error", () => thumbWrap.classList.add("is-missing"), { once: true });
    } else {
      thumbWrap.classList.add("is-missing");
    }

    const keywords = node.querySelector(".keywords");
    for (const keyword of result.metadata.keywords || []) {
      const pill = document.createElement("span");
      pill.className = "keyword";
      pill.textContent = keyword;
      keywords.appendChild(pill);
    }
    const pinButton = node.querySelector(".pin-button");
    const isPinned = pinnedResults.some((item) => item.key === resultKey(result));
    pinButton.classList.toggle("is-active", isPinned);
    pinButton.setAttribute("aria-pressed", String(isPinned));
    pinButton.textContent = isPinned ? "Pinned" : "Pin";
    resultsEl.appendChild(node);
  }
  exportButton.disabled = false;
}

function normalizeResult(result, mode) {
  if (mode !== "metadata") {
    return result;
  }
  return {
    rank: result.rank,
    score: result.score,
    video_id: result.video_id,
    group: result.video_id.split("_")[0],
    keyframe_id: null,
    frame_idx: "",
    pts_time: "",
    fps: "",
    keyframe_path: "",
    image_url: null,
    matched_terms: result.matched_terms || [],
    metadata: {
      title: result.title || "",
      author: result.author || "",
      publish_date: result.publish_date || "",
      watch_url: result.watch_url || "",
      keywords: result.keywords || [],
      description_preview: result.description_preview || "",
    },
  };
}

async function runSearch(event) {
  event.preventDefault();
  const originalQuery = queryInput.value.trim();
  const clipQuery = activeMode === "visual" ? clipQueryInput.value.trim() : "";
  const query = activeMode === "visual" ? clipQuery || originalQuery : originalQuery;
  if (!query) {
    queryInput.focus();
    return;
  }
  const params = new URLSearchParams({
    q: query,
    top_k: topKInput.value,
    candidate_pool: candidatePoolInput.value,
    max_frames_per_video: "1",
  });
  searchButton.disabled = true;
  setStatus(activeMode === "metadata" ? "Searching metadata..." : "Encoding query and searching...");
  try {
    const endpoint = activeMode === "metadata" ? "/api/metadata-search" : "/api/search";
    if (activeMode === "metadata") {
      params.delete("candidate_pool");
      params.delete("max_frames_per_video");
    }
    const payload = await fetchJson(`${endpoint}?${params.toString()}`);
    payload.original_query = originalQuery;
    payload.clip_query = activeMode === "visual" ? query : "";
    payload.mode = payload.mode || activeMode;
    saveHistoryItem(originalQuery, payload.clip_query, payload.mode);
    renderResults(payload);
    renderHistory();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    searchButton.disabled = false;
  }
}

function saveHistoryItem(originalQuery, clipQuery, mode) {
  const item = {
    mode,
    original_query: originalQuery,
    clip_query: clipQuery,
    created_at: new Date().toISOString(),
  };
  searchHistory = [
    item,
    ...searchHistory.filter(
      (entry) => entry.mode !== mode || entry.original_query !== originalQuery || entry.clip_query !== clipQuery
    ),
  ].slice(0, 30);
  saveStoredArray(STORAGE_KEYS.history, searchHistory);
}

async function translateQuery() {
  const query = queryInput.value.trim();
  if (!query) {
    queryInput.focus();
    return;
  }
  translateButton.disabled = true;
  setStatus("Translating query...");
  try {
    const params = new URLSearchParams({ q: query });
    const payload = await fetchJson(`/api/translate?${params.toString()}`);
    clipQueryInput.value = payload.translated_text;
    setStatus(`Translated with ${payload.model}. Review the CLIP query, then search.`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    translateButton.disabled = false;
  }
}

function updateJudgement(target) {
  const card = target.closest(".result-card");
  const key = card.dataset.resultKey;
  const state = reviewState.get(key) || {};
  state.manual_judgement = target.dataset.judgement;
  reviewState.set(key, state);
  for (const button of card.querySelectorAll(".judgement-button")) {
    const isActive = button === target;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  }
}

function updateNote(target) {
  const card = target.closest(".result-card");
  const key = card.dataset.resultKey;
  const state = reviewState.get(key) || {};
  state.manual_notes = target.value;
  reviewState.set(key, state);
}

function togglePin(target) {
  const card = target.closest(".result-card");
  const result = JSON.parse(card.dataset.result);
  const key = resultKey(result);
  const existing = pinnedResults.findIndex((item) => item.key === key);
  if (existing >= 0) {
    pinnedResults.splice(existing, 1);
    target.classList.remove("is-active");
    target.setAttribute("aria-pressed", "false");
    target.textContent = "Pin";
  } else {
    pinnedResults = [
      {
        key,
        pinned_at: new Date().toISOString(),
        query_text: currentPayload?.original_query || currentPayload?.query || "",
        clip_query: currentPayload?.mode === "metadata" ? "" : currentPayload?.clip_query || currentPayload?.query || "",
        mode: currentPayload?.mode || activeMode,
        result,
      },
      ...pinnedResults,
    ].slice(0, 50);
    target.classList.add("is-active");
    target.setAttribute("aria-pressed", "true");
    target.textContent = "Pinned";
  }
  saveStoredArray(STORAGE_KEYS.pins, pinnedResults);
  renderPins();
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (/[",\r\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function buildCsvRows() {
  if (!currentPayload) {
    return [];
  }
  const queryId = `ui_${Date.now()}`;
  const headers = [
    "query_id",
    "query_text",
    "mode",
    "clip_query",
    "rank",
    "video_id",
    "keyframe_id",
    "frame_idx",
    "pts_time",
    "score",
    "manual_judgement",
    "manual_notes",
    "title",
    "author",
    "publish_date",
    "keyframe_path",
  ];
  const rows = [headers];
  for (const result of currentPayload.results.map((item) => normalizeResult(item, currentPayload.mode))) {
    const state = reviewState.get(resultKey(result)) || {};
    rows.push([
      queryId,
      currentPayload.original_query || currentPayload.query || "",
      currentPayload.mode || activeMode,
      currentPayload.mode === "metadata" ? "" : currentPayload.clip_query || currentPayload.query || "",
      result.rank,
      result.video_id,
      result.keyframe_id,
      result.frame_idx,
      result.pts_time,
      result.score,
      state.manual_judgement || "",
      state.manual_notes || "",
      result.metadata.title || "",
      result.metadata.author || "",
      result.metadata.publish_date || "",
      result.keyframe_path || "",
    ]);
  }
  return rows;
}

function buildPinnedCsvRows() {
  const headers = [
    "pinned_at",
    "query_text",
    "mode",
    "clip_query",
    "video_id",
    "keyframe_id",
    "frame_idx",
    "pts_time",
    "score",
    "title",
    "author",
    "publish_date",
    "keyframe_path",
  ];
  const rows = [headers];
  for (const item of pinnedResults) {
    const result = item.result;
    rows.push([
      item.pinned_at,
      item.query_text,
      item.mode || "",
      item.clip_query,
      result.video_id,
      result.keyframe_id,
      result.frame_idx,
      result.pts_time,
      result.score,
      result.metadata?.title || "",
      result.metadata?.author || "",
      result.metadata?.publish_date || "",
      result.keyframe_path || "",
    ]);
  }
  return rows;
}

function exportCsv() {
  const rows = buildCsvRows();
  if (!rows.length) {
    return;
  }
  const csv = rows.map((row) => row.map(csvEscape).join(",")).join("\r\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  link.href = url;
  link.download = `aic_retrieval_review_${stamp}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  setStatus(`Exported ${rows.length - 1} review rows.`);
}

function downloadCsv(rows, filenamePrefix) {
  if (!rows.length) {
    return;
  }
  const csv = rows.map((row) => row.map(csvEscape).join(",")).join("\r\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  link.href = url;
  link.download = `${filenamePrefix}_${stamp}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function exportPinnedCsv() {
  const rows = buildPinnedCsvRows();
  if (rows.length <= 1) {
    return;
  }
  downloadCsv(rows, "aic_pinned_results");
  setStatus(`Exported ${rows.length - 1} pinned rows.`);
}

function renderPins() {
  pinsListEl.replaceChildren();
  exportPinsButton.disabled = pinnedResults.length === 0;
  if (!pinnedResults.length) {
    pinsListEl.appendChild(emptyCompactItem("No pinned results."));
    return;
  }
  for (const item of pinnedResults) {
    const result = item.result;
    const node = document.createElement("div");
    node.className = "compact-item";
    const title = document.createElement("strong");
    title.textContent = `${result.video_id} #${result.keyframe_id ?? "-"}`;
    const meta = document.createElement("div");
    meta.className = "compact-meta";
    meta.textContent = `${secondsLabel(result.pts_time)} - ${result.metadata?.title || "Untitled"}`;
    const remove = document.createElement("button");
    remove.className = "secondary-button small-button";
    remove.type = "button";
    remove.textContent = "Unpin";
    remove.addEventListener("click", () => {
      pinnedResults = pinnedResults.filter((entry) => entry.key !== item.key);
      saveStoredArray(STORAGE_KEYS.pins, pinnedResults);
      syncVisiblePinButtons();
      renderPins();
    });
    node.append(title, meta, remove);
    pinsListEl.appendChild(node);
  }
}

function renderHistory() {
  historyListEl.replaceChildren();
  if (!searchHistory.length) {
    historyListEl.appendChild(emptyCompactItem("No search history."));
    return;
  }
  for (const item of searchHistory) {
    const node = document.createElement("div");
    node.className = "compact-item";
    const button = document.createElement("button");
    button.className = "secondary-button";
    button.type = "button";
    button.textContent = item.original_query || item.clip_query;
    button.addEventListener("click", () => {
      setMode(item.mode || "visual");
      queryInput.value = item.original_query || "";
      clipQueryInput.value = item.clip_query || "";
      form.requestSubmit();
    });
    const meta = document.createElement("div");
    meta.className = "compact-meta";
    meta.textContent = item.clip_query ? `${item.mode || "visual"} - ${item.clip_query}` : item.mode || "visual";
    node.append(button, meta);
    historyListEl.appendChild(node);
  }
}

function setMode(mode) {
  activeMode = mode;
  const isVisual = mode === "visual";
  visualModeButton.classList.toggle("is-active", isVisual);
  metadataModeButton.classList.toggle("is-active", !isVisual);
  visualModeButton.setAttribute("aria-pressed", String(isVisual));
  metadataModeButton.setAttribute("aria-pressed", String(!isVisual));
  translateButton.disabled = !isVisual;
  clipQueryInput.disabled = !isVisual;
  candidatePoolInput.disabled = !isVisual;
  setStatus(isVisual ? "Visual search ready." : "Metadata search ready.");
}

function emptyCompactItem(text) {
  const node = document.createElement("div");
  node.className = "compact-item compact-meta";
  node.textContent = text;
  return node;
}

function clearHistory() {
  searchHistory = [];
  saveStoredArray(STORAGE_KEYS.history, searchHistory);
  renderHistory();
}

function syncVisiblePinButtons() {
  for (const card of resultsEl.querySelectorAll(".result-card")) {
    const button = card.querySelector(".pin-button");
    const isPinned = pinnedResults.some((item) => item.key === card.dataset.resultKey);
    button.classList.toggle("is-active", isPinned);
    button.setAttribute("aria-pressed", String(isPinned));
    button.textContent = isPinned ? "Pinned" : "Pin";
  }
}

async function loadHealth() {
  try {
    const payload = await fetchJson("/api/health");
    const translation = payload.translation_configured ? "translation on" : "translation off";
    healthEl.textContent = `${payload.groups.join(", ")} - ${payload.index_vectors} vectors - ${payload.index_dim} dim - ${payload.metadata_documents} metadata - ${translation}`;
  } catch (error) {
    healthEl.textContent = error.message;
  }
}

form.addEventListener("submit", runSearch);
translateButton.addEventListener("click", translateQuery);
exportButton.addEventListener("click", exportCsv);
exportPinsButton.addEventListener("click", exportPinnedCsv);
clearHistoryButton.addEventListener("click", clearHistory);
visualModeButton.addEventListener("click", () => setMode("visual"));
metadataModeButton.addEventListener("click", () => setMode("metadata"));
resultsEl.addEventListener("click", (event) => {
  if (event.target.classList.contains("judgement-button")) {
    updateJudgement(event.target);
  }
  if (event.target.classList.contains("pin-button")) {
    togglePin(event.target);
  }
});
resultsEl.addEventListener("input", (event) => {
  if (event.target.classList.contains("note-input")) {
    updateNote(event.target);
  }
});
renderPins();
renderHistory();
loadHealth();
