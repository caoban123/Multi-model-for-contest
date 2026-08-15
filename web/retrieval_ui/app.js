const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const clipQueryInput = document.querySelector("#clip-query");
const topKInput = document.querySelector("#top-k");
const candidatePoolInput = document.querySelector("#candidate-pool");
const searchButton = document.querySelector("#search-button");
const translateButton = document.querySelector("#translate-button");
const exportButton = document.querySelector("#export-button");
const statusEl = document.querySelector("#status");
const healthEl = document.querySelector("#health");
const resultsEl = document.querySelector("#results");
const resultTemplate = document.querySelector("#result-template");
let currentPayload = null;
let reviewState = new Map();

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
  const seconds = Number(value || 0);
  return `${seconds.toFixed(1)}s`;
}

function resultKey(result) {
  return `${result.video_id}:${result.keyframe_id}`;
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
  for (const result of payload.results) {
    const node = resultTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.resultKey = resultKey(result);
    const thumbWrap = node.querySelector(".thumb-wrap");
    const img = node.querySelector(".thumb");
    node.querySelector(".rank").textContent = `#${result.rank}`;
    node.querySelector(".video-id").textContent = result.video_id;
    node.querySelector(".score").textContent = result.score.toFixed(4);
    node.querySelector(".title").textContent = result.metadata.title || "Untitled";
    node.querySelector(".keyframe").textContent = `${result.keyframe_id}`;
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
    resultsEl.appendChild(node);
  }
  exportButton.disabled = false;
}

async function runSearch(event) {
  event.preventDefault();
  const originalQuery = queryInput.value.trim();
  const clipQuery = clipQueryInput.value.trim();
  const query = clipQuery || originalQuery;
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
  setStatus("Encoding query and searching...");
  try {
    const payload = await fetchJson(`/api/search?${params.toString()}`);
    payload.original_query = originalQuery;
    payload.clip_query = query;
    renderResults(payload);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    searchButton.disabled = false;
  }
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
  for (const result of currentPayload.results) {
    const state = reviewState.get(resultKey(result)) || {};
    rows.push([
      queryId,
      currentPayload.original_query || currentPayload.query || "",
      currentPayload.clip_query || currentPayload.query || "",
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

async function loadHealth() {
  try {
    const payload = await fetchJson("/api/health");
    const translation = payload.translation_configured ? "translation on" : "translation off";
    healthEl.textContent = `${payload.groups.join(", ")} · ${payload.index_vectors} vectors · ${payload.index_dim} dim · ${translation}`;
  } catch (error) {
    healthEl.textContent = error.message;
  }
}

form.addEventListener("submit", runSearch);
translateButton.addEventListener("click", translateQuery);
exportButton.addEventListener("click", exportCsv);
resultsEl.addEventListener("click", (event) => {
  if (event.target.classList.contains("judgement-button")) {
    updateJudgement(event.target);
  }
});
resultsEl.addEventListener("input", (event) => {
  if (event.target.classList.contains("note-input")) {
    updateNote(event.target);
  }
});
loadHealth();
