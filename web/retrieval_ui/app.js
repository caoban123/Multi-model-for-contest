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
const videoRankingButton = document.querySelector("#video-ranking-button");
const frameRankingButton = document.querySelector("#frame-ranking-button");
const rankingTabs = document.querySelector("#ranking-tabs");
const topKLabel = document.querySelector("#top-k-label");
const statusEl = document.querySelector("#status");
const healthEl = document.querySelector("#health");
const resultsEl = document.querySelector("#results");
const pinsListEl = document.querySelector("#pins-list");
const historyListEl = document.querySelector("#history-list");
const pinsCountEl = document.querySelector("#pins-count");
const historyCountEl = document.querySelector("#history-count");
const resultTemplate = document.querySelector("#result-template");
const structuredEnabledInput = document.querySelector("#structured-enabled");
const debugModeInput = document.querySelector("#debug-mode");
const enableClipInput = document.querySelector("#enable-clip");
const enableObjectsInput = document.querySelector("#enable-objects");
const enableAttributesInput = document.querySelector("#enable-attributes");
const enableMetadataInput = document.querySelector("#enable-metadata");
const objectFields = document.querySelector("#object-fields");
const attributeFields = document.querySelector("#attribute-fields");
const metadataFields = document.querySelector("#metadata-fields");
const objectLabelInput = document.querySelector("#object-label");
const objectMinCountInput = document.querySelector("#object-min-count");
const objectPositionInput = document.querySelector("#object-position");
const objectMinConfidenceInput = document.querySelector("#object-min-confidence");
const objectFilterModeInput = document.querySelector("#object-filter-mode");
const attributeColorInput = document.querySelector("#attribute-color");
const attributeFilterModeInput = document.querySelector("#attribute-filter-mode");
const metadataAuthorInput = document.querySelector("#metadata-author");
const metadataDateInput = document.querySelector("#metadata-date");
const metadataTitleInput = document.querySelector("#metadata-title");
const metadataFilterModeInput = document.querySelector("#metadata-filter-mode");
const fusionMethodInput = document.querySelector("#fusion-method");
const structuredNoteEl = document.querySelector(".structured-note");
const videoDialog = document.querySelector("#video-dialog");
const videoPlayer = document.querySelector("#video-player");
const videoDialogTitle = document.querySelector("#video-dialog-title");
const videoDialogMeta = document.querySelector("#video-dialog-meta");
const closeVideoButton = document.querySelector("#close-video-button");
let currentPayload = null;
let reviewState = new Map();
let activeMode = "visual";
let activeRankingMode = "video";
let structuredSearchAvailable = false;
let attributeSearchAvailable = false;
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
  statusEl.replaceChildren();
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function setStatusPills(items) {
  statusEl.replaceChildren();
  statusEl.classList.remove("error");
  for (const item of items) {
    const pill = document.createElement("span");
    pill.className = "status-pill";
    pill.textContent = item;
    statusEl.appendChild(pill);
  }
}

function renderEmptyResults(title, message) {
  resultsEl.replaceChildren();
  resultsEl.classList.add("is-empty");
  const state = document.createElement("div");
  state.className = "empty-results";
  const content = document.createElement("div");
  const heading = document.createElement("h3");
  heading.textContent = title;
  const detail = document.createElement("p");
  detail.textContent = message;
  content.append(heading, detail);
  state.appendChild(content);
  resultsEl.appendChild(state);
}

function secondsLabel(value) {
  if (value === "" || value === null || value === undefined) {
    return "-";
  }
  const seconds = Number(value || 0);
  return `${seconds.toFixed(1)}s`;
}

function resultKey(result) {
  if (result.ranking_mode === "video") {
    return `video:${result.video_id}`;
  }
  return `${result.ranking_mode || "frame"}:${result.video_id}:${result.keyframe_id ?? "metadata"}`;
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

function structuredConfig() {
  return {
    enabled: structuredEnabledInput.checked,
    enable_clip: enableClipInput.checked,
    enable_objects: structuredSearchAvailable && enableObjectsInput.checked,
    enable_attributes: attributeSearchAvailable && enableAttributesInput.checked,
    enable_metadata: enableMetadataInput.checked,
    object_label: objectLabelInput.value.trim(),
    object_min_count: objectMinCountInput.value,
    object_position: objectPositionInput.value,
    object_min_confidence: objectMinConfidenceInput.value,
    object_filter_mode: objectFilterModeInput.value,
    attribute_color: attributeColorInput.value.trim(),
    attribute_filter_mode: attributeFilterModeInput.value,
    metadata_author: metadataAuthorInput.value.trim(),
    metadata_date: metadataDateInput.value.trim(),
    metadata_title: metadataTitleInput.value.trim(),
    metadata_filter_mode: metadataFilterModeInput.value,
    fusion_method: fusionMethodInput.value,
    debug_mode: debugModeInput.value,
  };
}

function applyStructuredConfig(config = {}) {
  structuredEnabledInput.checked = Boolean(config.enabled);
  enableClipInput.checked = config.enable_clip ?? true;
  enableObjectsInput.checked = Boolean(config.enable_objects);
  enableAttributesInput.checked = Boolean(config.enable_attributes);
  enableMetadataInput.checked = Boolean(config.enable_metadata);
  objectLabelInput.value = config.object_label || "";
  objectMinCountInput.value = config.object_min_count || "1";
  objectPositionInput.value = config.object_position || "any";
  objectMinConfidenceInput.value = config.object_min_confidence || "0.3";
  objectFilterModeInput.value = config.object_filter_mode || "soft";
  attributeColorInput.value = config.attribute_color || "";
  attributeFilterModeInput.value = config.attribute_filter_mode || "soft";
  metadataAuthorInput.value = config.metadata_author || "";
  metadataDateInput.value = config.metadata_date || "";
  metadataTitleInput.value = config.metadata_title || "";
  metadataFilterModeInput.value = config.metadata_filter_mode || "soft";
  fusionMethodInput.value = config.fusion_method || "rrf";
  debugModeInput.value = config.debug_mode || "custom";
  updateStructuredControls();
}

function updateStructuredControls() {
  const enabled = structuredEnabledInput.checked && activeMode === "visual";
  if (!structuredSearchAvailable && enableObjectsInput.checked) {
    enableObjectsInput.checked = false;
  }
  if (!attributeSearchAvailable && enableAttributesInput.checked) {
    enableAttributesInput.checked = false;
  }
  for (const input of [debugModeInput, enableClipInput, enableObjectsInput, enableAttributesInput, enableMetadataInput, fusionMethodInput]) {
    input.disabled = !enabled;
  }
  enableObjectsInput.disabled = !enabled || !structuredSearchAvailable;
  enableAttributesInput.disabled = !enabled || !attributeSearchAvailable;
  objectFields.disabled = !enabled || !structuredSearchAvailable || !enableObjectsInput.checked;
  attributeFields.disabled = !enabled || !attributeSearchAvailable || !enableAttributesInput.checked;
  metadataFields.disabled = !enabled || !enableMetadataInput.checked;
  structuredNoteEl.textContent = structuredSearchAvailable || attributeSearchAvailable
    ? "Structured mode is experimental. Turning it off keeps the Phase 3 CLIP search path unchanged."
    : "Structured mode is experimental. Build local evidence stores or install image support before enabling extra channels.";
}

function applyDebugPreset() {
  const preset = debugModeInput.value;
  if (preset === "custom") return;
  enableClipInput.checked = ["clip-only", "hybrid"].includes(preset);
  enableObjectsInput.checked = structuredSearchAvailable && ["object-only", "hybrid"].includes(preset);
  enableAttributesInput.checked = attributeSearchAvailable && ["hybrid"].includes(preset);
  enableMetadataInput.checked = ["metadata-only", "hybrid"].includes(preset);
  updateStructuredControls();
}

function displayedResults(payload) {
  if (payload.mode === "metadata") {
    return (payload.results || []).map((item) => normalizeResult(item, "metadata"));
  }
  if (activeRankingMode === "video") {
    return (payload.video_results || []).map(normalizeVideoResult);
  }
  return (payload.raw_results || payload.results || [])
    .slice(0, payload.top_k || topKInput.value)
    .map((item) => ({ ...item, ranking_mode: "frame" }));
}

function renderResults(payload, resetReview = true) {
  currentPayload = payload;
  if (resetReview) {
    reviewState = new Map();
  }
  resultsEl.replaceChildren();
  resultsEl.classList.remove("is-empty");
  exportButton.disabled = true;
  const results = displayedResults(payload);
  if (!results.length) {
    setStatus("No matching results.");
    renderEmptyResults("No matching results", "Try another query or increase Candidate Pool.");
    return;
  }
  const resultKind = payload.mode === "metadata"
    ? "metadata results"
    : activeRankingMode === "video" ? "videos" : "frames";
  const candidates = (payload.raw_results || []).length || results.length;
  const statusItems = [`${results.length} ${resultKind}`];
  if (payload.mode === "visual") {
    statusItems.push(`Pool ${candidates}`);
    statusItems.push(`Retrieval ${Number(payload.retrieval_ms || 0).toFixed(1)} ms`);
    statusItems.push(`Aggregation ${Number(payload.aggregation_ms || 0).toFixed(1)} ms`);
  } else if (payload.mode === "structured") {
    statusItems.push(`Union ${candidates}`);
    statusItems.push(`Candidates ${Number(payload.candidate_generation_ms || 0).toFixed(1)} ms`);
    statusItems.push(`RRF ${Number(payload.fusion_ms || 0).toFixed(1)} ms`);
  } else {
    statusItems.push(`${Number(payload.elapsed_ms || 0).toFixed(1)} ms`);
  }
  setStatusPills(statusItems);
  for (const result of results) {
    const node = resultTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.resultKey = resultKey(result);
    node.dataset.result = JSON.stringify(result);
    const thumbWrap = node.querySelector(".thumb-wrap");
    const img = node.querySelector(".thumb");
    node.querySelector(".rank").textContent = `#${result.rank}`;
    node.querySelector(".video-id").textContent = result.video_id;
    node.querySelector(".score-label").textContent = result.ranking_mode === "video" ? "Video score" : "Frame score";
    node.querySelector(".score").textContent = Number(result.score).toFixed(4);
    node.querySelector(".title").textContent = result.metadata.title || "Untitled";
    node.querySelector(".keyframe").textContent = `${result.keyframe_id ?? "-"}`;
    node.querySelector(".keyframe-label").textContent = result.ranking_mode === "video" ? "Best keyframe" : "Keyframe";
    node.querySelector(".time").textContent = secondsLabel(result.pts_time);
    const matchedFact = node.querySelector(".matched-fact");
    matchedFact.hidden = result.ranking_mode !== "video";
    node.querySelector(".matched-count").textContent = result.ranking_mode === "video"
      ? `${result.frame_count} pool / ${result.matched_frame_count} shown`
      : "-";
    node.querySelector(".author").textContent = result.metadata.author || "-";
    node.querySelector(".date").textContent = result.metadata.publish_date || "-";
    node.querySelector(".missing-detail").textContent = result.keyframe_id === null || result.keyframe_id === undefined
      ? "No keyframe for this result"
      : `Keyframe ${result.keyframe_id}`;

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
    renderEvidenceChips(node.querySelector(".evidence-chips"), result);
    const pinButton = node.querySelector(".pin-button");
    const isPinned = pinnedResults.some((item) => item.key === resultKey(result));
    pinButton.classList.toggle("is-active", isPinned);
    pinButton.setAttribute("aria-pressed", String(isPinned));
    pinButton.textContent = isPinned ? "Pinned" : "Pin";
    const openVideoButton = node.querySelector(".open-video-button");
    openVideoButton.hidden = !result.video_url;
    const neighborhoodButton = node.querySelector(".neighborhood-button");
    const canShowNeighborhood = result.ranking_mode === "frame" && result.keyframe_id !== null && result.keyframe_id !== undefined;
    neighborhoodButton.hidden = !canShowNeighborhood;
    const exploreButton = node.querySelector(".explore-button");
    exploreButton.hidden = result.ranking_mode !== "video";
    resultsEl.appendChild(node);
  }
  exportButton.disabled = false;
}

function normalizeResult(result, mode) {
  if (mode !== "metadata") {
    return { ...result, ranking_mode: "frame" };
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
    video_url: result.video_url || null,
    ranking_mode: "metadata",
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

function renderEvidenceChips(container, result) {
  container.replaceChildren();
  const evidence = result.evidence;
  if (!evidence?.fusion) return;
  const definitions = [];
  if (evidence.clip.enabled) {
    definitions.push({ status: evidence.clip.status, text: evidence.clip.status === "matched"
      ? `CLIP #${evidence.clip.rank} · ${Number(evidence.clip.score || 0).toFixed(4)}`
      : `CLIP · ${evidence.clip.status}` });
  }
  if (evidence.objects.enabled) {
    definitions.push({ status: evidence.objects.status, text: evidence.objects.status === "unknown"
      ? "Object evidence unavailable"
      : evidence.objects.status === "matched" ? `Object #${evidence.objects.rank} · matched` : "Object · not matched" });
  }
  if (evidence.attributes?.enabled) {
    const matched = evidence.attributes.matches || [];
    const colors = [...new Set(matched.map((item) => item.color).filter(Boolean))].join("+");
    definitions.push({ status: evidence.attributes.status, text: evidence.attributes.status === "matched"
      ? `Attribute #${evidence.attributes.rank} - ${colors || "matched"}`
      : `Attribute - ${evidence.attributes.status}` });
  }
  if (evidence.metadata.enabled) {
    definitions.push({ status: evidence.metadata.status, text: evidence.metadata.status === "matched"
      ? `Metadata #${evidence.metadata.rank} · ${evidence.metadata.matched_fields.map((item) => item.field).join("+") || "matched"}`
      : `Metadata · ${evidence.metadata.status}` });
  }
  definitions.push({ status: "matched", text: `Final #${result.rank} · RRF` });
  for (const definition of definitions) {
    const chip = document.createElement("span");
    chip.className = `evidence-chip is-${definition.status}`;
    chip.textContent = definition.text;
    container.appendChild(chip);
  }
}

function normalizeVideoResult(result) {
  return {
    ...result,
    ranking_mode: "video",
    score: result.video_score,
    keyframe_id: result.best_keyframe_id,
    frame_idx: result.best_frame_idx,
    pts_time: result.best_pts_time,
    keyframe_path: result.best_keyframe_path,
  };
}

function openVideo(target) {
  const card = target.closest(".result-card");
  const result = JSON.parse(card.dataset.result);
  openVideoResult(result);
}

function openVideoResult(result) {
  if (!result.video_url) {
    setStatus("Raw video is unavailable for this result.", true);
    return;
  }
  const startTime = Number(result.pts_time || 0);
  videoDialogTitle.textContent = `${result.video_id} · ${result.keyframe_id ?? "video"}`;
  videoDialogMeta.textContent = Number.isFinite(startTime) && startTime > 0
    ? `Opening around ${secondsLabel(startTime)}`
    : "Opening from the beginning";
  if (videoDialog.open) {
    closeVideo();
  }
  videoPlayer.src = result.video_url;
  videoPlayer.addEventListener("loadedmetadata", () => {
    if (Number.isFinite(startTime) && startTime > 0 && startTime < videoPlayer.duration) {
      videoPlayer.currentTime = startTime;
    }
  }, { once: true });
  videoDialog.showModal();
}

function closeVideo() {
  videoPlayer.pause();
  videoPlayer.removeAttribute("src");
  videoPlayer.load();
  if (videoDialog.open) {
    videoDialog.close();
  }
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
    matched_frames_per_video: "5",
    aggregation_method: "max",
  });
  const structured = activeMode === "visual" ? structuredConfig() : { enabled: false };
  if (structured.enabled) {
    if (!structured.enable_clip && !structured.enable_objects && !structured.enable_attributes && !structured.enable_metadata) {
      setStatus("Enable at least one structured retrieval channel.", true);
      return;
    }
    for (const [key, value] of Object.entries(structured)) {
      if (key !== "enabled" && key !== "debug_mode" && value !== "") params.set(key, String(value));
    }
    params.set("matched_frames_per_video", "5");
    params.delete("max_frames_per_video");
    params.delete("aggregation_method");
  }
  searchButton.disabled = true;
  const searchButtonLabel = searchButton.innerHTML;
  searchButton.textContent = activeMode === "metadata" ? "Searching metadata..." : "Searching...";
  setStatus(activeMode === "metadata" ? "Searching metadata..." : "Encoding query and searching...");
  try {
    const endpoint = activeMode === "metadata"
      ? "/api/metadata-search"
      : structured.enabled ? "/api/structured-search" : "/api/search";
    if (activeMode === "metadata") {
      params.delete("candidate_pool");
      params.delete("max_frames_per_video");
      params.delete("matched_frames_per_video");
      params.delete("aggregation_method");
    }
    const payload = await fetchJson(`${endpoint}?${params.toString()}`);
    payload.original_query = originalQuery;
    payload.clip_query = activeMode === "visual" ? query : "";
    payload.mode = payload.mode || activeMode;
    saveHistoryItem(originalQuery, payload.clip_query, activeMode, structured.enabled ? structured : null, payload.fusion_config || null);
    renderResults(payload);
    renderHistory();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    searchButton.disabled = false;
    searchButton.innerHTML = searchButtonLabel;
  }
}

function saveHistoryItem(originalQuery, clipQuery, mode, structured = null, fusionConfig = null) {
  const item = {
    mode,
    retrieval_mode: structured ? "structured" : mode,
    original_query: originalQuery,
    clip_query: clipQuery,
    structured_config: structured,
    fusion_config: fusionConfig,
    created_at: new Date().toISOString(),
  };
  searchHistory = [
    item,
    ...searchHistory.filter(
      (entry) => entry.mode !== mode || entry.original_query !== originalQuery || entry.clip_query !== clipQuery || JSON.stringify(entry.structured_config) !== JSON.stringify(structured)
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
  const translateButtonLabel = translateButton.textContent;
  translateButton.textContent = "Translating...";
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
    translateButton.textContent = translateButtonLabel;
  }
}

function updateJudgement(target) {
  const card = target.closest(".result-card");
  const key = card.dataset.resultKey;
  const state = reviewState.get(key) || {};
  state.manual_judgement = target.dataset.judgement;
  reviewState.set(key, state);
  syncPinnedReview(key, state);
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
  syncPinnedReview(key, state);
}

function syncPinnedReview(key, state) {
  const pinned = pinnedResults.find((item) => item.key === key);
  if (!pinned) {
    return;
  }
  pinned.manual_judgement = state.manual_judgement || "";
  pinned.manual_notes = state.manual_notes || "";
  saveStoredArray(STORAGE_KEYS.pins, pinnedResults);
  renderPins();
}

function togglePin(target) {
  const card = target.closest(".result-card");
  const result = JSON.parse(card.dataset.result);
  const key = resultKey(result);
  const state = reviewState.get(key) || {};
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
        ranking_mode: result.ranking_mode || activeRankingMode,
        rank: result.rank ?? "",
        manual_judgement: state.manual_judgement || "",
        manual_notes: state.manual_notes || "",
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

async function toggleNeighborhood(target) {
  const card = target.closest(".result-card");
  const panel = card.querySelector(".neighborhood-panel");
  if (!panel.hidden) {
    panel.hidden = true;
    target.setAttribute("aria-expanded", "false");
    return;
  }

  const result = JSON.parse(card.dataset.result);
  target.disabled = true;
  panel.hidden = false;
  panel.replaceChildren();
  panel.appendChild(emptyNeighborhoodState("Loading nearby keyframes..."));
  try {
    const params = new URLSearchParams({
      video_id: result.video_id,
      keyframe_id: String(result.keyframe_id),
      radius: "3",
    });
    const payload = await fetchJson(`/api/neighborhood?${params.toString()}`);
    renderNeighborhood(panel, payload);
    target.setAttribute("aria-expanded", "true");
  } catch (error) {
    panel.replaceChildren();
    panel.appendChild(emptyNeighborhoodState(error.message));
  } finally {
    target.disabled = false;
  }
}

function toggleExplore(target) {
  const card = target.closest(".result-card");
  const panel = card.querySelector(".explore-panel");
  if (!panel.hidden) {
    panel.hidden = true;
    target.setAttribute("aria-expanded", "false");
    target.textContent = "Explore video";
    return;
  }

  const result = JSON.parse(card.dataset.result);
  renderExplorePanel(panel, result);
  panel.hidden = false;
  target.setAttribute("aria-expanded", "true");
  target.textContent = "Hide video details";
}

function renderExplorePanel(panel, result) {
  panel.replaceChildren();
  const heading = document.createElement("h3");
  heading.className = "explore-heading";
  heading.textContent = `Matched frames (${result.frame_count} in candidate pool)`;
  const note = document.createElement("p");
  note.className = "explore-note";
  note.textContent = "Matched frames scored against the query. Timeline neighbors are contextual frames and may not match the query.";
  const list = document.createElement("div");
  list.className = "matched-frame-list";
  for (const frame of result.frames || []) {
    list.appendChild(matchedFrameNode(result.video_id, frame));
  }
  if (!list.children.length) {
    list.appendChild(emptyNeighborhoodState("No matched frames available."));
  }
  const timeline = document.createElement("div");
  timeline.className = "timeline-panel";
  timeline.appendChild(emptyNeighborhoodState("Choose Timeline on a matched frame to load temporal neighbors."));
  panel.append(heading, note, list, timeline);
}

function matchedFrameNode(videoId, frame) {
  const node = document.createElement("div");
  node.className = "matched-frame";
  node.classList.toggle("is-representative", Boolean(frame.is_representative));
  const thumb = document.createElement("div");
  thumb.className = "matched-frame-thumb";
  if (frame.image_url) {
    const img = document.createElement("img");
    img.src = frame.image_url;
    img.alt = `${videoId} keyframe ${frame.keyframe_id}`;
    img.addEventListener("error", () => {
      thumb.replaceChildren(emptyNeighborhoodState("No image"));
    }, { once: true });
    thumb.appendChild(img);
  } else {
    thumb.textContent = "No image";
  }
  const meta = document.createElement("div");
  meta.className = "matched-frame-meta";
  const title = document.createElement("strong");
  title.textContent = `${frame.is_representative ? "★ Best match · " : ""}Keyframe ${frame.keyframe_id}`;
  const detail = document.createElement("span");
  detail.textContent = `${secondsLabel(frame.pts_time)} · Frame score ${Number(frame.score).toFixed(4)} · Raw rank #${frame.rank}`;
  meta.append(title, detail);
  const timelineButton = document.createElement("button");
  timelineButton.className = "timeline-button secondary-button";
  timelineButton.type = "button";
  timelineButton.dataset.videoId = videoId;
  timelineButton.dataset.keyframeId = String(frame.keyframe_id);
  timelineButton.textContent = "Timeline";
  node.append(thumb, meta, timelineButton);
  return node;
}

async function loadMatchedFrameTimeline(target) {
  const panel = target.closest(".explore-panel");
  const timeline = panel.querySelector(".timeline-panel");
  target.disabled = true;
  timeline.replaceChildren(emptyNeighborhoodState("Loading timeline neighbors..."));
  try {
    const params = new URLSearchParams({
      video_id: target.dataset.videoId,
      keyframe_id: target.dataset.keyframeId,
      radius: "3",
    });
    const payload = await fetchJson(`/api/neighborhood?${params.toString()}`);
    renderNeighborhood(timeline, payload);
    const label = document.createElement("div");
    label.className = "timeline-label";
    label.textContent = `Timeline neighbors · center keyframe ${payload.keyframe_id}`;
    timeline.prepend(label);
  } catch (error) {
    timeline.replaceChildren(emptyNeighborhoodState(error.message));
  } finally {
    target.disabled = false;
  }
}

function renderNeighborhood(panel, payload) {
  panel.replaceChildren();
  const header = document.createElement("div");
  header.className = "neighborhood-head";
  header.textContent = `${payload.video_id} · ${payload.start_keyframe_id}-${payload.end_keyframe_id} / ${payload.total_frames}`;
  const strip = document.createElement("div");
  strip.className = "neighborhood-strip";
  for (const frame of payload.frames) {
    strip.appendChild(neighborhoodFrameNode(frame));
  }
  panel.append(header, strip);
}

function neighborhoodFrameNode(frame) {
  const node = document.createElement("div");
  node.className = "neighbor-frame";
  node.classList.toggle("is-center", Boolean(frame.is_center));
  const thumb = document.createElement("div");
  thumb.className = "neighbor-thumb";
  if (frame.image_url) {
    const img = document.createElement("img");
    img.src = frame.image_url;
    img.alt = `${frame.video_id} keyframe ${frame.keyframe_id}`;
    img.addEventListener("error", () => {
      thumb.replaceChildren(emptyNeighborhoodState("No image"));
    }, { once: true });
    thumb.appendChild(img);
  } else {
    thumb.appendChild(emptyNeighborhoodState("No image"));
  }
  const meta = document.createElement("div");
  meta.className = "neighbor-meta";
  meta.textContent = `#${frame.keyframe_id} · ${secondsLabel(frame.pts_time)}`;
  node.append(thumb, meta);
  return node;
}

function emptyNeighborhoodState(text) {
  const node = document.createElement("div");
  node.className = "neighborhood-empty";
  node.textContent = text;
  return node;
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (/[",\r\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function evidenceCsvFields(result) {
  const evidence = result.evidence || {};
  const objectMatches = evidence.objects?.matches || [];
  const attributeMatches = evidence.attributes?.matches || [];
  const matchedObjects = [...new Set(objectMatches.flatMap((item) => item.matched_labels || item.detections?.map((detection) => detection.label_normalized) || []))].join("|");
  const objectScores = objectMatches.map((item) => Number(item.object_score)).filter(Number.isFinite);
  const attributeScores = attributeMatches.map((item) => Number(item.attribute_score)).filter(Number.isFinite);
  return {
    clip_score: result.clip_score ?? evidence.clip?.score ?? "",
    clip_rank: result.clip_rank ?? evidence.clip?.rank ?? "",
    object_score: result.object_score ?? (objectScores.length ? Math.max(...objectScores) : ""),
    object_rank: result.object_rank ?? evidence.objects?.rank ?? "",
    matched_objects: matchedObjects,
    attribute_score: result.attribute_score ?? (attributeScores.length ? Math.max(...attributeScores) : ""),
    attribute_rank: result.attribute_rank ?? evidence.attributes?.rank ?? "",
    matched_attributes: attributeMatches.map((item) => `${item.color}:${item.target}:${item.color_ratio}`).join("|"),
    metadata_score: result.metadata_score ?? "",
    metadata_rank: result.metadata_rank ?? evidence.metadata?.rank ?? "",
    matched_metadata_fields: (evidence.metadata?.matched_fields || []).map((item) => item.field).join("|"),
    fusion_score: result.fusion_score ?? (evidence.fusion ? result.video_score ?? result.score : ""),
    fusion_rank: evidence.fusion ? result.rank : "",
    fusion_method: evidence.fusion?.method || currentPayload?.fusion_method || "",
  };
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
    "ranking_mode",
    "clip_query",
    "rank",
    "video_id",
    "keyframe_id",
    "frame_idx",
    "pts_time",
    "score",
    "video_score",
    "frame_count",
    "aggregation_method",
    "clip_score",
    "clip_rank",
    "object_score",
    "object_rank",
    "matched_objects",
    "attribute_score",
    "attribute_rank",
    "matched_attributes",
    "metadata_score",
    "metadata_rank",
    "matched_metadata_fields",
    "fusion_score",
    "fusion_rank",
    "fusion_method",
    "is_pinned",
    "pinned_at",
    "manual_judgement",
    "manual_notes",
    "title",
    "author",
    "publish_date",
    "video_url",
    "watch_url",
    "description_preview",
    "keyframe_path",
  ];
  const rows = [headers];
  for (const result of displayedResults(currentPayload)) {
    const state = reviewState.get(resultKey(result)) || {};
    const pinned = pinnedResults.find((item) => item.key === resultKey(result));
    const evidence = evidenceCsvFields(result);
    rows.push([
      queryId,
      currentPayload.original_query || currentPayload.query || "",
      currentPayload.mode || activeMode,
      result.ranking_mode || "frame",
      currentPayload.mode === "metadata" ? "" : currentPayload.clip_query || currentPayload.query || "",
      result.rank,
      result.video_id,
      result.keyframe_id,
      result.frame_idx,
      result.pts_time,
      result.score,
      result.video_score ?? "",
      result.frame_count ?? "",
      result.aggregation_method ?? "",
      evidence.clip_score,
      evidence.clip_rank,
      evidence.object_score,
      evidence.object_rank,
      evidence.matched_objects,
      evidence.attribute_score,
      evidence.attribute_rank,
      evidence.matched_attributes,
      evidence.metadata_score,
      evidence.metadata_rank,
      evidence.matched_metadata_fields,
      evidence.fusion_score,
      evidence.fusion_rank,
      evidence.fusion_method,
      pinned ? "1" : "0",
      pinned?.pinned_at || "",
      state.manual_judgement || "",
      state.manual_notes || "",
      result.metadata.title || "",
      result.metadata.author || "",
      result.metadata.publish_date || "",
      result.video_url || "",
      result.metadata.watch_url || "",
      result.metadata.description_preview || "",
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
    "ranking_mode",
    "clip_query",
    "video_id",
    "keyframe_id",
    "frame_idx",
    "pts_time",
    "score",
    "video_score",
    "frame_count",
    "clip_score",
    "clip_rank",
    "object_score",
    "object_rank",
    "matched_objects",
    "attribute_score",
    "attribute_rank",
    "matched_attributes",
    "metadata_score",
    "metadata_rank",
    "matched_metadata_fields",
    "fusion_score",
    "fusion_rank",
    "fusion_method",
    "manual_judgement",
    "manual_notes",
    "submission_video_id",
    "submission_keyframe_id",
    "submission_pts_time",
    "title",
    "author",
    "publish_date",
    "video_url",
    "watch_url",
    "description_preview",
    "keyframe_path",
  ];
  const rows = [headers];
  for (const item of pinnedResults) {
    const result = item.result;
    const evidence = evidenceCsvFields(result);
    rows.push([
      item.pinned_at,
      item.query_text,
      item.mode || "",
      result.ranking_mode || "frame",
      item.clip_query,
      result.video_id,
      result.keyframe_id,
      result.frame_idx,
      result.pts_time,
      result.score,
      result.video_score ?? "",
      result.frame_count ?? "",
      evidence.clip_score,
      evidence.clip_rank,
      evidence.object_score,
      evidence.object_rank,
      evidence.matched_objects,
      evidence.attribute_score,
      evidence.attribute_rank,
      evidence.matched_attributes,
      evidence.metadata_score,
      evidence.metadata_rank,
      evidence.matched_metadata_fields,
      evidence.fusion_score,
      evidence.fusion_rank,
      evidence.fusion_method,
      item.manual_judgement || "",
      item.manual_notes || "",
      result.video_id,
      result.keyframe_id,
      result.pts_time,
      result.metadata?.title || "",
      result.metadata?.author || "",
      result.metadata?.publish_date || "",
      result.video_url || "",
      result.metadata?.watch_url || "",
      result.metadata?.description_preview || "",
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
  pinsCountEl.textContent = String(pinnedResults.length);
  exportPinsButton.disabled = pinnedResults.length === 0;
  if (!pinnedResults.length) {
    pinsListEl.appendChild(emptyCompactItem("No pinned results yet."));
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
    const reviewLabel = item.manual_judgement ? ` - ${item.manual_judgement}` : "";
    meta.textContent = `${secondsLabel(result.pts_time)} - rank #${item.rank || result.rank || "-"}${reviewLabel} - ${result.metadata?.title || "Untitled"}`;
    const query = document.createElement("div");
    query.className = "compact-meta compact-query";
    query.textContent = item.query_text ? `Query: ${item.query_text}` : "Query unavailable";
    const actions = document.createElement("div");
    actions.className = "pin-actions";
    if (result.video_url) {
      const openVideo = document.createElement("button");
      openVideo.className = "secondary-button small-button";
      openVideo.type = "button";
      openVideo.textContent = "Open video";
      openVideo.addEventListener("click", () => openVideoResult(result));
      actions.appendChild(openVideo);
    }
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
    actions.appendChild(remove);
    node.append(title, meta, query, actions);
    pinsListEl.appendChild(node);
  }
}

function renderHistory() {
  historyListEl.replaceChildren();
  historyCountEl.textContent = String(searchHistory.length);
  if (!searchHistory.length) {
    historyListEl.appendChild(emptyCompactItem("No recent searches yet."));
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
      applyStructuredConfig(item.structured_config || {});
      queryInput.value = item.original_query || "";
      clipQueryInput.value = item.clip_query || "";
      form.requestSubmit();
    });
    const meta = document.createElement("div");
    meta.className = "compact-meta";
    const modalityLabel = item.structured_config
      ? [item.structured_config.enable_clip && "CLIP", item.structured_config.enable_objects && "Objects", item.structured_config.enable_attributes && "Attributes", item.structured_config.enable_metadata && "Metadata"].filter(Boolean).join(" + ")
      : item.mode || "visual";
    meta.textContent = item.clip_query ? `${item.retrieval_mode || item.mode || "visual"} · ${modalityLabel} · ${item.clip_query}` : modalityLabel;
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
  visualModeButton.setAttribute("aria-selected", String(isVisual));
  metadataModeButton.setAttribute("aria-selected", String(!isVisual));
  translateButton.disabled = !isVisual;
  clipQueryInput.disabled = !isVisual;
  candidatePoolInput.disabled = !isVisual;
  rankingTabs.hidden = !isVisual;
  updateStructuredControls();
  topKLabel.textContent = isVisual
    ? (activeRankingMode === "video" ? "Top videos" : "Top frames")
    : "Top results";
  setStatus(isVisual ? "Visual search ready." : "Metadata search ready.");
}

function setRankingMode(mode) {
  activeRankingMode = mode;
  const isVideo = mode === "video";
  videoRankingButton.classList.toggle("is-active", isVideo);
  frameRankingButton.classList.toggle("is-active", !isVideo);
  videoRankingButton.setAttribute("aria-pressed", String(isVideo));
  frameRankingButton.setAttribute("aria-pressed", String(!isVideo));
  videoRankingButton.setAttribute("aria-selected", String(isVideo));
  frameRankingButton.setAttribute("aria-selected", String(!isVideo));
  topKLabel.textContent = isVideo ? "Top videos" : "Top frames";
  if (currentPayload?.mode === "visual") {
    renderResults(currentPayload, false);
  } else {
    setStatus(isVideo ? "Video Ranking ready." : "Frame Ranking debug mode ready.");
  }
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
    const badges = [
      { text: payload.groups.join(", "), className: "is-primary" },
      { text: `${Number(payload.index_vectors).toLocaleString()} keyframes` },
        { text: `${payload.index_dim} dim` },
        { text: `${payload.metadata_documents} videos` },
        { text: payload.structured_search_available ? "objects on" : "objects off", className: payload.structured_search_available ? "is-success" : "" },
        { text: payload.attribute_search_available ? "attributes on" : "attributes off", className: payload.attribute_search_available ? "is-success" : "" },
        { text: translation, className: payload.translation_configured ? "is-success" : "" },
      ];
    structuredSearchAvailable = Boolean(payload.structured_search_available);
    attributeSearchAvailable = Boolean(payload.attribute_search_available);
    updateStructuredControls();
    healthEl.replaceChildren();
    for (const badge of badges) {
      const node = document.createElement("span");
      node.className = `health-badge ${badge.className || ""}`.trim();
      node.textContent = badge.text;
      healthEl.appendChild(node);
    }
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
videoRankingButton.addEventListener("click", () => setRankingMode("video"));
frameRankingButton.addEventListener("click", () => setRankingMode("frame"));
structuredEnabledInput.addEventListener("change", updateStructuredControls);
enableObjectsInput.addEventListener("change", () => { debugModeInput.value = "custom"; updateStructuredControls(); });
enableAttributesInput.addEventListener("change", () => { debugModeInput.value = "custom"; updateStructuredControls(); });
enableMetadataInput.addEventListener("change", () => { debugModeInput.value = "custom"; updateStructuredControls(); });
enableClipInput.addEventListener("change", () => { debugModeInput.value = "custom"; updateStructuredControls(); });
debugModeInput.addEventListener("change", applyDebugPreset);
resultsEl.addEventListener("click", (event) => {
  if (event.target.classList.contains("judgement-button")) {
    updateJudgement(event.target);
  }
  if (event.target.classList.contains("pin-button")) {
    togglePin(event.target);
  }
  if (event.target.classList.contains("open-video-button")) {
    openVideo(event.target);
  }
  if (event.target.classList.contains("neighborhood-button")) {
    toggleNeighborhood(event.target);
  }
  if (event.target.classList.contains("explore-button")) {
    toggleExplore(event.target);
  }
  if (event.target.classList.contains("timeline-button")) {
    loadMatchedFrameTimeline(event.target);
  }
});
resultsEl.addEventListener("input", (event) => {
  if (event.target.classList.contains("note-input")) {
    updateNote(event.target);
  }
});
closeVideoButton.addEventListener("click", closeVideo);
videoDialog.addEventListener("cancel", (event) => {
  event.preventDefault();
  closeVideo();
});
renderPins();
renderHistory();
updateStructuredControls();
renderEmptyResults("Ready to retrieve", "Enter a query above to explore ranked video moments.");
loadHealth();
