const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const topKInput = document.querySelector("#top-k");
const candidatePoolInput = document.querySelector("#candidate-pool");
const searchButton = document.querySelector("#search-button");
const statusEl = document.querySelector("#status");
const healthEl = document.querySelector("#health");
const resultsEl = document.querySelector("#results");
const resultTemplate = document.querySelector("#result-template");

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

function renderResults(payload) {
  resultsEl.replaceChildren();
  if (!payload.results.length) {
    setStatus("No results.");
    return;
  }
  setStatus(`${payload.results.length} results in ${payload.elapsed_ms.toFixed(1)} ms.`);
  for (const result of payload.results) {
    const node = resultTemplate.content.firstElementChild.cloneNode(true);
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
}

async function runSearch(event) {
  event.preventDefault();
  const query = queryInput.value.trim();
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
    renderResults(payload);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    searchButton.disabled = false;
  }
}

async function loadHealth() {
  try {
    const payload = await fetchJson("/api/health");
    healthEl.textContent = `${payload.groups.join(", ")} · ${payload.index_vectors} vectors · ${payload.index_dim} dim`;
  } catch (error) {
    healthEl.textContent = error.message;
  }
}

form.addEventListener("submit", runSearch);
loadHealth();
