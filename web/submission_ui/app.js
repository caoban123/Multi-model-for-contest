const $ = (id) => document.getElementById(id);

let session = null;
let agentResult = null;
let qaPayload = null;
let qaDraft = null;
let trakePayload = null;
let selectedTrakeChain = null;
let previewSequence = [];
let previewIndex = 0;
const previewRegistry = new Map();

async function post(route, payload) {
  const response = await fetch(route, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.code ? `${data.code}: ${data.error}` : (data.error || 'Request failed'));
  return data;
}

function message(value, error = false) {
  $('message').textContent = value;
  $('message').className = error ? 'message error' : 'message';
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}

function activeTask() {
  return document.querySelector('input[name="task"]:checked').value;
}

function renderSession(value) {
  session = value;
  $('session-status').textContent = value.status;
  $('session-id').textContent = value.session_id;
  $('validate-button').disabled = !value.queries.length;
  $('done-button').disabled = !value.queries.length || value.status !== 'ACTIVE';
  $('download-link').classList.toggle('disabled', value.status !== 'DONE');
  $('download-link').setAttribute('aria-disabled', String(value.status !== 'DONE'));
  $('download-link').href = value.status === 'DONE' ? `/api/submission/session/${encodeURIComponent(value.session_id)}/download` : '';
  $('queue-count').textContent = String(value.queries.length);
  renderQueue(value.queries);
  updateTaskControls();
}

function renderQueue(queries) {
  $('queue').replaceChildren();
  for (const item of queries) {
    const row = document.createElement('div');
    row.className = 'queue-item';
    row.innerHTML = `<strong>${escapeHtml(item.query_id)}</strong><span>${escapeHtml(item.task)} · ${item.predictions.length} prediction · ${escapeHtml(item.output_file)}</span><div class="queue-actions"><button type="button" data-preview-csv>Preview</button><button type="button" class="secondary" data-remove>Remove</button></div>`;
    row.querySelector('[data-preview-csv]').onclick = () => { $('csv-preview').value = item.csv_preview; };
    row.querySelector('[data-remove]').onclick = () => removeQuery(item.query_id);
    $('queue').appendChild(row);
  }
  if (!queries.length) $('queue').innerHTML = '<p class="empty">No confirmed query.</p>';
}

function resetReviewWorkspace() {
  agentResult = null;
  qaPayload = null;
  qaDraft = null;
  trakePayload = null;
  selectedTrakeChain = null;
  previewRegistry.clear();
  $('agent-trace').hidden = true;
  $('agent-inspector').hidden = true;
  $('task-workflow').hidden = true;
  $('workflow-content').replaceChildren();
  $('candidates').innerHTML = '<p class="empty">No candidate result.</p>';
  $('selection-count').textContent = '0 selected';
  $('confirm-button').disabled = true;
}

function updateTaskControls() {
  const task = activeTask();
  const isQa = task === 'QA';
  $('qa-question-wrap').hidden = !isQa;
  $('qa-question').required = isQa;
  $('gemini-answer-wrap').hidden = !isQa;
  $('confirm-bar').hidden = task !== 'KIS';
  $('run-button').textContent = task === 'KIS' ? 'Run Agent' : (isQa ? 'Prepare evidence' : 'Plan and search sequence');
  $('run-button').disabled = !session || session.status !== 'ACTIVE';
}

function renderPlanInspector(plan = {}, trace = {}, channelCounts = {}, structured = null) {
  $('agent-trace').hidden = false;
  const pills = [
    `Profile: ${plan.profile || 'baseline'}`,
    `Routes: ${(plan.enabled_retrievers || ['clip']).join(' + ')}`,
    `Fusion: ${(structured?.applied ? 'rrf + structured' : plan.fusion_method) || 'single channel'}`,
  ];
  for (const [channel, count] of Object.entries(channelCounts || {})) pills.push(`${channel.toUpperCase()}: ${count} hits`);
  if (structured?.applied) pills.push(`Structured: ${structured.structured_hit_count || 0} hits`);
  $('agent-trace').innerHTML = pills.map((item) => `<span class="trace-pill">${escapeHtml(item)}</span>`).join('');
  $('agent-inspector').hidden = false;
  $('agent-profile').textContent = `${plan.intent || 'mixed'} · ${plan.profile || 'baseline'} · ${plan.planner_source || 'local'}`;
  $('clip-route').textContent = plan.visual_clip_query_en || '-';
  $('bge-route').textContent = plan.semantic_text_query || '-';
  $('bm25-route').textContent = plan.lexical_text_query || '-';
  $('gemini-state').textContent = trace.status || 'Not called';
  $('gemini-state').className = `state-badge is-${String(trace.status || 'idle').toLowerCase()}`;
  $('gemini-model').textContent = [trace.provider, trace.model].filter(Boolean).join(' / ') || '-';
  $('gemini-latency').textContent = Number.isFinite(Number(trace.latency_ms)) ? `${Number(trace.latency_ms).toFixed(1)} ms` : '-';
  $('gemini-raw').textContent = trace.raw_text || 'No raw Gemini response. The deterministic planner was used.';
  $('gemini-parsed').textContent = trace.parsed_output ? JSON.stringify(trace.parsed_output, null, 2) : '-';
  $('validated-plan').textContent = JSON.stringify(trace.validated_plan || plan, null, 2);
  const warning = trace.fallback || (trace.warnings || []).join('; ') || structured?.error || structured?.warning;
  $('gemini-warning').hidden = !warning;
  $('gemini-warning').textContent = warning ? `Fallback / warning: ${warning}` : '';
}

function renderAgent(result) {
  agentResult = result;
  renderPlanInspector(result.agent_plan || {}, result.agent_trace || {}, result.channel_hit_counts || {}, result.structured_constraints);
  $('candidates').replaceChildren();
  for (const candidate of result.candidates) $('candidates').appendChild(candidateRow(candidate));
  if (!result.candidates.length) $('candidates').innerHTML = '<p class="empty">No candidate result.</p>';
  updateSelection();
}

function candidateEvidence(candidate) {
  const provenance = candidate.retrieval?.provenance || {};
  const ranks = provenance.retriever_ranks || {};
  const contributions = provenance.contributions || {};
  const chips = Object.entries(ranks).map(([name, rank]) => {
    const contribution = contributions[name];
    return `<span class="evidence-chip is-${escapeHtml(name)}">${escapeHtml(name.toUpperCase())} #${Number(rank)}${contribution !== undefined ? ` · +${Number(contribution).toFixed(4)}` : ''}</span>`;
  });
  if (candidate.source_type || candidate.matched_text) chips.push(`<span class="evidence-chip is-source">${escapeHtml(candidate.source_type || 'evidence')}${candidate.matched_text ? ` · ${escapeHtml(candidate.matched_text.slice(0, 120))}` : ''}</span>`);
  return chips.join('');
}

function registerPreview(item) {
  const id = `preview-${previewRegistry.size + 1}`;
  previewRegistry.set(id, item);
  return id;
}

function previewButton(item, className = 'frame-button') {
  if (!item.image_url) return '<div class="missing-image">Preview unavailable</div>';
  const id = registerPreview(item);
  return `<button class="${className}" type="button" data-open-preview="${id}" aria-label="Open ${escapeHtml(item.label)}"><img src="${escapeHtml(item.image_url)}" alt="${escapeHtml(item.label)}" loading="lazy"></button>`;
}

function candidateRow(candidate) {
  const row = document.createElement('article');
  row.className = 'candidate inspectable';
  row.dataset.candidate = JSON.stringify(candidate);
  const metadata = candidate.metadata || {};
  const image = previewButton({image_url: candidate.image_url, label: `${candidate.video_id} keyframe ${candidate.keyframe_id}`, meta: `${candidate.video_id} · keyframe ${candidate.keyframe_id} · frame ${candidate.frame_idx} · ${Number(candidate.pts_time || 0).toFixed(2)}s`}, 'candidate-image-button');
  const openVideo = candidate.video_url ? `<a class="candidate-action" href="${escapeHtml(candidate.video_url)}" target="_blank" rel="noreferrer">Open video</a>` : '';
  const neighborAction = Number.isInteger(Number(candidate.keyframe_id))
    ? `<button class="secondary" type="button" data-neighbor data-video="${escapeHtml(candidate.video_id)}" data-keyframe="${Number(candidate.keyframe_id)}">Nearby frames</button>`
    : '';
  row.innerHTML = `
    <div class="candidate-media">${image}<span class="rank-badge">#${Number(candidate.rank)}</span><span class="zoom-hint">Click to inspect</span></div>
    <div class="candidate-main">
      <div class="candidate-title"><strong>${escapeHtml(candidate.video_id)}</strong><span>score ${Number(candidate.retrieval?.score || 0).toFixed(4)}</span></div>
      <p>${escapeHtml(metadata.title || 'Untitled video')}</p>
      <dl class="candidate-facts"><div><dt>Keyframe</dt><dd>${escapeHtml(candidate.keyframe_id)}</dd></div><div><dt>Frame index</dt><dd>${escapeHtml(candidate.frame_idx)}</dd></div><div><dt>Timestamp</dt><dd>${Number(candidate.pts_time || 0).toFixed(2)}s</dd></div><div><dt>Author</dt><dd>${escapeHtml(metadata.author || '-')}</dd></div></dl>
      <div class="candidate-evidence">${candidateEvidence(candidate)}</div>
      <div class="candidate-actions">${openVideo}${neighborAction}<button class="secondary" type="button" data-copy-candidate>Copy details</button></div>
    </div>
    <aside class="candidate-review"><label class="select-control"><input class="candidate-select" type="checkbox"><span>Select candidate</span></label><label class="official-input"><span>Verified official frame ID</span><input type="number" min="0" step="1" placeholder="Required before confirm"></label><small>Keyframe and frame index above are retrieval diagnostics only.</small></aside>
    <div class="neighbors" hidden></div>`;
  row.querySelector('.candidate-select').setAttribute('aria-label', `Select ${candidate.video_id}`);
  row.querySelector('.candidate-select').onchange = updateSelection;
  row.querySelector('.official-input input').oninput = updateSelection;
  wireReviewActions(row);
  return row;
}

async function toggleNeighbors(button) {
  const panel = button.closest('.inspectable').querySelector('.neighbors');
  if (!panel.hidden) {
    panel.hidden = true;
    button.textContent = 'Nearby frames';
    return;
  }
  panel.hidden = false;
  button.textContent = 'Hide nearby';
  await loadNeighbors(panel, button.dataset.video, Number(button.dataset.keyframe), Number(panel.dataset.radius || 6));
}

async function loadNeighbors(panel, videoId, keyframeId, radius) {
  panel.dataset.radius = String(radius);
  panel.innerHTML = '<span class="empty">Loading neighboring frames...</span>';
  try {
    const params = new URLSearchParams({video_id: videoId, keyframe_id: String(keyframeId), radius: String(radius)});
    const response = await fetch(`/api/neighborhood?${params}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Neighborhood failed');
    const sequence = `${videoId}:${keyframeId}:${radius}`;
    const ids = data.frames.map((frame) => registerPreview({image_url: frame.image_url, label: `${videoId} keyframe ${frame.keyframe_id}`, meta: `${videoId} · keyframe ${frame.keyframe_id} · frame ${frame.frame_idx} · ${Number(frame.pts_time || 0).toFixed(2)}s`, sequence}));
    panel.innerHTML = `<div class="neighbor-toolbar"><strong>${escapeHtml(videoId)} neighborhood</strong><label>Radius <select data-radius><option value="3">3</option><option value="6">6</option><option value="12">12</option><option value="20">20</option></select></label><span>${data.frames.length} frames</span></div><div class="neighbor-track">${data.frames.map((frame, index) => `<button class="neighbor${Number(frame.keyframe_id) === keyframeId ? ' is-current' : ''}" type="button" data-open-preview="${ids[index]}"><img src="${escapeHtml(frame.image_url || '')}" alt="frame ${Number(frame.keyframe_id)}"><small>K${Number(frame.keyframe_id)} · F${Number(frame.frame_idx)} · ${Number(frame.pts_time || 0).toFixed(1)}s</small></button>`).join('')}</div>`;
    const select = panel.querySelector('[data-radius]');
    select.value = String(radius);
    select.onchange = () => loadNeighbors(panel, videoId, keyframeId, Number(select.value));
    wireReviewActions(panel);
  } catch (error) {
    panel.innerHTML = `<span class="empty">${escapeHtml(error.message)}</span>`;
  }
}

function wireReviewActions(root) {
  root.querySelectorAll('[data-open-preview]').forEach((button) => { button.onclick = () => openPreviewById(button.dataset.openPreview); });
  root.querySelectorAll('[data-neighbor]').forEach((button) => { button.onclick = () => toggleNeighbors(button); });
  root.querySelectorAll('[data-copy-candidate]').forEach((button) => {
    button.onclick = async () => {
      const candidate = JSON.parse(button.closest('.candidate').dataset.candidate);
      await copyText(`${candidate.video_id}, keyframe ${candidate.keyframe_id}, frame ${candidate.frame_idx}, ${Number(candidate.pts_time || 0).toFixed(2)}s`);
      message('Candidate details copied.');
    };
  });
}

function openPreviewById(id) {
  const selected = previewRegistry.get(id);
  if (!selected) return;
  previewSequence = selected.sequence ? [...previewRegistry.values()].filter((item) => item.sequence === selected.sequence) : [selected];
  previewIndex = Math.max(0, previewSequence.indexOf(selected));
  renderPreview();
  $('image-dialog').showModal();
}

function renderPreview() {
  const item = previewSequence[previewIndex];
  if (!item) return;
  $('image-preview').src = item.image_url;
  $('image-preview').alt = item.label;
  $('image-dialog-title').textContent = item.label;
  $('image-meta').textContent = item.meta || item.label;
  $('image-previous').disabled = previewSequence.length < 2;
  $('image-next').disabled = previewSequence.length < 2;
}

function movePreview(offset) {
  if (!previewSequence.length) return;
  previewIndex = (previewIndex + offset + previewSequence.length) % previewSequence.length;
  renderPreview();
}

async function copyText(value) {
  if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(value);
  const area = document.createElement('textarea');
  area.value = value;
  document.body.appendChild(area);
  area.select();
  document.execCommand('copy');
  area.remove();
}

function selectedPredictions() {
  return [...document.querySelectorAll('.candidate')].filter((row) => row.querySelector('.candidate-select').checked).map((row) => {
    const candidate = JSON.parse(row.dataset.candidate);
    const value = row.querySelector('.official-input input').value;
    return {video_id: candidate.video_id, frame_id: value === '' ? null : Number(value), mapping_source: 'manual_official', candidate_id: candidate.candidate_id};
  });
}

function updateSelection() {
  const rows = selectedPredictions();
  const valid = rows.length > 0 && rows.every((row) => Number.isInteger(row.frame_id) && row.frame_id >= 0);
  $('selection-count').textContent = `${rows.length} selected`;
  $('confirm-button').disabled = !session || session.status !== 'ACTIVE' || !valid;
}

function frameImageUrl(value) {
  const frame = Array.isArray(value?.frames) ? value.frames[0] : value;
  const path = frame?.keyframe_path || frame?.best_keyframe_path || frame?.payload?.keyframe_path;
  return frame?.image_url || (path ? `/keyframe?path=${encodeURIComponent(path)}` : '');
}

function evidenceText(evidence) {
  const payload = evidence.payload || {};
  return payload.matched_text || payload.text_raw || payload.text || payload.value || payload.title || evidence.source || 'source evidence';
}

function renderQaWorkflow(selectedIds = []) {
  const pack = qaPayload.evidence_pack || {};
  const selected = new Set(selectedIds);
  $('task-workflow').hidden = false;
  $('workflow-title').textContent = 'Q&A evidence review';
  $('workflow-state').textContent = qaDraft ? (qaDraft.confidence_state || 'DRAFT') : 'EVIDENCE';
  const evidence = pack.evidence_refs || [];
  const evidenceRows = evidence.map((item) => {
    const preview = previewButton({image_url: frameImageUrl(item), label: `${item.video_id} ${item.modality}`, meta: `${item.video_id} · ${item.modality} · keyframe ${item.keyframe_id ?? '-'} · frame ${item.frame_idx ?? '-'}`}, 'evidence-image-button');
    const neighbor = item.keyframe_id === null || item.keyframe_id === undefined ? '' : `<button class="secondary" type="button" data-neighbor data-video="${escapeHtml(item.video_id)}" data-keyframe="${Number(item.keyframe_id)}">Nearby frames</button>`;
    const frameLevel = item.keyframe_id !== null && item.keyframe_id !== undefined;
    return `<article class="evidence-row inspectable"><label class="evidence-select"><input type="checkbox" data-evidence-id="${escapeHtml(item.evidence_id)}" data-frame-level="${frameLevel}" ${selected.has(item.evidence_id) ? 'checked' : ''}><span>${escapeHtml(item.modality)} · ${escapeHtml(item.video_id)} · K${item.keyframe_id ?? '-'} · F${item.frame_idx ?? '-'}</span></label>${preview}<p>${escapeHtml(evidenceText(item))}</p><div class="candidate-actions">${neighbor}</div><div class="neighbors" hidden></div></article>`;
  }).join('');
  const answer = qaDraft?.raw_answer || '';
  $('workflow-content').innerHTML = `<div class="workflow-summary"><span>${(pack.candidates || []).length} candidate videos</span><span>${evidence.length} evidence items</span><span>${escapeHtml(qaPayload.question_route?.question_type || 'UNKNOWN')}</span></div><div class="evidence-grid">${evidenceRows || '<p class="empty">No source-backed evidence is available.</p>'}</div><div class="workflow-actions"><button id="qa-draft-button" type="button">Draft answer from selected evidence</button><label><span>Final answer</span><input id="qa-final-answer" value="${escapeHtml(answer)}" placeholder="Review or edit the proposed answer"></label><label><span>Verified official frame ID</span><input id="qa-official-frame" type="number" min="0" step="1" placeholder="Required"></label><button id="qa-confirm-button" type="button" disabled>Review and add to queue</button></div><details class="workflow-details"><summary>Draft diagnostics</summary><pre>${escapeHtml(JSON.stringify(qaDraft || {status: 'No draft yet'}, null, 2))}</pre></details>`;
  $('workflow-content').querySelectorAll('[data-evidence-id]').forEach((input) => { input.onchange = updateQaButtons; });
  $('qa-draft-button').onclick = draftQaAnswer;
  $('qa-final-answer').oninput = updateQaButtons;
  $('qa-official-frame').oninput = updateQaButtons;
  $('qa-confirm-button').onclick = confirmQaToQueue;
  wireReviewActions($('workflow-content'));
  updateQaButtons();
}

function selectedQaEvidenceIds() {
  return [...$('workflow-content').querySelectorAll('[data-evidence-id]:checked')].map((input) => input.dataset.evidenceId);
}

function updateQaButtons() {
  const selected = selectedQaEvidenceIds();
  const hasFrameEvidence = [...$('workflow-content').querySelectorAll('[data-evidence-id]:checked')].some((input) => input.dataset.frameLevel === 'true');
  if ($('qa-draft-button')) $('qa-draft-button').disabled = !selected.length;
  const frameValue = Number($('qa-official-frame')?.value);
  const finalAnswer = $('qa-final-answer')?.value.trim();
  if ($('qa-confirm-button')) $('qa-confirm-button').disabled = !qaDraft || !qaDraft.raw_answer || !selected.length || !hasFrameEvidence || !finalAnswer || !Number.isInteger(frameValue) || frameValue < 0 || $('qa-official-frame').value === '';
}

async function prepareQa() {
  qaPayload = await post('/api/qa/prepare', {query_id: $('query-id').value.trim(), event_query: $('query').value.trim(), question: $('qa-question').value.trim(), use_hybrid_retrieval: $('use-hybrid').checked, use_gemini_planner: $('use-gemini').checked});
  qaDraft = null;
  const context = qaPayload.evidence_pack?.retrieval_context || {};
  renderPlanInspector(context.query_plan || {}, context.agent_trace || {}, context.channel_hit_counts || {}, context.structured_constraints);
  renderQaWorkflow();
  $('candidates').innerHTML = '';
  message('Q&A evidence is ready. Select source evidence, then draft and review the answer.');
}

async function draftQaAnswer() {
  const selectedIds = selectedQaEvidenceIds();
  qaPayload = await post('/api/qa/draft-answer', {session_id: qaPayload.session.session_id, selected_evidence_ids: selectedIds, answer_method: $('use-gemini-answer').checked ? 'gemini' : 'evidence_first'});
  qaDraft = qaPayload.answer_draft;
  renderQaWorkflow(qaDraft.evidence_refs || selectedIds);
  message(qaDraft.raw_answer ? 'Answer draft created. Review it before confirmation.' : 'No supported answer was produced from the selected evidence.', !qaDraft.raw_answer);
}

async function confirmQaToQueue() {
  const selectedIds = selectedQaEvidenceIds();
  const finalAnswer = $('qa-final-answer').value.trim();
  const reviewed = await post('/api/qa/review', {session_id: qaPayload.session.session_id, draft_id: qaDraft.draft_id, decision: finalAnswer === qaDraft.raw_answer ? 'confirmed' : 'edited', final_answer: finalAnswer, selected_evidence_ids: selectedIds, reviewer: 'submission-ui-reviewer'});
  const imported = await post('/api/submission/import/qa', {session_id: session.session_id, qa_session_id: qaPayload.session.session_id, review_id: reviewed.review.review_id, official_frame_id: Number($('qa-official-frame').value), mapping_source: 'manual_official'});
  renderSession(imported);
  $('csv-preview').value = imported.queries.find((item) => item.query_id === $('query-id').value.trim())?.csv_preview || '';
  message(`${$('query-id').value.trim()} Q&A result added to the official queue.`);
}

function allTrakeChains(state) {
  return (state.alignments || []).flatMap((alignment) => alignment.chains || []);
}

function trakeCandidatePreview(candidate, eventId) {
  const preview = previewButton({image_url: frameImageUrl(candidate), label: `${candidate.video_id} ${eventId}`, meta: `${candidate.video_id} · ${eventId} · keyframe ${candidate.keyframe_id} · frame ${candidate.frame_idx} · ${Number(candidate.pts_time || 0).toFixed(2)}s`}, 'event-image-button');
  return `<div class="trake-event inspectable">${preview}<strong>${escapeHtml(eventId)}</strong><span>K${Number(candidate.keyframe_id)} · F${Number(candidate.frame_idx)} · ${Number(candidate.pts_time || 0).toFixed(2)}s</span><button class="secondary" type="button" data-neighbor data-video="${escapeHtml(candidate.video_id)}" data-keyframe="${Number(candidate.keyframe_id)}">Nearby frames</button><div class="neighbors" hidden></div></div>`;
}

function renderTrakeWorkflow() {
  const state = trakePayload.state;
  const chains = allTrakeChains(state);
  if (!selectedTrakeChain || !chains.some((item) => item.chain_id === selectedTrakeChain.chain_id)) selectedTrakeChain = chains[0] || null;
  $('task-workflow').hidden = false;
  $('workflow-title').textContent = 'TRAKE sequence review';
  $('workflow-state').textContent = selectedTrakeChain ? 'CHAIN READY' : 'NO CHAIN';
  const planRows = (state.request?.events || []).map((event) => `<li><strong>${escapeHtml(event.event_id)}</strong> ${escapeHtml(event.text)} <span>${escapeHtml((event.modalities || []).join(' + '))}</span></li>`).join('');
  const chainRows = chains.map((chain) => {
    const selected = chain.chain_id === selectedTrakeChain?.chain_id;
    const events = (chain.events || []).filter((event) => event.candidate).map((event) => trakeCandidatePreview(event.candidate, event.event_id)).join('');
    const score = chain.score_components?.final_rerank_score ?? chain.score?.final_score ?? 0;
    return `<article class="trake-chain${selected ? ' is-selected' : ''}"><label class="chain-select"><input type="radio" name="trake-chain" value="${escapeHtml(chain.chain_id)}" ${selected ? 'checked' : ''}><span>${escapeHtml(chain.video_id)} · score ${Number(score).toFixed(4)}</span></label><div class="trake-events">${events}</div><details><summary>Score and warnings</summary><pre>${escapeHtml(JSON.stringify({score: chain.score_components || chain.score, warnings: chain.warnings || []}, null, 2))}</pre></details></article>`;
  }).join('');
  const officialInputs = selectedTrakeChain ? selectedTrakeChain.events.filter((event) => event.candidate).map((event) => `<label><span>${escapeHtml(event.event_id)} official frame ID</span><input data-trake-frame="${escapeHtml(event.event_id)}" type="number" min="0" step="1" placeholder="Required"></label>`).join('') : '';
  const traces = state.diagnostics?.retrieval?.hybrid_query_traces || {};
  $('workflow-content').innerHTML = `<div class="workflow-notice">TRAKE currently validates L21 sequences. The Agent retrieves one video, then checks event order inside that video.</div><ol class="trake-plan">${planRows}</ol><div class="chain-list">${chainRows || '<p class="empty">No valid same-video temporal chain was found.</p>'}</div><div class="workflow-actions trake-submit">${officialInputs}<button id="trake-confirm-button" type="button" ${selectedTrakeChain ? '' : 'disabled'}>Review chain and add to queue</button></div><details class="workflow-details"><summary>Per-event Gemini planner traces</summary><pre>${escapeHtml(JSON.stringify(traces, null, 2))}</pre></details>`;
  $('workflow-content').querySelectorAll('input[name="trake-chain"]').forEach((input) => { input.onchange = () => { selectedTrakeChain = chains.find((chain) => chain.chain_id === input.value) || null; renderTrakeWorkflow(); }; });
  $('workflow-content').querySelectorAll('[data-trake-frame]').forEach((input) => { input.oninput = updateTrakeButton; });
  if ($('trake-confirm-button')) $('trake-confirm-button').onclick = confirmTrakeToQueue;
  wireReviewActions($('workflow-content'));
  updateTrakeButton();
}

function updateTrakeButton() {
  const inputs = [...$('workflow-content').querySelectorAll('[data-trake-frame]')];
  const valid = inputs.length > 0 && inputs.every((input) => input.value !== '' && Number.isInteger(Number(input.value)) && Number(input.value) >= 0);
  if ($('trake-confirm-button')) $('trake-confirm-button').disabled = !selectedTrakeChain || !valid;
}

async function prepareTrake() {
  message('TRAKE is decomposing events and retrieving candidate sequences...');
  const planned = await post('/api/trake/plan', {query_id: $('query-id').value.trim(), query: $('query').value.trim(), constraints: {hybrid_retrieval: $('use-hybrid').checked, hybrid_use_gemini: $('use-gemini').checked}});
  trakePayload = await post('/api/trake/search', {session_id: planned.state.session_id});
  try { trakePayload = await post('/api/trake/align', {session_id: planned.state.session_id}); }
  catch (error) { renderTrakeWorkflow(); throw error; }
  renderTrakeWorkflow();
  const plans = trakePayload.state.diagnostics?.retrieval?.hybrid_query_plans || {};
  const profiles = Object.values(plans).map((plan) => plan.profile).filter(Boolean);
  $('agent-trace').hidden = false;
  $('agent-trace').innerHTML = `<span class="trace-pill">Events: ${(trakePayload.state.request?.events || []).length}</span><span class="trace-pill">Routes: ${escapeHtml(profiles.join(' | ') || 'baseline')}</span><span class="trace-pill">Chains: ${allTrakeChains(trakePayload.state).length}</span>`;
  $('agent-inspector').hidden = true;
  $('candidates').innerHTML = '';
  message(selectedTrakeChain ? 'TRAKE chain is ready for frame review.' : 'TRAKE found no valid same-video sequence.', !selectedTrakeChain);
}

async function confirmTrakeToQueue() {
  const officialFrameIds = [...$('workflow-content').querySelectorAll('[data-trake-frame]')].map((input) => Number(input.value));
  const reviewed = await post('/api/trake/review', {session_id: trakePayload.state.session_id, chain_id: selectedTrakeChain.chain_id, decision: 'confirmed', reviewer: 'submission-ui-reviewer'});
  const imported = await post('/api/submission/import/trake', {session_id: session.session_id, trake_session_id: trakePayload.state.session_id, review_id: reviewed.review.review_id, official_frame_ids: officialFrameIds, mapping_source: 'manual_official'});
  renderSession(imported);
  $('csv-preview').value = imported.queries.find((item) => item.query_id === $('query-id').value.trim())?.csv_preview || '';
  message(`${$('query-id').value.trim()} TRAKE result added to the official queue.`);
}

$('start-button').onclick = async () => {
  try { renderSession(await post('/api/submission/session/start', {})); resetReviewWorkspace(); message('Submission session started.'); }
  catch (error) { message(error.message, true); }
};

$('agent-form').onsubmit = async (event) => {
  event.preventDefault();
  if (!session) return;
  $('run-button').disabled = true;
  previewRegistry.clear();
  try {
    const task = activeTask();
    if (task === 'KIS') {
      const result = await post('/api/submission/agent/run', {session_id: session.session_id, query_id: $('query-id').value.trim(), task: 'KIS', query: $('query').value.trim(), use_hybrid: $('use-hybrid').checked, use_gemini: $('use-gemini').checked});
      $('task-workflow').hidden = true;
      renderAgent(result);
      message(result.mapping_warning);
    } else if (task === 'QA') await prepareQa();
    else await prepareTrake();
  } catch (error) { message(error.message, true); }
  finally { $('run-button').disabled = !session || session.status !== 'ACTIVE'; }
};

$('confirm-button').onclick = async () => {
  try {
    const selected = selectedPredictions();
    const predictions = selected.map(({candidate_id, ...row}) => row);
    const source = {candidate_ids: selected.map((row) => row.candidate_id), agent_schema: agentResult.schema_version};
    renderSession(await post('/api/submission/query/confirm', {session_id: session.session_id, query_id: agentResult.query_id, task: 'KIS', predictions, source}));
    message(`${agentResult.query_id} confirmed to the official queue.`);
  } catch (error) { message(error.message, true); }
};

async function removeQuery(queryId) {
  try { renderSession(await post('/api/submission/query/remove', {session_id: session.session_id, query_id: queryId})); $('csv-preview').value = ''; message(`${queryId} removed.`); }
  catch (error) { message(error.message, true); }
}

$('validate-button').onclick = async () => {
  try { renderValidation(await post('/api/submission/session/validate', {session_id: session.session_id})); }
  catch (error) { message(error.message, true); }
};

function renderValidation(result) {
  $('validation-summary').textContent = result.valid ? `Valid · ${result.warning_count} warning` : `${result.error_count} error · ${result.warning_count} warning`;
  $('validation-summary').className = `validation-summary ${result.valid ? 'valid' : 'invalid'}`;
  $('issues').innerHTML = result.issues.map((issue) => `<div class="issue ${issue.severity.toLowerCase()}"><strong>${escapeHtml(issue.code)}</strong><br>${escapeHtml(issue.message)}</div>`).join('');
  if (!result.issues.length) $('issues').innerHTML = '<p class="empty">No validation issue.</p>';
  message(result.valid ? 'Submission queue is valid.' : 'Fix validation errors before Done.', !result.valid);
}

$('done-button').onclick = async () => {
  try { const result = await post('/api/submission/session/done', {session_id: session.session_id}); renderSession(result); renderValidation(result.validation); message('submission.zip is ready.'); }
  catch (error) { message(error.message, true); }
};

$('image-close').onclick = () => $('image-dialog').close();
$('image-previous').onclick = () => movePreview(-1);
$('image-next').onclick = () => movePreview(1);
$('copy-frame').onclick = async () => { await copyText($('image-meta').textContent); message('Frame details copied.'); };
$('image-dialog').addEventListener('click', (event) => { if (event.target === $('image-dialog')) $('image-dialog').close(); });
document.addEventListener('keydown', (event) => {
  if (!$('image-dialog').open) return;
  if (event.key === 'ArrowLeft') movePreview(-1);
  if (event.key === 'ArrowRight') movePreview(1);
});

for (const input of document.querySelectorAll('input[name="task"]')) input.addEventListener('change', () => { resetReviewWorkspace(); updateTaskControls(); });
updateTaskControls();
