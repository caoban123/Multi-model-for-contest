const $ = (id) => document.getElementById(id);

let session = null;
let agentResult = null;
let qaPayload = null;
let qaDraft = null;
let activeQaVideoId = null;
let qaOfficialFrameId = '';
let trakePayload = null;
let selectedTrakeChain = null;
let trakeVerification = null;
let trakePlanDraft = null;
let trakePlanDraftRevision = null;
const trakeOfficialFrames = new Map();
const trakeRefinements = new Map();
let previewSequence = [];
let previewIndex = 0;
let focusedCsvQueryId = null;
const csvEditorStates = new Map();
const previewRegistry = new Map();

async function post(route, payload) {
  const response = await fetch(route, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.code ? `${data.code}: ${data.error}` : (data.error || 'Request failed'));
  return data;
}

async function get(route) {
  const response = await fetch(route);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed');
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

function csvStateFor(item) {
  let state = csvEditorStates.get(item.query_id);
  if (!state) {
    state = {queryId: item.query_id, filename: item.output_file, content: item.csv_preview, draft: item.csv_preview, dirty: false, open: false};
    csvEditorStates.set(item.query_id, state);
  } else if (!state.dirty) {
    state.filename = item.output_file;
    state.content = item.csv_preview;
    state.draft = item.csv_preview;
  }
  return state;
}

function updateSubmissionActions() {
  if (!session) return;
  const dirty = [...csvEditorStates.values()].some((state) => state.dirty);
  $('validate-button').disabled = !session.queries.length || dirty;
  $('done-button').disabled = !session.queries.length || session.status !== 'ACTIVE' || dirty;
}

function renderCsvPreviews(queries) {
  const validIds = new Set(queries.map((item) => item.query_id));
  for (const queryId of csvEditorStates.keys()) if (!validIds.has(queryId)) csvEditorStates.delete(queryId);
  if (focusedCsvQueryId && !validIds.has(focusedCsvQueryId)) focusedCsvQueryId = null;
  $('csv-file-count').textContent = String(queries.length);
  $('csv-preview-list').replaceChildren();
  for (const item of queries) {
    const state = csvStateFor(item);
    if (item.query_id === focusedCsvQueryId) state.open = true;
    const details = document.createElement('details');
    details.className = 'csv-file-item';
    details.dataset.queryId = item.query_id;
    details.open = state.open;
    details.innerHTML = `<summary><span><strong>${escapeHtml(item.output_file)}</strong><small>${escapeHtml(item.task)} · ${item.predictions.length} rows</small></span><span class="csv-edit-state">${state.dirty ? 'Unsaved' : 'Saved'}</span></summary><div class="csv-editor"><textarea aria-label="Edit ${escapeHtml(item.output_file)}" rows="${Math.min(12, Math.max(4, item.predictions.length + 1))}" spellcheck="false"></textarea><div class="preview-actions"><button class="secondary" type="button" data-reset-csv>Reset</button><button type="button" data-save-changes>Save changes</button><button class="secondary csv-download" type="button" data-save-csv>Download</button></div></div>`;
    const textarea = details.querySelector('textarea');
    const reset = details.querySelector('[data-reset-csv]');
    const saveChanges = details.querySelector('[data-save-changes]');
    const download = details.querySelector('[data-save-csv]');
    textarea.value = state.draft;
    textarea.readOnly = session.status !== 'ACTIVE';
    reset.disabled = !state.dirty;
    saveChanges.disabled = !state.dirty || session.status !== 'ACTIVE';
    download.disabled = state.dirty;
    details.ontoggle = () => {
      state.open = details.open;
      if (!details.open && focusedCsvQueryId === state.queryId) focusedCsvQueryId = null;
    };
    textarea.oninput = () => {
      state.draft = textarea.value;
      state.dirty = state.draft !== state.content;
      reset.disabled = !state.dirty;
      saveChanges.disabled = !state.dirty;
      download.disabled = state.dirty;
      details.querySelector('.csv-edit-state').textContent = state.dirty ? 'Unsaved' : 'Saved';
      updateSubmissionActions();
    };
    reset.onclick = () => resetCsvEdit(state.queryId);
    saveChanges.onclick = () => saveCsvChanges(state.queryId);
    download.onclick = () => saveCsv(state.queryId);
    $('csv-preview-list').appendChild(details);
  }
  if (!queries.length) $('csv-preview-list').innerHTML = '<p class="empty">No confirmed CSV.</p>';
  updateSubmissionActions();
}

function focusCsvPreview(queryId) {
  focusedCsvQueryId = queryId;
  renderCsvPreviews(session?.queries || []);
  const details = [...$('csv-preview-list').querySelectorAll('.csv-file-item')].find((item) => item.dataset.queryId === queryId);
  if (details) {
    details.open = true;
    details.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  }
}

function resetCsvEdit(queryId) {
  const item = session?.queries.find((row) => row.query_id === queryId);
  if (!item) return;
  const open = csvEditorStates.get(queryId)?.open || false;
  csvEditorStates.set(queryId, {queryId, filename: item.output_file, content: item.csv_preview, draft: item.csv_preview, dirty: false, open});
  renderCsvPreviews(session.queries);
  message(`${item.output_file} restored from the queue.`);
}

async function saveCsvChanges(queryId) {
  const state = csvEditorStates.get(queryId);
  if (!state?.dirty) return;
  try {
    const updated = await post('/api/submission/query/csv', {session_id: session.session_id, query_id: queryId, csv_text: state.draft});
    csvEditorStates.delete(queryId);
    focusedCsvQueryId = queryId;
    renderSession(updated);
    message(`${state.filename} saved to the queue.`);
  } catch (error) {
    message(error.message, true);
  }
}

async function saveCsv(queryId) {
  const state = csvEditorStates.get(queryId);
  if (!state || state.dirty) return;
  try {
    if (window.showSaveFilePicker) {
      const handle = await window.showSaveFilePicker({
        suggestedName: state.filename,
        types: [{description: 'CSV file', accept: {'text/csv': ['.csv']}}],
      });
      const writable = await handle.createWritable();
      await writable.write(state.content);
      await writable.close();
    } else {
      const url = URL.createObjectURL(new Blob([state.content], {type: 'text/csv;charset=utf-8'}));
      const link = document.createElement('a');
      link.href = url;
      link.download = state.filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }
    message(`${state.filename} saved.`);
  } catch (error) {
    if (error.name !== 'AbortError') message(error.message, true);
  }
}

function timeFrameTool(videoOptions, target = '') {
  const options = videoOptions.filter((item) => item?.video_id);
  if (!options.length) return '';
  const initialSeconds = Math.max(0, Number(options[0].pts_time || 0));
  const minutes = Math.floor(initialSeconds / 60);
  const seconds = initialSeconds - minutes * 60;
  const choices = options.map((item) => `<option value="${escapeHtml(item.video_id)}">${escapeHtml(item.video_id)}</option>`).join('');
  return `<section class="time-frame-tool" data-time-frame data-target="${escapeHtml(target)}"><div class="time-frame-head"><strong>Time to frame</strong><span>FPS from mapping</span></div><div class="time-frame-inputs"><label><span>Video</span><select data-time-video>${choices}</select></label><label><span>Minute</span><input data-time-minute type="number" min="0" step="1" value="${minutes}"></label><label><span>Second</span><input data-time-second type="number" min="0" max="59.999" step="0.001" value="${seconds.toFixed(3)}"></label><button type="button" data-time-convert>Calculate</button></div><div class="time-frame-result" data-time-result>Enter video time, then calculate.</div><div class="time-frame-actions"><button class="secondary" type="button" data-time-inspect hidden>Inspect nearest</button><button type="button" data-time-use disabled>Use frame</button></div></section>`;
}

function bindTimeFrameTools(root, targetResolver) {
  root.querySelectorAll('[data-time-frame]').forEach((tool) => {
    const calculate = tool.querySelector('[data-time-convert]');
    const use = tool.querySelector('[data-time-use]');
    const inspect = tool.querySelector('[data-time-inspect]');
    const result = tool.querySelector('[data-time-result]');
    calculate.onclick = async () => {
      calculate.disabled = true;
      result.textContent = 'Calculating...';
      try {
        const params = new URLSearchParams({
          video_id: tool.querySelector('[data-time-video]').value,
          minutes: tool.querySelector('[data-time-minute]').value,
          seconds: tool.querySelector('[data-time-second]').value,
        });
        const data = await get(`/api/time-to-frame?${params}`);
        tool.dataset.frame = String(data.estimated_frame_idx);
        const nearest = data.nearest_keyframe || {};
        result.textContent = `Frame ${data.estimated_frame_idx} at ${Number(data.pts_time).toFixed(3)}s (${Number(data.fps).toFixed(3)} FPS). Nearest keyframe K${nearest.keyframe_id ?? '-'} / F${nearest.frame_idx ?? '-'}.`;
        use.disabled = false;
        inspect.hidden = !nearest.image_url;
        if (nearest.image_url) {
          const previewId = registerPreview({image_url: nearest.image_url, label: `${data.video_id} time ${Number(data.pts_time).toFixed(3)}s`, meta: `${data.video_id} · estimated frame ${data.estimated_frame_idx} · nearest keyframe ${nearest.keyframe_id} · mapped frame ${nearest.frame_idx}`});
          inspect.onclick = () => openPreviewById(previewId);
        }
      } catch (error) {
        tool.dataset.frame = '';
        use.disabled = true;
        inspect.hidden = true;
        result.textContent = error.message;
      } finally {
        calculate.disabled = false;
      }
    };
    use.onclick = () => {
      const targetInput = targetResolver(tool);
      if (!targetInput || !tool.dataset.frame) return;
      targetInput.value = tool.dataset.frame;
      targetInput.dispatchEvent(new Event('input', {bubbles: true}));
      message(`Frame ${tool.dataset.frame} inserted. Verify it against the video before confirmation.`);
    };
  });
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
  renderCsvPreviews(value.queries);
  updateTaskControls();
}

function renderQueue(queries) {
  $('queue').replaceChildren();
  for (const item of queries) {
    const row = document.createElement('div');
    row.className = 'queue-item';
    row.innerHTML = `<strong>${escapeHtml(item.query_id)}</strong><span>${escapeHtml(item.task)} · ${item.predictions.length} prediction · ${escapeHtml(item.output_file)}</span><div class="queue-actions"><button type="button" data-preview-csv>Preview</button><button type="button" class="secondary" data-remove>Remove</button></div>`;
    row.querySelector('[data-preview-csv]').onclick = () => focusCsvPreview(item.query_id);
    row.querySelector('[data-remove]').onclick = () => removeQuery(item.query_id);
    $('queue').appendChild(row);
  }
  if (!queries.length) $('queue').innerHTML = '<p class="empty">No confirmed query.</p>';
}

function resetReviewWorkspace() {
  agentResult = null;
  qaPayload = null;
  qaDraft = null;
  activeQaVideoId = null;
  qaOfficialFrameId = '';
  trakePayload = null;
  selectedTrakeChain = null;
  trakeVerification = null;
  trakePlanDraft = null;
  trakePlanDraftRevision = null;
  trakeOfficialFrames.clear();
  trakeRefinements.clear();
  previewRegistry.clear();
  $('agent-trace').hidden = true;
  $('agent-inspector').hidden = true;
  $('clip-lanes').hidden = true;
  $('clip-lanes').replaceChildren();
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
  $('dense-frame-wrap').hidden = task === 'TRAKE';
  $('deep-frame-wrap').hidden = !isQa;
  $('gemini-frame-wrap').hidden = !isQa;
  $('qa-frame-settings').hidden = !isQa;
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
  const visualQueries = plan.visual_clip_queries_en?.length
    ? plan.visual_clip_queries_en
    : (plan.visual_clip_query_en ? [plan.visual_clip_query_en] : []);
  $('clip-route').textContent = visualQueries.length
    ? visualQueries.map((query, index) => `Q${index + 1}. ${query}`).join('\n')
    : '-';
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
  renderClipLanes(result);
  updateSelection();
}

function renderClipLanes(result) {
  const panel = $('clip-lanes');
  const rescue = result.channel_postprocessing?.clip?.variant_rescue;
  const lanes = rescue?.lanes || [];
  const sequenceCandidates = rescue?.temporal_sequence?.candidates || [];
  const queries = result.agent_plan?.visual_clip_queries_en?.length
    ? result.agent_plan.visual_clip_queries_en
    : (result.agent_plan?.visual_clip_query_en ? [result.agent_plan.visual_clip_query_en] : []);
  if (!rescue?.enabled || (lanes.length < 2 && !sequenceCandidates.length)) {
    panel.hidden = true;
    panel.replaceChildren();
    return;
  }
  const availableVideos = new Set((result.candidates || []).map((candidate) => candidate.video_id));
  const controls = [{id: 'all', label: `All ${availableVideos.size}`, videos: [...availableVideos]}];
  const sequenceVideos = sequenceCandidates.map((candidate) => candidate.video_id).filter((videoId) => availableVideos.has(videoId));
  if (sequenceVideos.length) {
    const best = sequenceCandidates.find((candidate) => sequenceVideos.includes(candidate.video_id));
    controls.push({
      id: 'clip:sequence',
      label: `Sequence ${sequenceVideos.length}`,
      videos: sequenceVideos,
      query: best ? `${best.matched_events}/${best.event_count} ordered events in ${Number(best.span_seconds).toFixed(1)}s` : 'Ordered event candidates',
    });
  }
  lanes.forEach((lane, index) => {
    const videos = (lane.video_ids || []).filter((videoId) => availableVideos.has(videoId));
    controls.push({id: lane.channel || `clip:q${index + 1}`, label: `Q${index + 1} ${videos.length}`, videos, query: queries[index] || ''});
  });
  panel.hidden = false;
  panel.innerHTML = `<span>Candidate lanes</span>${controls.map((control, index) => `<button class="lane-button${index === 0 ? ' is-active' : ''}" type="button" data-lane="${escapeHtml(control.id)}" title="${escapeHtml(control.query || 'Show all fused and rescued candidates')}">${escapeHtml(control.label)}</button>`).join('')}<small>Sequence keeps events in time order; Q lanes show strong videos for each visual moment.</small>`;
  panel.querySelectorAll('[data-lane]').forEach((button) => {
    button.onclick = () => {
      const control = controls.find((item) => item.id === button.dataset.lane) || controls[0];
      const allowed = new Set(control.videos);
      panel.querySelectorAll('[data-lane]').forEach((item) => item.classList.toggle('is-active', item === button));
      $('candidates').querySelectorAll('.candidate').forEach((row) => {
        row.hidden = control.id !== 'all' && !allowed.has(row.dataset.videoId);
      });
      message(control.id === 'all' ? 'Showing the fused candidate list.' : `${control.label}: ${control.query}`);
    };
  });
}

function candidateEvidence(candidate) {
  const provenance = candidate.retrieval?.provenance || {};
  const ranks = provenance.retriever_ranks || {};
  const contributions = provenance.contributions || {};
  const chips = Object.entries(ranks).map(([name, rank]) => {
    const contribution = contributions[name];
    return `<span class="evidence-chip is-${escapeHtml(name)}">${escapeHtml(name.toUpperCase())} #${Number(rank)}${contribution !== undefined ? ` · +${Number(contribution).toFixed(4)}` : ''}</span>`;
  });
  const clipEvidence = provenance.evidence?.clip;
  const clipQueries = clipEvidence?.provenance?.queries || [];
  const variantRanks = clipEvidence?.provenance?.variant_fusion?.retriever_ranks || {};
  for (const [name, rank] of Object.entries(variantRanks)) {
    const queryIndex = Math.max(0, Number(String(name).split('q').pop() || 1) - 1);
    const query = clipQueries[queryIndex] || name;
    chips.push(`<span class="evidence-chip is-clip" title="${escapeHtml(query)}">CLIP Q${queryIndex + 1} #${Number(rank)}</span>`);
  }
  const sequence = clipEvidence?.provenance?.temporal_sequence;
  if (sequence?.matched_events >= 2) {
    const sequenceFrames = (sequence.frames || []).map((frame) => `${frame.channel}: ${Number(frame.pts_time).toFixed(1)}s`).join(' -> ');
    chips.push(`<span class="evidence-chip is-sequence" title="${escapeHtml(sequenceFrames)}">SEQ ${Number(sequence.matched_events)}/${Number(sequence.event_count)} Â· ${Number(sequence.span_seconds).toFixed(1)}s</span>`);
  }
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
  row.dataset.videoId = candidate.video_id;
  row.dataset.candidate = JSON.stringify(candidate);
  const metadata = candidate.metadata || {};
  const hasKeyframe = candidate.keyframe_id !== null && candidate.keyframe_id !== undefined && Number.isInteger(Number(candidate.keyframe_id));
  const image = previewButton({image_url: candidate.image_url, label: `${candidate.video_id} keyframe ${candidate.keyframe_id}`, meta: `${candidate.video_id} · keyframe ${candidate.keyframe_id} · frame ${candidate.frame_idx} · ${Number(candidate.pts_time || 0).toFixed(2)}s`}, 'candidate-image-button');
  const openVideo = candidate.video_url ? `<a class="candidate-action" href="${escapeHtml(candidate.video_url)}" target="_blank" rel="noreferrer">Open video</a>` : '';
  const neighborAction = hasKeyframe
    ? `<button class="secondary" type="button" data-neighbor data-video="${escapeHtml(candidate.video_id)}" data-keyframe="${Number(candidate.keyframe_id)}">Nearby frames</button>`
    : '';
  const refineAction = `<button class="secondary" type="button" data-refine-candidate>Refine in video</button>`;
  row.innerHTML = `
    <div class="candidate-media">${image}<span class="rank-badge">#${Number(candidate.rank)}</span><span class="zoom-hint">Click to inspect</span></div>
    <div class="candidate-main">
      <div class="candidate-title"><strong>${escapeHtml(candidate.video_id)}</strong><span>score ${Number(candidate.retrieval?.score || 0).toFixed(4)}</span></div>
      <p>${escapeHtml(metadata.title || 'Untitled video')}</p>
      <dl class="candidate-facts"><div><dt>Keyframe</dt><dd>${escapeHtml(candidate.keyframe_id)}</dd></div><div><dt>Frame index</dt><dd>${escapeHtml(candidate.frame_idx)}</dd></div><div><dt>Timestamp</dt><dd>${Number(candidate.pts_time || 0).toFixed(2)}s</dd></div><div><dt>Author</dt><dd>${escapeHtml(metadata.author || '-')}</dd></div></dl>
      <div class="candidate-evidence">${candidateEvidence(candidate)}</div>
      <div class="candidate-actions">${openVideo}${refineAction}${neighborAction}<button class="secondary" type="button" data-copy-candidate>Copy details</button></div>
    </div>
    <aside class="candidate-review"><label class="select-control"><input class="candidate-select" type="checkbox"><span>Select candidate</span></label><label class="official-input"><span>Verified official frame ID</span><input type="number" min="0" step="1" placeholder="Required before confirm"></label>${timeFrameTool([{video_id: candidate.video_id, pts_time: candidate.pts_time}])}<small>Timestamp conversion is an estimate. Verify the frame against the video before confirmation.</small></aside>
    <div class="candidate-refinement" hidden></div>
    <div class="neighbors" hidden></div>`;
  row.querySelector('.candidate-select').setAttribute('aria-label', `Select ${candidate.video_id}`);
  row.querySelector('.candidate-select').onchange = updateSelection;
  row.querySelector('.official-input input').oninput = updateSelection;
  bindTimeFrameTools(row, () => row.querySelector('.official-input input'));
  wireReviewActions(row);
  return row;
}

async function refineCandidate(button, requestedScope = null) {
  const row = button.closest('.candidate');
  const panel = row.querySelector('.candidate-refinement');
  const candidate = JSON.parse(row.dataset.candidate);
  const hasKeyframe = candidate.keyframe_id !== null && candidate.keyframe_id !== undefined && Number.isInteger(Number(candidate.keyframe_id));
  const scope = requestedScope || (hasKeyframe ? '40' : 'full');
  const fullVideo = scope === 'full';
  button.disabled = true;
  button.textContent = 'Refining...';
  panel.hidden = false;
  panel.innerHTML = '<span class="empty">Ranking frames inside the selected video...</span>';
  try {
    const payload = await post('/api/submission/candidate/refine', {
      session_id: session.session_id,
      query_id: agentResult.query_id,
      query: agentResult.query,
      video_id: candidate.video_id,
      seed_keyframe_id: fullVideo ? null : candidate.keyframe_id,
      radius_seconds: fullVideo ? 180 : Number(scope),
      frame_limit: 12,
      use_gemini: $('use-gemini').checked,
      use_dense: $('use-dense-frames').checked,
    });
    renderCandidateRefinement(panel, payload, row, button, scope);
    message(`${candidate.video_id}: ${payload.frames.length} candidate-local frames ready for review.`, !payload.frames.length);
  } catch (error) {
    panel.innerHTML = `<span class="empty">${escapeHtml(error.message)}</span>`;
    message(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = 'Refine in video';
  }
}

function refinementEvidence(frame) {
  const ranks = frame.refinement_ranks || {};
  const chips = Object.entries(ranks).map(([name, rank]) => `<span class="evidence-chip is-${escapeHtml(name)}">${escapeHtml(name.toUpperCase())} #${Number(rank)}</span>`);
  const text = (frame.refinement_evidence || []).find((item) => item.matched_text)?.matched_text;
  if (text) chips.push(`<span class="evidence-chip is-source">${escapeHtml(text.slice(0, 140))}</span>`);
  return chips.join('');
}

function renderCandidateRefinement(panel, payload, row, trigger, scope) {
  const frames = payload.frames || [];
  const sequence = `refine:${payload.query_id}:${payload.video_id}:${Date.now()}`;
  const options = [['20', '±20s'], ['40', '±40s'], ['90', '±90s'], ['180', '±180s'], ['full', 'Full video']]
    .map(([value, label]) => `<option value="${value}" ${value === scope ? 'selected' : ''}>${label}</option>`).join('');
  const cards = frames.map((frame, index) => {
    const dense = Boolean(frame.dense_frame);
    const frameLabel = dense ? `decoded frame ${frame.frame_idx}` : `keyframe ${frame.keyframe_id}`;
    const previewId = registerPreview({
      image_url: frame.image_url,
      label: `${payload.video_id} ${frameLabel}`,
      meta: `${payload.video_id} · ${dense ? 'dense video frame' : 'keyframe'} · anchor K${frame.keyframe_id} · frame ${frame.frame_idx} · ${Number(frame.pts_time || 0).toFixed(3)}s`,
      sequence,
    });
    const image = frame.image_url
      ? `<button class="refinement-image" type="button" data-open-preview="${previewId}"><img src="${escapeHtml(frame.image_url)}" alt="${escapeHtml(payload.video_id)} frame ${Number(frame.frame_idx)}" loading="lazy"></button>`
      : '<div class="missing-image">Preview unavailable</div>';
    const scoreLabel = dense ? `CLIP ${Number(frame.dense_score || 0).toFixed(4)}` : `RRF ${Number(frame.refinement_score || 0).toFixed(4)}`;
    const denseBadge = dense ? '<span class="dense-frame-badge">Dense frame</span>' : '';
    return `<article class="refinement-frame${index === 0 ? ' is-leading' : ''}">${image}<div class="refinement-frame-body"><div><strong>#${Number(frame.refinement_rank)} · F${Number(frame.frame_idx)}${dense ? ` / anchor K${Number(frame.keyframe_id)}` : ` / K${Number(frame.keyframe_id)}`}</strong><span>${Number(frame.pts_time || 0).toFixed(3)}s · ${scoreLabel}</span>${denseBadge}</div><div class="candidate-evidence">${refinementEvidence(frame)}</div><div class="candidate-actions"><button type="button" data-use-refined-frame="${index}">Use frame</button><button class="secondary" type="button" data-neighbor data-video="${escapeHtml(payload.video_id)}" data-keyframe="${Number(frame.keyframe_id)}">Nearby keyframes</button></div></div></article>`;
  }).join('');
  const counts = payload.refinement?.channel_hit_counts || {};
  panel.innerHTML = `<div class="refinement-toolbar"><div><strong>${escapeHtml(payload.video_id)} frame refinement</strong><span>${Object.entries(counts).map(([name, count]) => `${name.toUpperCase()} ${count}`).join(' · ') || 'No channel hits'}</span></div><label><span>Search window</span><select data-refine-scope>${options}</select></label><button class="secondary" type="button" data-rerun-refinement>Rerun</button></div><div class="refinement-track">${cards || '<span class="empty">No frame survived candidate-local refinement.</span>'}</div><details class="refinement-trace"><summary>Refinement trace</summary><pre>${escapeHtml(JSON.stringify(payload.refinement || {}, null, 2))}</pre></details>`;
  panel.querySelector('[data-rerun-refinement]').onclick = () => refineCandidate(trigger, panel.querySelector('[data-refine-scope]').value);
  panel.querySelectorAll('[data-use-refined-frame]').forEach((useButton) => {
    useButton.onclick = () => {
      const frame = frames[Number(useButton.dataset.useRefinedFrame)];
      if (frame) applyRefinedFrame(row, frame, payload);
    };
  });
  wireReviewActions(panel);
}

function applyRefinedFrame(row, frame, payload) {
  const candidate = JSON.parse(row.dataset.candidate);
  const selected = row.querySelector('.candidate-select').checked;
  const officialFrame = row.querySelector('.official-input input').value;
  const evidence = (frame.refinement_evidence || []).find((item) => item.matched_text);
  const updated = {
    ...candidate,
    keyframe_id: frame.keyframe_id,
    frame_idx: frame.frame_idx,
    pts_time: frame.pts_time,
    image_url: frame.image_url,
    keyframe_path: frame.keyframe_path,
    source_type: (frame.refinement_channels || []).join('+') || candidate.source_type,
    matched_text: evidence?.matched_text || candidate.matched_text,
    refinement: {schema_version: payload.schema_version, ...payload.refinement, selected_frame: frame},
  };
  const index = agentResult.candidates.findIndex((item) => item.candidate_id === candidate.candidate_id);
  if (index >= 0) agentResult.candidates[index] = updated;
  const replacement = candidateRow(updated);
  replacement.querySelector('.candidate-select').checked = selected;
  replacement.querySelector('.official-input input').value = officialFrame;
  row.replaceWith(replacement);
  updateSelection();
  message(`${updated.video_id}: ${updated.dense_frame ? 'decoded' : 'keyframe'} frame ${updated.frame_idx} selected for review. Official frame ID was not changed.`);
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
  root.querySelectorAll('[data-refine-candidate]').forEach((button) => { button.onclick = () => refineCandidate(button); });
  root.querySelectorAll('[data-copy-candidate]').forEach((button) => {
    button.onclick = async () => {
      const candidate = JSON.parse(button.closest('.candidate, .qa-candidate').dataset.candidate);
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

function qaCandidateFrame(candidate) {
  const frame = Array.isArray(candidate.frames) && candidate.frames.length ? candidate.frames[0] : candidate;
  return {
    video_id: candidate.video_id,
    keyframe_id: frame.keyframe_id ?? candidate.best_keyframe_id,
    frame_idx: frame.frame_idx ?? candidate.best_frame_idx,
    pts_time: frame.pts_time ?? candidate.best_pts_time,
    keyframe_path: frame.keyframe_path ?? candidate.best_keyframe_path,
    image_url: frame.image_url || candidate.image_url,
  };
}

function qaFrameShortlist(candidate) {
  const frames = Array.isArray(candidate.frames) ? candidate.frames.slice(0, 12) : [];
  if (frames.length < 2) return '';
  return `<div class="qa-frame-shortlist"><div class="shortlist-head"><strong>Frame shortlist</strong><span>${frames.length} frames inside this video</span></div><div class="shortlist-track">${frames.map((frame) => {
    const imageUrl = frame.image_url || frameImageUrl(frame);
    const dense = Boolean(frame.dense_frame);
    const preview = previewButton({image_url: imageUrl, label: `${candidate.video_id} ${dense ? 'decoded frame' : 'keyframe'} ${frame.frame_idx}`, meta: `${candidate.video_id} / ${dense ? 'dense video frame' : `keyframe ${frame.keyframe_id}`} / frame ${frame.frame_idx} / ${Number(frame.pts_time || 0).toFixed(3)}s`}, 'shortlist-image-button');
    const vlm = frame.vlm_relevance === undefined ? '' : `<span class="shortlist-vlm">Gemini ${Number(frame.vlm_relevance).toFixed(2)}</span>`;
    const badge = dense ? '<span class="dense-frame-badge">Dense frame</span>' : '';
    return `<article class="shortlist-frame">${preview}<div><strong>F${escapeHtml(frame.frame_idx)}${dense ? ` / anchor K${escapeHtml(frame.keyframe_id)}` : ` / K${escapeHtml(frame.keyframe_id)}`}</strong><span>${Number(frame.pts_time || 0).toFixed(3)}s</span>${badge}${vlm}</div><button class="secondary" type="button" data-use-qa-frame="${escapeHtml(frame.frame_idx)}">Use frame ID</button></article>`;
  }).join('')}</div></div>`;
}

function useQaFrame(frameIdx) {
  qaOfficialFrameId = String(frameIdx);
  const input = $('qa-official-frame');
  if (input) input.value = qaOfficialFrameId;
  updateQaButtons();
  message(`Frame ${frameIdx} copied to the Q&A official frame field. Verify it against the source video before submission.`);
}

function renderQaCandidates() {
  const candidates = qaPayload?.evidence_pack?.candidates || [];
  if (!activeQaVideoId || !candidates.some((item) => item.video_id === activeQaVideoId)) activeQaVideoId = candidates[0]?.video_id || null;
  $('candidates').replaceChildren();
  for (const candidate of candidates) {
    const frame = qaCandidateFrame(candidate);
    const metadata = candidate.metadata || {};
    const row = document.createElement('article');
    row.className = `qa-candidate inspectable${candidate.video_id === activeQaVideoId ? ' is-active' : ''}`;
    row.dataset.candidate = JSON.stringify({...candidate, ...frame});
    const image = previewButton({image_url: frame.image_url || frameImageUrl(frame), label: `${candidate.video_id} keyframe ${frame.keyframe_id ?? '-'}`, meta: `${candidate.video_id} / keyframe ${frame.keyframe_id ?? '-'} / frame ${frame.frame_idx ?? '-'} / ${Number(frame.pts_time || 0).toFixed(2)}s`}, 'candidate-image-button');
    const neighbor = frame.keyframe_id === null || frame.keyframe_id === undefined ? '' : `<button class="secondary" type="button" data-neighbor data-video="${escapeHtml(candidate.video_id)}" data-keyframe="${Number(frame.keyframe_id)}">Nearby frames</button>`;
    const openVideo = candidate.video_url ? `<a class="candidate-action" href="${escapeHtml(candidate.video_url)}" target="_blank" rel="noreferrer">Open video</a>` : '';
    const score = Number(candidate.retrieval?.score ?? candidate.score ?? candidate.raw_score ?? 0);
    row.innerHTML = `<div class="candidate-media">${image}<span class="rank-badge">#${Number(candidate.rank || 0)}</span><span class="zoom-hint">Click to inspect</span></div><div class="candidate-main"><div class="candidate-title"><strong>${escapeHtml(candidate.video_id)}</strong><span>score ${score.toFixed(4)}</span></div><p>${escapeHtml(metadata.title || 'Untitled video')}</p><dl class="candidate-facts"><div><dt>Keyframe</dt><dd>${escapeHtml(frame.keyframe_id ?? '-')}</dd></div><div><dt>Frame index</dt><dd>${escapeHtml(frame.frame_idx ?? '-')}</dd></div><div><dt>Timestamp</dt><dd>${Number(frame.pts_time || 0).toFixed(2)}s</dd></div><div><dt>Author</dt><dd>${escapeHtml(metadata.author || '-')}</dd></div><div><dt>Publish date</dt><dd>${escapeHtml(metadata.publish_date || '-')}</dd></div></dl><div class="candidate-evidence">${candidateEvidence(candidate)}</div><div class="candidate-actions">${openVideo}${neighbor}<button class="secondary" type="button" data-copy-candidate>Copy details</button></div></div><aside class="qa-candidate-review"><label class="select-control"><input type="radio" name="qa-active-video" value="${escapeHtml(candidate.video_id)}" ${candidate.video_id === activeQaVideoId ? 'checked' : ''}><span>Use this video</span></label><small>Only evidence from this video can be used in one Q&A answer.</small></aside>${qaFrameShortlist(candidate)}<div class="neighbors" hidden></div>`;
    row.querySelector('input[name="qa-active-video"]').onchange = () => {
      activeQaVideoId = candidate.video_id;
      qaDraft = null;
      qaOfficialFrameId = '';
      renderQaCandidates();
      renderQaWorkflow();
      message(`${activeQaVideoId} selected. Review its frame, metadata, OCR, ASR and object evidence.`);
    };
    row.querySelectorAll('[data-use-qa-frame]').forEach((button) => { button.onclick = () => useQaFrame(Number(button.dataset.useQaFrame)); });
    wireReviewActions(row);
    $('candidates').appendChild(row);
  }
  if (!candidates.length) $('candidates').innerHTML = '<p class="empty">No Q&A candidate video.</p>';
}

function renderQaWorkflow(selectedIds = []) {
  const pack = qaPayload.evidence_pack || {};
  const selected = new Set(selectedIds);
  $('task-workflow').hidden = false;
  $('workflow-title').textContent = 'Q&A evidence review';
  $('workflow-state').textContent = qaDraft ? (qaDraft.confidence_state || 'DRAFT') : 'EVIDENCE';
  const allEvidence = pack.evidence_refs || [];
  const evidence = allEvidence.filter((item) => item.video_id === activeQaVideoId);
  const evidenceRows = evidence.map((item) => {
    const preview = previewButton({image_url: frameImageUrl(item), label: `${item.video_id} ${item.modality}`, meta: `${item.video_id} · ${item.modality} · keyframe ${item.keyframe_id ?? '-'} · frame ${item.frame_idx ?? '-'}`}, 'evidence-image-button');
    const neighbor = item.keyframe_id === null || item.keyframe_id === undefined ? '' : `<button class="secondary" type="button" data-neighbor data-video="${escapeHtml(item.video_id)}" data-keyframe="${Number(item.keyframe_id)}">Nearby frames</button>`;
    const frameLevel = item.keyframe_id !== null && item.keyframe_id !== undefined;
    return `<article class="evidence-row inspectable"><label class="evidence-select"><input type="checkbox" data-evidence-id="${escapeHtml(item.evidence_id)}" data-frame-level="${frameLevel}" ${selected.has(item.evidence_id) ? 'checked' : ''}><span>${escapeHtml(item.modality)} · ${escapeHtml(item.video_id)} · K${item.keyframe_id ?? '-'} · F${item.frame_idx ?? '-'}</span></label>${preview}<p>${escapeHtml(evidenceText(item))}</p><div class="candidate-actions">${neighbor}</div><div class="neighbors" hidden></div></article>`;
  }).join('');
  const answer = qaDraft?.raw_answer || '';
  const activeCandidate = (pack.candidates || []).find((item) => item.video_id === activeQaVideoId);
  const activeFrame = activeCandidate ? qaCandidateFrame(activeCandidate) : null;
  const qaVideoOptions = activeFrame ? [{video_id: activeQaVideoId, pts_time: activeFrame.pts_time}] : [];
  $('workflow-content').innerHTML = `<div class="workflow-summary"><span>${(pack.candidates || []).length} candidate videos</span><span>${allEvidence.length} total evidence</span><span>${evidence.length} for ${escapeHtml(activeQaVideoId || '-')}</span><span>${escapeHtml(qaPayload.question_route?.question_type || 'UNKNOWN')}</span></div><div class="workflow-notice">Active video: <strong>${escapeHtml(activeQaVideoId || 'none')}</strong>. Switching videos clears the current answer draft.</div><div class="evidence-grid">${evidenceRows || '<p class="empty">No source-backed evidence is available for this video.</p>'}</div><div class="workflow-actions"><button id="qa-draft-button" type="button">Draft answer from selected evidence</button><label><span>Final answer</span><input id="qa-final-answer" value="${escapeHtml(answer)}" maxlength="100" placeholder="Review or edit the proposed answer"></label><label><span>Verified official frame ID</span><input id="qa-official-frame" type="number" min="0" step="1" value="${escapeHtml(qaOfficialFrameId)}" placeholder="Required"></label><button id="qa-confirm-button" type="button" disabled>Review and add to queue</button>${timeFrameTool(qaVideoOptions)}</div><details class="workflow-details"><summary>Draft diagnostics</summary><pre>${escapeHtml(JSON.stringify(qaDraft || {status: 'No draft yet'}, null, 2))}</pre></details>`;
  $('workflow-content').querySelectorAll('[data-evidence-id]').forEach((input) => { input.onchange = updateQaButtons; });
  $('qa-draft-button').onclick = draftQaAnswer;
  $('qa-final-answer').oninput = updateQaButtons;
  $('qa-official-frame').oninput = () => { qaOfficialFrameId = $('qa-official-frame').value; updateQaButtons(); };
  $('qa-confirm-button').onclick = confirmQaToQueue;
  bindTimeFrameTools($('workflow-content'), () => $('qa-official-frame'));
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
  qaPayload = await post('/api/qa/prepare', {query_id: $('query-id').value.trim(), event_query: $('query').value.trim(), question: $('qa-question').value.trim(), use_hybrid_retrieval: $('use-hybrid').checked, use_gemini_planner: $('use-gemini').checked, deep_frame_search: $('deep-frame-search').checked, frame_search_radius_seconds: Number($('frame-search-radius').value), frame_search_limit: Number($('frame-search-limit').value), use_gemini_frame_reranker: $('use-gemini-frame').checked, use_dense_frames: $('use-dense-frames').checked});
  qaDraft = null;
  activeQaVideoId = qaPayload.evidence_pack?.candidates?.[0]?.video_id || null;
  qaOfficialFrameId = '';
  const context = qaPayload.evidence_pack?.retrieval_context || {};
  renderPlanInspector(context.query_plan || {}, context.agent_trace || {}, context.channel_hit_counts || {}, context.structured_constraints);
  const localization = context.frame_localization || {};
  const visualRerank = context.gemini_frame_rerank || {};
  if ($('deep-frame-search').checked || $('use-gemini-frame').checked) {
    $('agent-trace').insertAdjacentHTML('beforeend', `<span class="trace-pill">Frame search: ${escapeHtml(localization.status || 'off')}</span><span class="trace-pill">Gemini frames: ${escapeHtml(visualRerank.status || 'off')}</span>`);
  }
  renderQaCandidates();
  renderQaWorkflow();
  message('Q&A candidates are ready. Choose one video, inspect its evidence, then draft the answer.');
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
  focusCsvPreview($('query-id').value.trim());
  message(`${$('query-id').value.trim()} Q&A result added to the official queue.`);
}

function allTrakeChains(state) {
  const chains = (state.alignments || []).flatMap((alignment) => alignment.chains || []);
  if (state.manual_chain && !chains.some((chain) => chain.chain_id === state.manual_chain.chain_id)) chains.unshift(state.manual_chain);
  return chains;
}

function trakeCandidatePreview(candidate, eventId) {
  const preview = previewButton({image_url: frameImageUrl(candidate), label: `${candidate.video_id} ${eventId}`, meta: `${candidate.video_id} · ${eventId} · keyframe ${candidate.keyframe_id} · frame ${candidate.frame_idx} · ${Number(candidate.pts_time || 0).toFixed(2)}s`}, 'event-image-button');
  return `<div class="trake-event inspectable">${preview}<strong>${escapeHtml(eventId)}</strong><span>K${Number(candidate.keyframe_id)} · F${Number(candidate.frame_idx)} · ${Number(candidate.pts_time || 0).toFixed(2)}s</span><button class="secondary" type="button" data-neighbor data-video="${escapeHtml(candidate.video_id)}" data-keyframe="${Number(candidate.keyframe_id)}">Nearby frames</button><div class="neighbors" hidden></div></div>`;
}

function trakeAlternativePanel(state, chain, chainEvent) {
  const candidates = (state.pools || []).find((pool) => pool.event?.event_id === chainEvent.event_id)?.candidates || [];
  const sameVideo = candidates.filter((candidate) => candidate.video_id === chain.video_id).slice(0, 10);
  const locked = (state.locked_event_ids || []).includes(chainEvent.event_id);
  const alternatives = sameVideo.map((candidate) => {
    const current = candidate.candidate_id === chainEvent.candidate?.candidate_id;
    const preview = previewButton({image_url: frameImageUrl(candidate), label: `${chain.video_id} ${chainEvent.event_id} alternative`, meta: `${chain.video_id} / ${chainEvent.event_id} / K${candidate.keyframe_id} / F${candidate.frame_idx} / ${Number(candidate.pts_time || 0).toFixed(2)}s`}, 'alternative-image-button');
    return `<article class="trake-alternative${current ? ' is-current' : ''}">${preview}<div><strong>K${Number(candidate.keyframe_id)} / F${Number(candidate.frame_idx)}</strong><span>${Number(candidate.pts_time || 0).toFixed(2)}s / score ${Number(candidate.local_score || 0).toFixed(3)}</span></div><button type="button" data-trake-replace="${escapeHtml(chainEvent.event_id)}" data-candidate-id="${escapeHtml(candidate.candidate_id)}" ${current ? 'disabled' : ''}>${current ? 'Current' : 'Replace + lock'}</button></article>`;
  }).join('');
  const refinement = trakeRefinements.get(chainEvent.event_id);
  const refinedPreview = refinement?.image_url ? previewButton({image_url: refinement.image_url, label: `${chain.video_id} ${chainEvent.event_id} refined frame`, meta: `${chain.video_id} / ${chainEvent.event_id} / refined ${Number(refinement.refined_pts_time || 0).toFixed(3)}s`}, 'alternative-image-button') : '';
  const refinementStatus = refinement ? `<div class="refinement-result">${refinedPreview}<span><strong>${escapeHtml(refinement.status)}</strong><br>${escapeHtml(refinement.reason || '')}${refinement.refined_pts_time !== undefined ? `<br>${Number(refinement.refined_pts_time).toFixed(3)}s` : ''}</span></div>` : '';
  return `<details class="event-alternatives"><summary>${sameVideo.length} same-video candidates ${locked ? '/ locked' : ''}</summary><div class="alternative-actions"><button class="secondary" type="button" data-trake-refine="${escapeHtml(chainEvent.event_id)}">Refine around current frame</button></div>${refinementStatus}<div class="trake-alternative-grid">${alternatives || '<p class="empty">No alternative candidate for this video.</p>'}</div></details>`;
}

function renderTrakeWorkflow() {
  const state = trakePayload.state;
  if (trakePlanDraftRevision !== state.plan_revision || !trakePlanDraft) {
    trakePlanDraft = (state.request?.events || []).map((event) => ({...event}));
    trakePlanDraftRevision = state.plan_revision;
  }
  const chains = allTrakeChains(state);
  if (!selectedTrakeChain || !chains.some((item) => item.chain_id === selectedTrakeChain.chain_id)) selectedTrakeChain = chains[0] || null;
  $('task-workflow').hidden = false;
  $('workflow-title').textContent = 'TRAKE sequence review';
  $('workflow-state').textContent = selectedTrakeChain ? 'CHAIN READY' : 'NO CHAIN';
  const planRows = trakePlanDraft.map((event, index) => `<div class="trake-plan-row"><span><strong>${escapeHtml(event.event_id)}</strong> ${escapeHtml((event.modalities || []).join(' + '))}</span><input data-trake-plan-text="${escapeHtml(event.event_id)}" value="${escapeHtml(event.text)}"><select data-trake-plan-required="${escapeHtml(event.event_id)}"><option value="true" ${event.required ? 'selected' : ''}>required</option><option value="false" ${!event.required ? 'selected' : ''}>optional</option></select><div class="plan-row-actions"><button class="secondary" type="button" data-trake-plan-up="${escapeHtml(event.event_id)}" ${index === 0 ? 'disabled' : ''}>Up</button><button class="secondary" type="button" data-trake-plan-down="${escapeHtml(event.event_id)}" ${index === trakePlanDraft.length - 1 ? 'disabled' : ''}>Down</button><button class="secondary" type="button" data-trake-plan-remove="${escapeHtml(event.event_id)}" ${trakePlanDraft.length <= 1 ? 'disabled' : ''}>Remove</button></div></div>`).join('');
  const chainRows = chains.map((chain) => {
    const selected = chain.chain_id === selectedTrakeChain?.chain_id;
    const events = (chain.events || []).filter((event) => event.candidate).map((event) => `<section class="trake-event-review">${trakeCandidatePreview(event.candidate, event.event_id)}${trakeAlternativePanel(state, chain, event)}</section>`).join('');
    const score = chain.score_components?.final_rerank_score ?? chain.score?.final_score ?? 0;
    return `<article class="trake-chain${selected ? ' is-selected' : ''}"><label class="chain-select"><input type="radio" name="trake-chain" value="${escapeHtml(chain.chain_id)}" ${selected ? 'checked' : ''}><span>${escapeHtml(chain.video_id)} · score ${Number(score).toFixed(4)}</span></label><div class="trake-events">${events}</div><details><summary>Score and warnings</summary><pre>${escapeHtml(JSON.stringify({score: chain.score_components || chain.score, warnings: chain.warnings || []}, null, 2))}</pre></details></article>`;
  }).join('');
  const officialInputs = selectedTrakeChain ? selectedTrakeChain.events.filter((event) => event.candidate).map((event) => `<div class="trake-frame-entry"><label><span>${escapeHtml(event.event_id)} official frame ID</span><input data-trake-frame="${escapeHtml(event.event_id)}" type="number" min="0" step="1" value="${escapeHtml(trakeOfficialFrames.get(event.event_id) ?? '')}" placeholder="Required"></label>${timeFrameTool([{video_id: event.candidate.video_id, pts_time: event.candidate.pts_time}], event.event_id)}</div>`).join('') : '';
  const traces = state.diagnostics?.retrieval?.hybrid_query_traces || {};
  const verificationResult = trakeVerification?.result || {};
  const verificationRows = (verificationResult.chains || []).map((chain) => `<article class="vlm-chain-result"><strong>${escapeHtml(chain.chain_id)} / support ${Number(chain.support || 0).toFixed(2)}</strong><span>${chain.temporal_consistency ? 'Order visually coherent' : 'Order needs review'}</span><p>${escapeHtml(chain.reason || '')}</p>${(chain.event_support || []).map((event) => `<small>${escapeHtml(event.event_id)} ${Number(event.support || 0).toFixed(2)} / ${escapeHtml(event.reason || '')}</small>`).join('')}</article>`).join('');
  const verificationPanel = trakeVerification ? `<div class="vlm-verification is-${escapeHtml(String(trakeVerification.status || '').toLowerCase())}"><div><strong>Gemini sequence advisory: ${escapeHtml(trakeVerification.status || '-')}</strong><span>${escapeHtml((trakeVerification.trigger_reasons || []).join(' / ') || trakeVerification.reason || 'No trigger reason')}</span></div>${verificationRows || '<p class="empty">No visual advisory was returned. The server chain remains unchanged.</p>'}<details><summary>Raw verifier response</summary><pre>${escapeHtml(JSON.stringify(trakeVerification, null, 2))}</pre></details></div>` : '';
  $('workflow-content').innerHTML = `<div class="workflow-notice">TRAKE retrieves a single video and validates event order. Review the Agent plan before accepting the sequence.</div><details class="trake-plan-editor" open><summary>Agent event plan / revision ${Number(state.plan_revision || 1)}</summary><div class="trake-plan">${planRows}</div><div class="plan-editor-actions"><button class="secondary" id="trake-add-event" type="button" ${trakePlanDraft.length >= 5 ? 'disabled' : ''}>Add event</button><button id="trake-update-plan" type="button">Save plan and rerun retrieval</button></div></details><div class="trake-review-tools"><button class="secondary" id="trake-verify-button" type="button" ${chains.length ? '' : 'disabled'}>Ask Gemini to review low-confidence chains</button><span>Advisory only. Gemini cannot create or replace a frame.</span></div>${verificationPanel}<div class="chain-list">${chainRows || '<p class="empty">No valid same-video temporal chain was found.</p>'}</div><div class="workflow-actions trake-submit">${officialInputs}<button id="trake-confirm-button" type="button" ${selectedTrakeChain ? '' : 'disabled'}>Review chain and add to queue</button></div><details class="workflow-details"><summary>Per-event Gemini planner traces</summary><pre>${escapeHtml(JSON.stringify(traces, null, 2))}</pre></details>`;
  $('workflow-content').querySelectorAll('input[name="trake-chain"]').forEach((input) => { input.onchange = () => { selectedTrakeChain = chains.find((chain) => chain.chain_id === input.value) || null; trakeOfficialFrames.clear(); renderTrakeWorkflow(); }; });
  $('workflow-content').querySelectorAll('[data-trake-frame]').forEach((input) => { input.oninput = () => { trakeOfficialFrames.set(input.dataset.trakeFrame, input.value); updateTrakeButton(); }; });
  $('workflow-content').querySelectorAll('[data-trake-replace]').forEach((button) => { button.onclick = () => replaceTrakeCandidate(button.dataset.trakeReplace, button.dataset.candidateId); });
  $('workflow-content').querySelectorAll('[data-trake-refine]').forEach((button) => { button.onclick = () => refineTrakeCandidate(button.dataset.trakeRefine); });
  bindTimeFrameTools($('workflow-content'), (tool) => [...$('workflow-content').querySelectorAll('[data-trake-frame]')].find((input) => input.dataset.trakeFrame === tool.dataset.target));
  if ($('trake-update-plan')) $('trake-update-plan').onclick = updateTrakePlan;
  if ($('trake-add-event')) $('trake-add-event').onclick = addTrakeEvent;
  if ($('trake-verify-button')) $('trake-verify-button').onclick = verifyTrakeWithGemini;
  $('workflow-content').querySelectorAll('[data-trake-plan-up]').forEach((button) => { button.onclick = () => moveTrakeEvent(button.dataset.trakePlanUp, -1); });
  $('workflow-content').querySelectorAll('[data-trake-plan-down]').forEach((button) => { button.onclick = () => moveTrakeEvent(button.dataset.trakePlanDown, 1); });
  $('workflow-content').querySelectorAll('[data-trake-plan-remove]').forEach((button) => { button.onclick = () => removeTrakeEvent(button.dataset.trakePlanRemove); });
  if ($('trake-confirm-button')) $('trake-confirm-button').onclick = confirmTrakeToQueue;
  wireReviewActions($('workflow-content'));
  updateTrakeButton();
}

function syncTrakePlanDraft() {
  trakePlanDraft = trakePlanDraft.map((event) => {
    const textInput = $('workflow-content').querySelector(`[data-trake-plan-text="${CSS.escape(event.event_id)}"]`);
    const requiredInput = $('workflow-content').querySelector(`[data-trake-plan-required="${CSS.escape(event.event_id)}"]`);
    return {...event, text: textInput?.value.trim() || event.text, required: requiredInput ? requiredInput.value === 'true' : event.required};
  });
}

function renumberTrakePlanDraft() {
  trakePlanDraft = trakePlanDraft.map((event, index) => ({...event, event_id: `e${index + 1}`, order: index + 1}));
}

function moveTrakeEvent(eventId, offset) {
  syncTrakePlanDraft();
  const index = trakePlanDraft.findIndex((event) => event.event_id === eventId);
  const target = index + offset;
  if (index < 0 || target < 0 || target >= trakePlanDraft.length) return;
  [trakePlanDraft[index], trakePlanDraft[target]] = [trakePlanDraft[target], trakePlanDraft[index]];
  renumberTrakePlanDraft();
  renderTrakeWorkflow();
}

function removeTrakeEvent(eventId) {
  syncTrakePlanDraft();
  if (trakePlanDraft.length <= 1) return;
  trakePlanDraft = trakePlanDraft.filter((event) => event.event_id !== eventId);
  renumberTrakePlanDraft();
  renderTrakeWorkflow();
}

function addTrakeEvent() {
  syncTrakePlanDraft();
  if (trakePlanDraft.length >= 5) return;
  const order = trakePlanDraft.length + 1;
  trakePlanDraft.push({event_id: `e${order}`, order, text: 'new event', raw_text: 'new event', clip_query: 'new event', visual_query: 'new event', query_variants: ['new event'], modalities: ['clip'], required: true, object_constraints: [], attribute_constraints: [], ocr_constraints: [], asr_constraints: [], ocr_terms: [], asr_terms: [], min_gap_seconds: null, max_gap_seconds: null, confidence: 1, source: 'manual'});
  renderTrakeWorkflow();
}

async function replaceTrakeCandidate(eventId, candidateId) {
  try {
    trakeVerification = null;
    trakePayload = await post('/api/trake/replace', {session_id: trakePayload.state.session_id, chain_id: selectedTrakeChain.chain_id, event_id: eventId, candidate_id: candidateId, lock_event: true});
    selectedTrakeChain = trakePayload.state.manual_chain;
    renderTrakeWorkflow();
    message(`${eventId} replaced and locked after temporal validation.`);
  } catch (error) { message(error.message, true); }
}

async function refineTrakeCandidate(eventId) {
  try {
    const data = await post('/api/trake/refine', {session_id: trakePayload.state.session_id, chain_id: selectedTrakeChain.chain_id, event_id: eventId});
    trakeRefinements.set(eventId, data.refinement);
    renderTrakeWorkflow();
    message(`${eventId}: ${data.refinement.status}. ${data.refinement.reason || ''}`, data.refinement.status === 'REFINEMENT_UNAVAILABLE');
  } catch (error) { message(error.message, true); }
}

async function updateTrakePlan() {
  syncTrakePlanDraft();
  const events = trakePlanDraft.map((event) => {
    const text = event.text;
    return {...event, text, raw_text: text, clip_query: text, visual_query: text, query_variants: [text], source: 'manual'};
  });
  message('Saving the event plan and rerunning TRAKE retrieval...');
  trakePayload = await post('/api/trake/update-plan', {session_id: trakePayload.state.session_id, events, constraints: trakePayload.state.request.constraints || {}});
  selectedTrakeChain = null;
  trakePlanDraft = null;
  trakePlanDraftRevision = null;
  trakeOfficialFrames.clear();
  trakeRefinements.clear();
  trakeVerification = null;
  await searchAndAlignTrake(trakePayload.state.session_id);
  message(selectedTrakeChain ? 'Updated TRAKE plan produced a reviewable chain.' : 'Updated plan produced no valid same-video chain.', !selectedTrakeChain);
}

function updateTrakeButton() {
  const inputs = [...$('workflow-content').querySelectorAll('[data-trake-frame]')];
  const valid = inputs.length > 0 && inputs.every((input) => input.value !== '' && Number.isInteger(Number(input.value)) && Number(input.value) >= 0);
  if ($('trake-confirm-button')) $('trake-confirm-button').disabled = !selectedTrakeChain || !valid;
}

async function prepareTrake() {
  message('TRAKE is decomposing events and retrieving candidate sequences...');
  const planned = await post('/api/trake/plan', {query_id: $('query-id').value.trim(), query: $('query').value.trim(), constraints: {hybrid_retrieval: $('use-hybrid').checked, hybrid_use_gemini: $('use-gemini').checked}});
  trakeOfficialFrames.clear();
  trakeRefinements.clear();
  trakeVerification = null;
  trakePlanDraft = null;
  trakePlanDraftRevision = null;
  await searchAndAlignTrake(planned.state.session_id);
}

async function verifyTrakeWithGemini() {
  const button = $('trake-verify-button');
  if (button) button.disabled = true;
  message('Gemini is reviewing the allowlisted TRAKE frames...');
  try {
    trakeVerification = await post('/api/trake/verify', {session_id: trakePayload.state.session_id});
    renderTrakeWorkflow();
    const unavailable = ['DISABLED', 'UNAVAILABLE', 'SKIPPED_INVALID'].includes(trakeVerification.status);
    message(`TRAKE verifier: ${trakeVerification.status}. The server chain was not changed.`, unavailable);
  } catch (error) {
    message(error.message, true);
  } finally {
    if ($('trake-verify-button')) $('trake-verify-button').disabled = !allTrakeChains(trakePayload.state).length;
  }
}

async function searchAndAlignTrake(sessionId) {
  trakePayload = await post('/api/trake/search', {session_id: sessionId});
  try { trakePayload = await post('/api/trake/align', {session_id: sessionId}); }
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
  focusCsvPreview($('query-id').value.trim());
  message(`${$('query-id').value.trim()} TRAKE result added to the official queue.`);
}

$('start-button').onclick = async () => {
  try {
    csvEditorStates.clear();
    focusedCsvQueryId = null;
    renderSession(await post('/api/submission/session/start', {}));
    resetReviewWorkspace();
    message('Submission session started. Use a distinct Query ID for each queue item.');
  }
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
    const confirmed = await post('/api/submission/query/confirm', {session_id: session.session_id, query_id: agentResult.query_id, task: 'KIS', predictions, source});
    renderSession(confirmed);
    focusCsvPreview(agentResult.query_id);
    message(`${agentResult.query_id} confirmed to the official queue.`);
  } catch (error) { message(error.message, true); }
};

async function removeQuery(queryId) {
  try { renderSession(await post('/api/submission/query/remove', {session_id: session.session_id, query_id: queryId})); message(`${queryId} removed.`); }
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

function resetFormAndWorkspace() {
  $('query-id').value = '';
  $('query').value = '';
  if ($('qa-question')) $('qa-question').value = '';
  const defaultTask = document.querySelector('input[name="task"][value="KIS"]');
  if (defaultTask) defaultTask.checked = true;
  if ($('use-hybrid')) $('use-hybrid').checked = true;
  if ($('use-gemini')) $('use-gemini').checked = true;
  if ($('use-gemini-answer')) $('use-gemini-answer').checked = true;
  if ($('deep-frame-search')) $('deep-frame-search').checked = true;
  if ($('use-dense-frames')) $('use-dense-frames').checked = true;
  if ($('use-gemini-frame')) $('use-gemini-frame').checked = false;
  resetReviewWorkspace();
  updateTaskControls();
  message('Form and review workspace reset.');
}

if ($('reset-button')) $('reset-button').onclick = resetFormAndWorkspace;

for (const input of document.querySelectorAll('input[name="task"]')) input.addEventListener('change', () => { resetReviewWorkspace(); updateTaskControls(); });
updateTaskControls();
