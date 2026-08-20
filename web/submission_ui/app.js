const $ = (id) => document.getElementById(id);
let session = null;
let agentResult = null;

async function post(route, payload) {
  const response = await fetch(route, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
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

function renderSession(value) {
  session = value;
  $('session-status').textContent = value.status;
  $('session-id').textContent = value.session_id;
  $('run-button').disabled = value.status !== 'ACTIVE';
  $('validate-button').disabled = !value.queries.length;
  $('done-button').disabled = !value.queries.length || value.status !== 'ACTIVE';
  $('download-link').classList.toggle('disabled', value.status !== 'DONE');
  $('download-link').setAttribute('aria-disabled', String(value.status !== 'DONE'));
  $('download-link').href = value.status === 'DONE' ? `/api/submission/session/${encodeURIComponent(value.session_id)}/download` : '';
  $('queue-count').textContent = String(value.queries.length);
  renderQueue(value.queries);
  updateTaskControls();
}

function activeTask() {
  return document.querySelector('input[name="task"]:checked').value;
}

function updateTaskControls() {
  const task = activeTask();
  const isKis = task === 'KIS';
  $('workflow-import').hidden = isKis;
  $('query-id').disabled = !isKis;
  $('query').disabled = !isKis;
  $('query-id').required = isKis;
  $('query').required = isKis;
  $('use-hybrid').disabled = !isKis;
  $('use-gemini').disabled = !isKis;
  for (const id of ['workflow-session-id', 'workflow-review-id', 'workflow-frame-ids']) {
    $(id).disabled = isKis;
    $(id).required = !isKis;
  }
  $('official-frames-label').textContent = task === 'TRAKE' ? 'Official frame IDs (event order)' : 'Official frame ID';
  $('workflow-frame-ids').placeholder = task === 'TRAKE' ? '1200,1850,2100' : '1450';
  $('run-button').textContent = isKis ? 'Run Agent' : 'Import reviewed result';
  $('run-button').disabled = !session || session.status !== 'ACTIVE';
}

function renderQueue(queries) {
  $('queue').replaceChildren();
  for (const item of queries) {
    const row = document.createElement('div');
    row.className = 'queue-item';
    row.innerHTML = `<strong>${escapeHtml(item.query_id)}</strong><span>${escapeHtml(item.task)} · ${item.predictions.length} prediction · ${escapeHtml(item.output_file)}</span><div class="queue-actions"><button type="button" data-preview>Preview</button><button type="button" class="secondary" data-remove>Remove</button></div>`;
    row.querySelector('[data-preview]').onclick = () => { $('csv-preview').value = item.csv_preview; };
    row.querySelector('[data-remove]').onclick = () => removeQuery(item.query_id);
    $('queue').appendChild(row);
  }
  if (!queries.length) $('queue').innerHTML = '<p class="empty">No confirmed query.</p>';
}

function renderAgent(result) {
  agentResult = result;
  $('agent-trace').hidden = false;
  const plan = result.agent_plan || {};
  const agentTrace = result.agent_trace || {};
  const trace = [`Profile: ${plan.profile || 'baseline'}`, `Routes: ${(plan.enabled_retrievers || ['clip']).join(' + ')}`];
  for (const [channel, count] of Object.entries(result.channel_hit_counts || {})) trace.push(`${channel.toUpperCase()}: ${count} hits`);
  if (Object.keys(result.failures || {}).length) trace.push(`Failures: ${Object.keys(result.failures).join(', ')}`);
  $('agent-trace').innerHTML = trace.map((item) => `<span class="trace-pill">${escapeHtml(item)}</span>`).join('');
  $('agent-inspector').hidden = false;
  $('agent-profile').textContent = `${plan.intent || 'mixed'} · ${plan.profile || 'baseline'} · ${plan.planner_source || 'local'}`;
  $('clip-route').textContent = plan.visual_clip_query_en || '-';
  $('bge-route').textContent = plan.semantic_text_query || '-';
  $('bm25-route').textContent = plan.lexical_text_query || '-';
  $('gemini-state').textContent = agentTrace.status || 'Not called';
  $('gemini-state').className = `state-badge is-${String(agentTrace.status || 'idle').toLowerCase()}`;
  $('gemini-model').textContent = [agentTrace.provider, agentTrace.model].filter(Boolean).join(' / ') || '-';
  $('gemini-latency').textContent = Number.isFinite(Number(agentTrace.latency_ms)) ? `${Number(agentTrace.latency_ms).toFixed(1)} ms` : '-';
  $('gemini-raw').textContent = agentTrace.raw_text || 'No raw Gemini response. The local planner was used.';
  $('gemini-parsed').textContent = agentTrace.parsed_output ? JSON.stringify(agentTrace.parsed_output, null, 2) : '-';
  $('validated-plan').textContent = JSON.stringify(agentTrace.validated_plan || plan, null, 2);
  const warning = agentTrace.fallback || (agentTrace.warnings || []).join('; ');
  $('gemini-warning').hidden = !warning;
  $('gemini-warning').textContent = warning ? `Fallback / warning: ${warning}` : '';
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
  if (candidate.source_type || candidate.matched_text) {
    chips.push(`<span class="evidence-chip is-source">${escapeHtml(candidate.source_type || 'evidence')}${candidate.matched_text ? ` · ${escapeHtml(candidate.matched_text.slice(0, 120))}` : ''}</span>`);
  }
  return chips.join('');
}

function candidateRow(candidate) {
  const row = document.createElement('article');
  row.className = 'candidate';
  row.dataset.candidate = JSON.stringify(candidate);
  const metadata = candidate.metadata || {};
  const image = candidate.image_url
    ? `<img src="${escapeHtml(candidate.image_url)}" alt="${escapeHtml(candidate.video_id)} keyframe ${escapeHtml(candidate.keyframe_id)}" loading="lazy">`
    : '<div class="missing-image">Preview unavailable</div>';
  const openVideo = candidate.video_url
    ? `<a class="candidate-action" href="${escapeHtml(candidate.video_url)}" target="_blank" rel="noreferrer">Open video</a>`
    : '';
  row.innerHTML = `
    <div class="candidate-media">${image}<span class="rank-badge">#${Number(candidate.rank)}</span></div>
    <div class="candidate-main">
      <div class="candidate-title"><strong>${escapeHtml(candidate.video_id)}</strong><span>RRF ${Number(candidate.retrieval?.score || 0).toFixed(4)}</span></div>
      <p>${escapeHtml(metadata.title || 'Untitled video')}</p>
      <dl class="candidate-facts">
        <div><dt>Keyframe</dt><dd>${escapeHtml(candidate.keyframe_id)}</dd></div>
        <div><dt>Frame index</dt><dd>${escapeHtml(candidate.frame_idx)}</dd></div>
        <div><dt>Timestamp</dt><dd>${Number(candidate.pts_time || 0).toFixed(2)}s</dd></div>
        <div><dt>Author</dt><dd>${escapeHtml(metadata.author || '-')}</dd></div>
      </dl>
      <div class="candidate-evidence">${candidateEvidence(candidate)}</div>
      <div class="candidate-actions">${openVideo}<button class="neighbor-button secondary" type="button">Nearby frames</button></div>
    </div>
    <aside class="candidate-review">
      <label class="select-control"><input class="candidate-select" type="checkbox"><span>Select candidate</span></label>
      <label class="official-input"><span>Verified official frame ID</span><input type="number" min="0" step="1" placeholder="Required before confirm"></label>
      <small>Keyframe and frame index above are retrieval diagnostics only.</small>
    </aside>
    <div class="neighbors" hidden></div>`;
  row.querySelector('.candidate-select').setAttribute('aria-label', `Select ${candidate.video_id}`);
  row.querySelector('.candidate-select').onchange = updateSelection;
  row.querySelector('.official-input input').oninput = updateSelection;
  row.querySelector('.neighbor-button').onclick = () => toggleNeighbors(row, candidate);
  return row;
}

async function toggleNeighbors(row, candidate) {
  const panel = row.querySelector('.neighbors');
  if (!panel.hidden) { panel.hidden = true; return; }
  panel.hidden = false;
  panel.textContent = 'Loading...';
  try {
    const params = new URLSearchParams({video_id: candidate.video_id, keyframe_id: candidate.keyframe_id, radius: '3'});
    const response = await fetch(`/api/neighborhood?${params}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Neighborhood failed');
    panel.innerHTML = data.frames.map((frame) => `<div class="neighbor"><img src="${escapeHtml(frame.image_url || '')}" alt="frame ${Number(frame.keyframe_id)}"><small>K${Number(frame.keyframe_id)} · F${Number(frame.frame_idx)}</small></div>`).join('');
  } catch (error) { panel.textContent = error.message; }
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

$('start-button').onclick = async () => {
  try { renderSession(await post('/api/submission/session/start', {})); agentResult = null; $('candidates').innerHTML = '<p class="empty">No candidate result.</p>'; message('Submission session started.'); }
  catch (error) { message(error.message, true); }
};

$('agent-form').onsubmit = async (event) => {
  event.preventDefault();
  if (!session) return;
  $('run-button').disabled = true;
  try {
    const task = activeTask();
    if (task === 'KIS') {
      const result = await post('/api/submission/agent/run', {session_id: session.session_id, query_id: $('query-id').value.trim(), task: 'KIS', query: $('query').value.trim(), use_hybrid: $('use-hybrid').checked, use_gemini: $('use-gemini').checked});
      renderAgent(result);
      message(result.mapping_warning);
    } else {
      const rawFrames = $('workflow-frame-ids').value.split(',').map((value) => value.trim()).filter(Boolean);
      if (!rawFrames.length || rawFrames.some((value) => !/^\d+$/.test(value))) throw new Error('Official frame IDs must be non-negative integers.');
      const base = {session_id: session.session_id, review_id: $('workflow-review-id').value.trim(), mapping_source: 'manual_official'};
      const payload = task === 'QA'
        ? await post('/api/submission/import/qa', {...base, qa_session_id: $('workflow-session-id').value.trim(), official_frame_id: Number(rawFrames[0])})
        : await post('/api/submission/import/trake', {...base, trake_session_id: $('workflow-session-id').value.trim(), official_frame_ids: rawFrames.map(Number)});
      renderSession(payload);
      $('csv-preview').value = payload.queries[payload.queries.length - 1]?.csv_preview || '';
      message(`${task} reviewed result imported to the official queue.`);
    }
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

for (const input of document.querySelectorAll('input[name="task"]')) input.addEventListener('change', updateTaskControls);
updateTaskControls();
