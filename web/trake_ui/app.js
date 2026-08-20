const $ = id => document.getElementById(id);
let state = null;
let selectedChain = null;
let lastReview = null;
let hybridAvailable = false;

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'})[char]);
}

async function api(route, payload) {
  const response = await fetch(`/api/trake/${route}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.code ? `${data.code}: ${data.error}` : (data.error || 'Request failed'));
  return data;
}

function message(text, error = false) {
  $('message').textContent = text;
  $('message').className = error ? 'warnings' : '';
}

function eventCard(event) {
  const id = escapeHtml(event.event_id);
  return `<div class="event"><div class="event-head"><span class="marker">${Number(event.order)}</span><button data-delete="${id}">×</button></div>` +
    `<div class="event-row"><button data-up="${id}">↑</button><input data-text="${id}" value="${escapeHtml(event.text)}">` +
    `<select data-required="${id}"><option value="true" ${event.required ? 'selected' : ''}>required</option><option value="false" ${!event.required ? 'selected' : ''}>optional</option></select></div>` +
    `<label>Modalities<input data-modalities="${id}" value="${escapeHtml((event.modalities || []).join(','))}"></label><small>${escapeHtml(event.source)}</small></div>`;
}

function wireEditor() {
  document.querySelectorAll('[data-up]').forEach(button => button.onclick = () => {
    const index = state.request.events.findIndex(event => event.event_id === button.dataset.up);
    if (index > 0) [state.request.events[index - 1], state.request.events[index]] = [state.request.events[index], state.request.events[index - 1]];
    renumber(); render({state});
  });
  document.querySelectorAll('[data-delete]').forEach(button => button.onclick = () => {
    if (state.request.events.length > 1) {
      state.request.events = state.request.events.filter(event => event.event_id !== button.dataset.delete);
      renumber(); render({state});
    }
  });
}

function renumber() {
  state.request.events.forEach((event, index) => { event.order = index + 1; event.event_id = `e${index + 1}`; });
}

function keyframeImage(candidate, label) {
  if (!candidate || !candidate.keyframe_path) return '<div class="thumb missing">No image</div>';
  return `<img class="thumb" src="/keyframe?path=${encodeURIComponent(candidate.keyframe_path)}" alt="${escapeHtml(label)}" loading="lazy">`;
}

function videoCard(video) {
  const previews = (state.pools || []).map(pool => ({event: pool.event, candidate: pool.candidates.find(item => item.video_id === video.video_id)})).filter(item => item.candidate);
  const figures = previews.map(item => `<figure>${keyframeImage(item.candidate, `${video.video_id} ${item.event.event_id}`)}<figcaption>${escapeHtml(item.event.event_id)}<br>F${Number(item.candidate.keyframe_id)} · ${Number(item.candidate.pts_time).toFixed(1)}s</figcaption></figure>`).join('');
  return `<div class="video"><b>${escapeHtml(video.video_id)}</b><p>${Number(video.required_coverage)}/${Number(video.required_total)} required · ${video.complete ? 'complete' : 'incomplete'}</p><small>Per-event candidates only — not a validated TRAKE chain.</small><div class="preview-row">${figures || '<small>No keyframe candidate.</small>'}</div><p class="warnings">${escapeHtml((video.warnings || []).join(', '))}</p></div>`;
}

function chainCard(chain) {
  const selected = chain.events.filter(event => event.candidate);
  const max = Math.max(1, ...selected.map(event => event.candidate.pts_time));
  const score = chain.score_components?.final_rerank_score ?? chain.score.final_score;
  const dots = selected.map(event => `<i style="left:${Math.min(98, event.candidate.pts_time / max * 98)}%" title="${escapeHtml(event.event_id)} @ ${Number(event.candidate.pts_time)}s"></i>`).join('');
  const figures = selected.map(event => `<figure>${keyframeImage(event.candidate, `${chain.video_id} ${event.event_id}`)}<figcaption><b>${escapeHtml(event.event_id)}</b><br>frame ${Number(event.candidate.frame_idx)}<br>${Number(event.candidate.pts_time).toFixed(2)}s</figcaption></figure>`).join('');
  const gaps = selected.slice(1).map((event, index) => `${selected[index].event_id} → ${event.event_id}: Δt ${(event.candidate.pts_time - selected[index].candidate.pts_time).toFixed(2)}s`).join(' · ');
  const dense = selected.some(event => event.candidate.evidence?.dense_refinement);
  const verification = chain.score_components?.semantic_verification?.status;
  const status = dense
    ? 'Dense temporal/scene-consistent candidate — semantic action review required'
    : 'PTS-only candidate — continuity and semantic review required';
  const semanticNote = verification === 'REQUIRES_MANUAL_REVIEW'
    ? 'CLIP and scene continuity cannot prove that each requested action occurred.'
    : '';
  return `<div class="chain"><div class="chain-head"><b>${escapeHtml(chain.video_id)}</b><span class="score">${Number(score).toFixed(3)}</span></div><small>${status}</small><div class="timeline">${dots}</div><div class="chain-events">${figures}</div><p>${escapeHtml(gaps)}</p><p class="warnings">${escapeHtml(semanticNote)}</p><p class="warnings">${escapeHtml((chain.warnings || []).join(', '))}</p><details><summary>Score breakdown</summary><pre>${escapeHtml(JSON.stringify(chain.score_components || chain.score, null, 2))}</pre></details><button data-chain="${escapeHtml(chain.chain_id)}">Select chain</button></div>`;
}

function render(data) {
  state = data.state;
  $('events').innerHTML = state.request.events.map(eventCard).join('');
  wireEditor();
  $('search').disabled = false;
  $('save-plan').disabled = false;
  $('add-event').disabled = state.request.events.length >= 5;
  const retrievedVideoId = state.retrieved_video_id;
  const retrievedVideos = (state.videos || []).filter(video => !retrievedVideoId || video.video_id === retrievedVideoId);
  $('videos').innerHTML = retrievedVideos.map(videoCard).join('') || '<p>No single video satisfies all required events in temporal order.</p>';
  $('chains').innerHTML = (state.alignments || []).flatMap(alignment => alignment.chains.map(chainCard)).join('') || '<p>No chain yet. Run alignment.</p>';
  $('answer').textContent = state.trake_answer?.answer_text
    ? `Answer: ${state.trake_answer.answer_text}`
    : 'No TRAKE answer: one video with a temporally valid sequence was not found.';
  $('debug').textContent = JSON.stringify(state.diagnostics || {}, null, 2);
  document.querySelectorAll('[data-chain]').forEach(button => button.onclick = () => selectChain(button.dataset.chain));
}

function selectChain(id) {
  selectedChain = (state.alignments || []).flatMap(alignment => alignment.chains).find(chain => chain.chain_id === id);
  $('selection').textContent = selectedChain ? `${selectedChain.chain_id} · ${selectedChain.video_id}` : 'No chain selected';
  $('confirm').disabled = !selectedChain;
  $('reject').disabled = !selectedChain;
  if (!selectedChain) return;
  const pools = Object.fromEntries((state.pools || []).map(pool => [pool.event.event_id, pool.candidates.filter(candidate => candidate.video_id === selectedChain.video_id)]));
  $('corrections').innerHTML = selectedChain.events.map(event => {
    const eventId = escapeHtml(event.event_id);
    const options = (pools[event.event_id] || []).map(candidate => `<option value="${escapeHtml(candidate.candidate_id)}" ${event.candidate && candidate.candidate_id === event.candidate.candidate_id ? 'selected' : ''}>frame ${Number(candidate.keyframe_id)} @ ${Number(candidate.pts_time).toFixed(2)}s</option>`).join('');
    return `<label>${eventId}<select data-replace="${eventId}">${options}</select><button data-apply="${eventId}">Replace + lock</button><button data-refine="${eventId}">Refine</button></label>`;
  }).join('');
  document.querySelectorAll('[data-apply]').forEach(button => button.onclick = () => replaceEvent(button.dataset.apply));
  document.querySelectorAll('[data-refine]').forEach(button => button.onclick = () => refineEvent(button.dataset.refine));
}

$('plan').onclick = async () => {
  try {
    const queryId = $('query-id').value.trim();
    if (!queryId) throw new Error('Query ID is required.');
    const constraints = {
      hybrid_retrieval: hybridAvailable && $('hybrid-retrieval').checked,
      hybrid_use_gemini: $('gemini-planner').checked,
    };
    const data = await api('plan', {query_id: queryId, query: $('query').value, constraints});
    render(data);
    $('align').disabled = true;
    message(constraints.hybrid_retrieval ? 'Agent plan created; review event routes before retrieval.' : 'Baseline plan created; review events before retrieval.');
  } catch (error) { message(error.message, true); }
};
$('add-event').onclick = () => { if (state.request.events.length < 5) { const order = state.request.events.length + 1; state.request.events.push({event_id: `e${order}`, order, text: 'new event', clip_query: 'new event', modalities: ['clip'], required: true, object_constraints: [], attribute_constraints: [], ocr_constraints: [], asr_constraints: [], min_gap_seconds: null, max_gap_seconds: null, confidence: 1, source: 'manual'}); render({state}); } };
$('save-plan').onclick = async () => {
  try {
    state.request.events.forEach(event => {
      const id = CSS.escape(event.event_id);
      event.text = document.querySelector(`[data-text="${id}"]`).value;
      event.clip_query = event.text;
      event.required = document.querySelector(`[data-required="${id}"]`).value === 'true';
      event.modalities = document.querySelector(`[data-modalities="${id}"]`).value.split(',').map(value => value.trim()).filter(Boolean);
      event.source = 'manual';
    });
    const data = await api('update-plan', {session_id: state.session_id, events: state.request.events, constraints: state.request.constraints});
    render(data); $('align').disabled = true; message(`Saved plan revision ${state.plan_revision}; retrieve again.`);
  } catch (error) { message(error.message, true); }
};
$('search').onclick = async () => { try { const data = await api('search', {session_id: state.session_id}); render(data); $('align').disabled = !state.retrieved_video_id; const gate = state.diagnostics?.retrieval?.semantic_seed_gate; message(state.retrieved_video_id ? `Retrieved one feasible video: ${state.retrieved_video_id}.` : (gate?.enabled ? 'No TRAKE video: no path satisfies both semantic-seed quality and temporal order.' : 'No single video covers all required events in temporal order.')); } catch (error) { message(error.message, true); } };
$('align').onclick = async () => {
  const alignButton = $('align');
  alignButton.disabled = true;
  $('search').disabled = true;
  message('Aligning timeline and running bounded dense refinement…');
  try {
    const data = await api('align', {session_id: state.session_id});
    render(data);
    message(state.trake_answer?.answer_text ? `TRAKE answer ready: ${state.trake_answer.answer_text}` : 'No valid semantic-keyframe answer for the retrieved video.');
  } catch (error) {
    message(error.message, true);
  } finally {
    $('search').disabled = false;
    alignButton.disabled = !(state && state.pools && state.pools.length);
  }
};

async function review(decision) {
  try { const data = await api('review', {session_id: state.session_id, chain_id: selectedChain.chain_id, decision, reviewer: $('reviewer').value}); lastReview = data.review; $('export').disabled = decision !== 'confirmed'; message(`Review ${decision} appended.`); }
  catch (error) { message(error.message, true); }
}

$('confirm').onclick = () => review('confirmed');
$('reject').onclick = () => review('rejected');
$('export').onclick = async () => { try { await api('export', {session_id: state.session_id, review_id: lastReview.review_id}); message('Internal record exported; this is not the final contest format.'); } catch (error) { message(error.message, true); } };

async function loadHealth() {
  try {
    const response = await fetch('/api/trake/health');
    const health = await response.json();
    hybridAvailable = Boolean(health.hybrid_retrieval_enabled);
    $('hybrid-retrieval').disabled = !hybridAvailable;
    if (!hybridAvailable) $('hybrid-retrieval').checked = false;
    $('gemini-planner').disabled = !hybridAvailable || !health.gemini_configured;
    if (!health.gemini_configured) $('gemini-planner').checked = false;
  } catch (error) {
    hybridAvailable = false;
    $('hybrid-retrieval').disabled = true;
    $('gemini-planner').disabled = true;
  }
}

async function replaceEvent(eventId) {
  try { const candidateId = document.querySelector(`[data-replace="${CSS.escape(eventId)}"]`).value; const data = await api('replace', {session_id: state.session_id, chain_id: selectedChain.chain_id, event_id: eventId, candidate_id: candidateId, lock_event: true}); render(data); selectedChain = data.state.manual_chain; message(`${eventId} replaced and locked after invariant validation.`); }
  catch (error) { message(error.message, true); }
}

async function refineEvent(eventId) {
  try { const data = await api('refine', {session_id: state.session_id, chain_id: selectedChain.chain_id, event_id: eventId}); message(`${data.refinement.status}: ${data.refinement.reason}`); }
  catch (error) { message(error.message, true); }
}

loadHealth();
