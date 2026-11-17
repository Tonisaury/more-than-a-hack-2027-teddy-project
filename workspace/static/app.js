/* Possession Workspace client: the canvas of cards, the realtime voice call and the typed fallback.
   Voice: WebRTC straight to the realtime model, with the SDP exchanged through the local server (the key stays
   there). The model calls tools; this page runs them against the local server, draws the card, and hands the
   model only the tool's summary. Typed questions use the same tools through the server's Responses loop, or the
   live call when one is open. */
(function () {
  'use strict';
  const { el, icon } = window.Blocks;
  const $ = (id) => document.getElementById(id);
  const state = { cards: new Map(), config: null, peer: null, channel: null, stream: null, silent: null, generation: 0, connecting: false, muted: false,
    listenOnly: false, history: [], partials: new Map(), busy: false, contextTimer: 0, captionTimer: 0, handled: new Set(),
    protocol: 'realtime', pending: 0, continueTimer: 0, events: [] };

  const QUICK = [
    ['book', 'The goal', 'show_explainer', { topic: 'goal' }], ['target', 'The score', 'show_explainer', { topic: 'scoring' }], ['flow', 'What was built', 'show_explainer', { topic: 'pipeline' }],
    ['timeline', 'Public 5 minutes', 'show_timeline', { half: 1, from_minute: 0, to_minute: 5, rows: ['truth', 'ours', 'baseline'], lanes: [] }],
    ['eye-off', 'Missing ball', 'show_ball_coverage', {}], ['shield', 'Validate', 'show_validation', {}], ['trophy', 'Scoreboard', 'show_scoreboard', {}],
  ];
  const STARTERS = [
    ['Explain the goal like I am five', 'show_explainer', { topic: 'goal' }],
    ['What did we build, and why two views?', 'show_explainer', { topic: 'pipeline' }],
    ['Show the five minutes we can check and where we went wrong', 'show_turn_matching', { variant: 'ours' }],
    ['Can we validate this result somehow?', 'show_validation', {}],
    ['What do we answer when the ball is missing?', 'show_ball_coverage', {}],
  ];
  const TOUR = [
    ['show_explainer', { topic: 'goal' }, 'The task: one letter for every frame. A run of one colour is one possession, and the judges count possessions.'],
    ['show_overview', {}, 'Where the entry stands: two scores on the five public minutes, and the whole match as we predict it.'],
    ['show_turn_matching', { variant: 'ours' }, 'Possession by possession. Each true one is linked to the one we paired with it; dashed red ones were missed.'],
    ['show_explainer', { topic: 'pipeline' }, 'How it works: a ball view, a player view that never sees the ball, and a patient referee that joins them.'],
    ['show_evidence', { half: 1, from_minute: 2.5, to_minute: 5 }, 'Why it decided what it decided: the answers on top, the evidence each view had underneath. Hover to read any instant.'],
    ['show_ball_coverage', {}, 'The hard part: the ball is missing on four graded frames in ten, in long stretches.'],
    ['show_validation', {}, 'What can be checked without a second answer key, and what the rules lock away.'],
  ];

  const GALLERY = [                                      // /?gallery: one of every kind of card, without the assistant
    ['show_pitch', { half: 1, minute: 2.5, replay_seconds: 10 }], ['show_scoreboard', {}], ['show_confusion', { variant: 'ours' }],
    ['show_breakdown', { rows: 'period15', columns: 'label', half: 0, only_label: 'in_play', measure: 'share_of_row', chart: 'bars' }],
    ['show_breakdown', { rows: 'ball_view', columns: 'player_view', half: 0, only_label: 'any', measure: 'share_of_row', chart: 'heatmap' }],
    ['show_breakdown', { rows: 'ball_seen', columns: 'label', half: 0, only_label: 'any', measure: 'minutes', chart: 'bars' }],
    ['show_durations', { kind: 'possessions', team: 'both' }], ['show_tags', {}], ['show_explainer', { topic: 'rules' }], ['show_explainer', { topic: 'scoring' }],
    ['show_html', { title: 'Sample custom visual', html: '<h2>Possession in the public five minutes</h2><div class="strip"><span class="h" style="width:46%"></span><span class="a" style="width:34%"></span><span class="d" style="width:20%"></span></div><div class="kpi-row"><div class="kpi"><span>Barcelona</span><b>46%</b></div><div class="kpi"><span>Athletic</span><b>34%</b></div><div class="kpi"><span>Dead ball</span><b>20%</b></div></div><p class="note">Truth labels, frames 11,185 to 18,684.</p>' }],
  ];

  // ------------------------------------------------------------------ small helpers
  for (const n of document.querySelectorAll('[data-icon]')) n.appendChild(icon(n.dataset.icon));
  async function api(path, body) {
    const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Workspace': '1' }, body: JSON.stringify(body) });
    let data = null; try { data = await r.json(); } catch (_) { /* handled below */ }
    if (!r.ok) throw new Error((data && data.error) || 'The workspace server did not answer (' + r.status + ').');
    return data;
  }
  function alertBar(text) { $('vw-alert').hidden = !text; $('vw-alert-text').textContent = text || ''; }
  function status(text) { $('vw-status').textContent = text; }
  function addMessage(role, text, name) {
    const p = el('p', 'vw-message ' + role); p.appendChild(el('b', null, name || (role === 'user' ? 'You' : 'Assistant')));
    const parts = String(text).split('**');                              // **bold** from typed answers; text nodes only
    if (role === 'user' || parts.length < 3) p.appendChild(document.createTextNode(text)); else parts.forEach((part, i) => { if (part) p.appendChild(i % 2 ? el('strong', null, part) : document.createTextNode(part)); });
    const box = $('vw-messages'); box.querySelector('.vw-welcome')?.remove(); box.appendChild(p); box.scrollTop = box.scrollHeight; return p;
  }
  function caption(role, text, hold) {
    const box = $('vw-caption'); clearTimeout(state.captionTimer);
    if (!text.trim() || !$('vw-transcript').hidden) { box.hidden = true; return; }
    box.dataset.role = role; box.querySelector('b').textContent = role === 'user' ? 'You' : 'Assistant'; box.querySelector('span').textContent = text.length > 260 ? '…' + text.slice(-260) : text; box.hidden = false;
    state.captionTimer = setTimeout(() => { box.hidden = true; }, hold || 9000);
  }
  function remember(role, content) { state.history.push({ role, content: content.slice(0, 1500) }); state.history = state.history.slice(-12); }
  function channelSend(value) { if (state.channel && state.channel.readyState === 'open') state.channel.send(JSON.stringify(value)); }

  // ------------------------------------------------------------------ the canvas
  function visible() { return [...state.cards.values()].map((c) => ({ id: c.data.id, title: c.data.title, pinned: c.pinned })); }
  function shareContext() {
    clearTimeout(state.contextTimer);
    if (state.protocol === 'live') return;              // in a live session every tool result already carries its card id
    state.contextTimer = setTimeout(() => channelSend({ type: 'conversation.item.create', item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Cards now on the canvas (context only, not a request): ' + JSON.stringify(visible()) }] } }), 600);
  }
  function layout() {
    const choice = $('vw-layout').value, width = $('vw-stage').clientWidth || window.innerWidth;
    $('vw-stage').style.setProperty('--vw-columns', choice === 'auto' ? (width < 1000 ? 1 : width < 1750 ? 2 : 3) : choice);
  }
  function show() { const any = state.cards.size > 0; $('vw-empty').hidden = any; $('vw-stage').hidden = !any; layout(); }
  function flash(node) { node.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); node.classList.add('flash'); setTimeout(() => node.classList.remove('flash'), 1400); }
  function removeCard(id) { const c = state.cards.get(id); if (!c) return; c.node.remove(); state.cards.delete(id); $('vw-stage').classList.toggle('has-solo', !!$('vw-stage').querySelector('.solo')); show(); shareContext(); }
  function setPinned(id, on) { const c = state.cards.get(id); if (!c) return; c.pinned = on; const b = c.node.querySelector('[data-act="pin"]'); b.setAttribute('aria-pressed', String(on)); b.title = on ? 'Unpin' : 'Pin: keep when clearing'; }
  function setSolo(id, on) { for (const c of state.cards.values()) c.node.classList.toggle('solo', on && c.data.id === id); $('vw-stage').classList.toggle('has-solo', on); const c = state.cards.get(id); if (c) c.node.querySelector('[data-act="expand"]').setAttribute('aria-pressed', String(on)); }
  function clearCards() { for (const [id, c] of [...state.cards]) if (!c.pinned) { c.node.remove(); state.cards.delete(id); } $('vw-stage').classList.remove('has-solo'); show(); shareContext(); }
  function addCard(card) {
    const node = el('article', 'vw-card' + (window.Blocks.wide(card) ? ' wide' : '') + (window.Blocks.tall(card) ? ' tall' : '')); node.dataset.kind = card.icon || 'chart'; node.id = 'card-' + card.id; node.setAttribute('aria-label', card.title);
    const head = el('header', 'vw-card-head'), ic = el('span', 'vw-card-icon'); ic.appendChild(icon(card.icon)); head.appendChild(ic);
    const heading = el('div', 'vw-card-heading'); heading.appendChild(el('h3', null, card.title)); if (card.subtitle) heading.appendChild(el('div', 'vw-card-meta', card.subtitle)); head.appendChild(heading);
    const actions = el('div', 'vw-card-actions');
    const button = (act, name, label) => { const b = el('button', 'vw-icon-button'); b.type = 'button'; b.dataset.act = act; b.title = label; b.setAttribute('aria-label', label + ': ' + card.title); b.appendChild(icon(name)); actions.appendChild(b); return b; };
    button('pin', 'pin', 'Pin: keep when clearing').addEventListener('click', () => { setPinned(card.id, !state.cards.get(card.id).pinned); shareContext(); });
    button('expand', 'expand', 'Expand').addEventListener('click', () => setSolo(card.id, !node.classList.contains('solo')));
    button('close', 'close', 'Close').addEventListener('click', () => removeCard(card.id));
    head.appendChild(actions); node.appendChild(head);
    const body = el('div', 'vw-card-body'); for (const block of card.blocks) body.appendChild(window.Blocks.render(block)); node.appendChild(body);
    const foot = el('footer', 'vw-card-foot'); foot.appendChild(el('i')); foot.appendChild(el('span', null, card.foot || '')); node.appendChild(foot);
    state.cards.set(card.id, { data: card, node, pinned: false });
    $('vw-stage').classList.remove('has-solo'); for (const c of state.cards.values()) c.node.classList.remove('solo');
    $('vw-stage').appendChild(node); show(); flash(node); shareContext();
    $('vw-live').textContent = 'New card: ' + card.title;
  }
  function workspaceAction(a) {
    if (!a) return;
    const id = a.card_id, has = state.cards.has(id);
    if (a.action === 'clear') clearCards();
    else if (a.action.startsWith('layout_')) { $('vw-layout').value = { layout_auto: 'auto', layout_one: '1', layout_two: '2', layout_three: '3' }[a.action]; layout(); }
    else if (!has) return;
    else if (a.action === 'focus') { setSolo(id, false); flash(state.cards.get(id).node); }
    else if (a.action === 'pin' || a.action === 'unpin') setPinned(id, a.action === 'pin');
    else if (a.action === 'expand') setSolo(id, true);
    else if (a.action === 'close') removeCard(id);
  }
  function apply(result) { if (result.card) addCard(result.card); if (result.action) workspaceAction(result.action); }
  async function manual(name, args) { try { alertBar(''); const r = await api('/api/tools', { name, arguments: args }); if (r.status !== 'ok') throw new Error(r.error); apply(r); return r; } catch (e) { alertBar(e.message); return null; } }

  // ------------------------------------------------------------------ asking
  async function ask(text) {
    text = text.trim(); if (!text) return;
    alertBar('');
    if (state.channel && state.channel.readyState === 'open') {             // a live call is open: the same assistant answers aloud
      addMessage('user', text); remember('user', text); caption('user', text, 4000); $('vw-input').value = '';
      if (state.protocol === 'live') { finishPartials(); channelSend({ type: 'response.item.create', item: { type: 'message', role: 'user', content: [{ type: 'input_text', text }] } }); channelSend({ type: 'response.create' }); }
      else { interrupt(); channelSend({ type: 'conversation.item.create', item: { type: 'message', role: 'user', content: [{ type: 'input_text', text }] } }); channelSend({ type: 'response.create' }); }
      status('Thinking…'); return;
    }
    if (state.busy) { alertBar('Wait for this answer first.'); return; }
    if (!state.config?.configured) { alertBar('No OpenAI key on the server, so the assistant is off. The quick cards and the tour still work.'); return; }
    state.busy = true; $('vw-send').disabled = true; const history = state.history.slice();
    addMessage('user', text); remember('user', text); $('vw-input').value = ''; status('Thinking…');
    try {
      const result = await api('/api/message', { message: text, history, cards: visible() });
      for (const a of result.actions || []) { if (a.status === 'ok') apply(a); else if (a.error) alertBar(a.error); }
      const reply = result.reply || 'The cards you asked for are on the canvas.';
      addMessage('assistant', reply); remember('assistant', reply); caption('assistant', reply.replaceAll('**', ''), 14000); $('vw-live').textContent = reply.replaceAll('**', '');
    } catch (e) { alertBar(e.message); } finally { state.busy = false; $('vw-send').disabled = false; status(state.peer ? 'Listening' : 'Ready when you are'); }
  }
  async function starter(prompt, tool, args) { if (state.config?.configured) return ask(prompt); return manual(tool, args); }
  async function tour() {
    alertBar('');
    for (const [name, args, line] of TOUR) { const r = await manual(name, args); if (!r) return; addMessage('assistant', line, 'Guide'); await new Promise((ok) => setTimeout(ok, 350)); }
    transcript(true); $('vw-stage').scrollTo({ top: 0, behavior: 'smooth' });
  }

  // ------------------------------------------------------------------ realtime events
  function interrupt() { channelSend({ type: 'response.cancel' }); channelSend({ type: 'output_audio_buffer.clear' }); }
  async function runCalls(calls, gen) {
    status('Updating the canvas…');
    for (const call of calls.slice(0, 8)) {
      if (gen !== state.generation || state.handled.has(call.call_id)) continue;
      state.handled.add(call.call_id);
      let model;
      try { const r = await api('/api/tools', { name: call.name, arguments: JSON.parse(call.arguments || '{}') }); apply(r); model = r.model; if (r.status !== 'ok') alertBar(r.error); }
      catch (e) { model = { status: 'failed', error: e.message }; alertBar(e.message); }
      if (gen !== state.generation) return;
      channelSend({ type: 'conversation.item.create', item: { type: 'function_call_output', call_id: call.call_id, output: JSON.stringify(model) } });
    }
    if (gen === state.generation) channelSend({ type: 'response.create' });
  }
  // ---- Live API (gpt-live-*): the live model speaks; a Responses model behind it calls our tools.
  function finishPartials(role) {
    for (const [key, p] of [...state.partials]) { if (role && key !== role) continue; clearTimeout(p.timer); if (p.text.trim()) remember(key, p.text.trim()); else p.node.remove(); state.partials.delete(key); }
  }
  function livePartial(role, delta) {
    if (!delta) return;
    finishPartials(role === 'user' ? 'assistant' : 'user');
    let p = state.partials.get(role); if (!p) { p = { node: addMessage(role, ''), text: '', timer: 0 }; state.partials.set(role, p); }
    p.text += delta; p.node.lastChild.textContent = p.text; $('vw-messages').scrollTop = $('vw-messages').scrollHeight;
    caption(role, p.text, role === 'user' ? 4000 : 9000); status(role === 'user' ? 'Listening to you…' : 'Assistant is speaking…');
    clearTimeout(p.timer); p.timer = setTimeout(() => { finishPartials(role); status(state.listenOnly ? 'Type your question' : state.muted ? 'Microphone muted' : 'Listening'); }, 2600);
  }
  async function liveCall(item, gen) {
    if (state.handled.has(item.call_id)) return;
    state.handled.add(item.call_id); state.pending++; clearTimeout(state.continueTimer); status('Updating the canvas…');
    let model;
    try { const r = await api('/api/tools', { name: item.name, arguments: JSON.parse(item.arguments || '{}') }); apply(r); model = r.model; if (r.status !== 'ok') alertBar(r.error); }
    catch (e) { model = { status: 'failed', error: e.message }; alertBar(e.message); }
    state.pending--;
    if (gen !== state.generation) return;
    channelSend({ type: 'response.item.create', item: { type: 'function_call_output', call_id: item.call_id, output: JSON.stringify(model) } });
    // continue the backend's turn once every call it made has its result
    clearTimeout(state.continueTimer);
    state.continueTimer = setTimeout(() => { if (!state.pending && gen === state.generation) channelSend({ type: 'response.create' }); }, 350);
  }
  async function onLiveEvent(v, gen) {
    const t = v.type;
    if (t === 'session.started') status(state.listenOnly ? 'Type your question' : 'Listening');
    else if (t === 'session.input_transcript.delta') livePartial('user', v.delta);
    else if (t === 'session.output_transcript.delta') livePartial('assistant', v.delta);
    else if (t === 'session.input_transcript.done' || t === 'session.input_transcript.completed') finishPartials('user');
    else if (t === 'session.output_transcript.done' || t === 'session.output_transcript.completed') finishPartials('assistant');
    else if (t === 'response.event' && v.event) {
      const e = v.event;
      if (e.type === 'response.output_item.done' && e.item?.type === 'function_call') await liveCall(e.item, gen);
      else if (e.type === 'response.failed' || e.type === 'error') alertBar('The assistant behind the voice reported: ' + (e.response?.error?.message || e.error?.message || e.message || 'an error') + '.');
    }
    else if (t === 'error') { alertBar('The voice service reported: ' + (v.error?.message || v.message || 'an error') + '.'); }
    else if (t === 'session.closed') { if (gen === state.generation) stopVoice('Conversation ended'); }
  }
  async function onEvent(v, gen) {
    if (gen !== state.generation) return;
    state.events.push(v.type + (v.event ? ' > ' + v.event.type + (v.event.item ? ' [' + v.event.item.type + ']' : '') : '') + (v.type === 'error' ? ' ' + JSON.stringify(v.error || v).slice(0, 300) : '')); if (state.events.length > 400) state.events.shift();
    if (state.protocol === 'live') return onLiveEvent(v, gen);
    const t = v.type;
    if (t === 'input_audio_buffer.speech_started') { meter.speaking = true; status('Listening to you…'); }
    else if (t === 'input_audio_buffer.speech_stopped') { meter.speaking = false; status('Thinking…'); }
    else if (t === 'conversation.item.input_audio_transcription.completed') { if (v.transcript && v.transcript.trim()) { addMessage('user', v.transcript.trim()); remember('user', v.transcript.trim()); caption('user', v.transcript.trim(), 4000); } }
    else if (t === 'response.output_audio_transcript.delta' || t === 'response.audio_transcript.delta' || t === 'response.output_text.delta') {
      let p = state.partials.get(v.item_id); if (!p) { p = { node: addMessage('assistant', ''), text: '' }; state.partials.set(v.item_id, p); }
      p.text += v.delta; p.node.lastChild.textContent = p.text; $('vw-messages').scrollTop = $('vw-messages').scrollHeight; caption('assistant', p.text); status('Assistant is speaking…');
    }
    else if (t === 'response.output_audio_transcript.done' || t === 'response.audio_transcript.done' || t === 'response.output_text.done') { const p = state.partials.get(v.item_id); if (p) { remember('assistant', p.text); state.partials.delete(v.item_id); } }
    else if (t === 'output_audio_buffer.stopped') status(state.listenOnly ? 'Type your question' : state.muted ? 'Microphone muted' : 'Listening');
    else if (t === 'error') { if (!['response_cancel_not_active', 'output_audio_buffer_clear_not_active'].includes(v.error?.code)) alertBar('The voice service reported: ' + (v.error?.message || 'an error') + '.'); }
    else if (t === 'response.done') {
      const calls = (v.response?.output || []).filter((x) => x.type === 'function_call');
      if (calls.length) await runCalls(calls, gen);
      else if (v.response?.status === 'failed') alertBar('The assistant could not answer: ' + (v.response.status_details?.error?.message || 'unknown error') + '.');
      else if (!state.partials.size) status(state.listenOnly ? 'Type your question' : state.muted ? 'Microphone muted' : 'Listening');
    }
  }

  // ------------------------------------------------------------------ the level meter (nothing it measures leaves the page)
  const meter = { ctx: null, mic: null, out: null, frame: 0, speaking: false, heights: new Array(19).fill(3) };
  function listen(stream) { meter.ctx = meter.ctx || new AudioContext(); meter.ctx.resume?.(); const src = meter.ctx.createMediaStreamSource(stream), an = meter.ctx.createAnalyser(); an.fftSize = 256; src.connect(an); return an; }
  function level(an) { if (!an) return 0; const d = new Uint8Array(an.fftSize); an.getByteTimeDomainData(d); let s = 0; for (const x of d) s += (x - 128) * (x - 128); return Math.min(1, Math.sqrt(s / d.length) / 40); }
  function drawMeter() {
    const canvas = $('vw-wave-canvas'), g = canvas.getContext('2d'), css = getComputedStyle(document.documentElement);
    const out = level(meter.out), mic = state.muted || state.listenOnly ? 0 : level(meter.mic);
    const who = out > 0.04 ? 'assistant' : (mic > 0.06 && meter.speaking) ? 'user' : null, lvl = who === 'assistant' ? out : who === 'user' ? mic : 0;
    const wave = $('vw-wave'), tagEl = $('vw-speaker');
    if (who) wave.dataset.speaker = who; else delete wave.dataset.speaker;
    tagEl.hidden = !who && !state.muted; tagEl.textContent = who === 'assistant' ? 'Assistant' : who === 'user' ? 'You' : 'Muted';
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.fillStyle = css.getPropertyValue(who === 'assistant' ? '--vw-wave-assistant' : who === 'user' ? '--vw-wave-user' : '--vw-wave-idle');
    const n = meter.heights.length, w = canvas.width / n, now = performance.now() / 160;
    for (let i = 0; i < n; i++) {
      const target = who ? 6 + lvl * (canvas.height - 10) * (0.45 + 0.55 * Math.abs(Math.sin(now + i * 0.9))) * (1 - Math.abs(i - n / 2) / n) : 4;
      meter.heights[i] += (target - meter.heights[i]) * 0.35;
      const h = meter.heights[i]; g.beginPath(); g.roundRect(i * w + w * 0.25, (canvas.height - h) / 2, w * 0.5, h, w * 0.25); g.fill();
    }
    meter.frame = requestAnimationFrame(drawMeter);
  }
  function startMeter(stream) { try { meter.mic = stream ? listen(stream) : null; } catch (_) { meter.mic = null; } cancelAnimationFrame(meter.frame); drawMeter(); }

  // ------------------------------------------------------------------ the voice call
  function voiceUI(on) {
    $('vw-connect').classList.toggle('active', on); $('vw-connect-label').textContent = on ? 'End conversation' : 'Start talking';
    $('vw-mute').hidden = !on || state.listenOnly; $('vw-interrupt').hidden = !on || state.protocol === 'live'; if (!on) $('vw-play').hidden = true;
  }
  function silentStream() {
    const ctx = new AudioContext(), dst = ctx.createMediaStreamDestination(), osc = ctx.createOscillator(), gain = ctx.createGain();
    gain.gain.value = 0; osc.connect(gain).connect(dst); osc.start(); state.silent = ctx; return dst.stream;
  }
  async function stopVoice(note) {
    state.generation++; state.connecting = false;
    const had = !!state.peer;
    if (state.protocol === 'live') channelSend({ type: 'session.close' });
    clearTimeout(state.continueTimer); state.pending = 0;
    for (const p of state.partials.values()) clearTimeout(p.timer);
    try { state.channel?.close(); } catch (_) { /* already closed */ }
    try { state.peer?.close(); } catch (_) { /* already closed */ }
    state.stream?.getTracks().forEach((t) => t.stop()); state.silent?.close().catch(() => {});
    state.peer = state.channel = state.stream = state.silent = null; state.muted = false; state.listenOnly = false; state.partials.clear(); meter.mic = meter.out = null; meter.speaking = false;
    $('vw-audio').srcObject = null; $('vw-mute').setAttribute('aria-pressed', 'false'); $('vw-mute').textContent = 'Mute';
    $('vw-connect').disabled = false; voiceUI(false); status(note || 'Ready when you are');
    if (had && state.protocol !== 'live') api('/api/voice/close', {}).catch(() => {});
  }
  async function connect(forceListenOnly) {
    if (state.peer || state.connecting) { await stopVoice(); return; }
    if (!state.config?.configured) { alertBar('No OpenAI key on the server. Start it with --env-file to talk; the cards and the tour work without it.'); return; }
    if (!window.RTCPeerConnection) { alertBar('This browser has no WebRTC. Type your questions instead.'); return; }
    const gen = ++state.generation; state.connecting = true; status('Connecting…'); $('vw-connect').disabled = true; alertBar('');
    try {
      let stream = null;
      if (!forceListenOnly && navigator.mediaDevices?.getUserMedia) {
        try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } }); }
        catch (e) { alertBar((e.name === 'NotAllowedError' ? 'Microphone permission was denied' : 'No microphone was found') + '. Connected in listen-only mode: type your questions and the assistant will answer aloud.'); }
      }
      if (gen !== state.generation) { stream?.getTracks().forEach((t) => t.stop()); return; }
      state.listenOnly = !stream; if (!stream) stream = silentStream();
      state.stream = stream; startMeter(state.listenOnly ? null : stream);
      const pc = new RTCPeerConnection(); state.peer = pc; const dc = pc.createDataChannel('oai-events'); state.channel = dc;
      stream.getTracks().forEach((t) => pc.addTrack(t, stream));
      pc.ontrack = (e) => { $('vw-audio').srcObject = e.streams[0]; try { meter.out = listen(e.streams[0]); } catch (_) { /* meter only */ } $('vw-audio').play().catch(() => { $('vw-play').hidden = false; alertBar('Choose Enable audio to hear the assistant.'); }); };
      dc.onmessage = (e) => { let v; try { v = JSON.parse(e.data); } catch (_) { return; } onEvent(v, gen).catch((err) => alertBar(err.message)); };
      dc.onopen = () => { state.connecting = false; voiceUI(true); $('vw-connect').disabled = false; status(state.listenOnly ? 'Type your question' : 'Listening'); shareContext(); };
      dc.onclose = () => { if (gen === state.generation) stopVoice('Conversation ended'); };
      pc.onconnectionstatechange = () => { if (['failed', 'disconnected'].includes(pc.connectionState) && gen === state.generation) { alertBar('Voice disconnected. Start the conversation again to reconnect.'); stopVoice(); } };
      const offer = await pc.createOffer(); await pc.setLocalDescription(offer);
      const answer = await api('/api/voice', { sdp: offer.sdp, sensitivity: $('vw-sensitivity').value, microphone: $('vw-microphone').value });
      if (gen !== state.generation) return;
      state.protocol = answer.protocol || 'realtime';
      await pc.setRemoteDescription({ type: 'answer', sdp: answer.sdp });
    } catch (e) { await stopVoice(); alertBar(e.message); }
  }
  function updateAudio() {
    if (state.protocol === 'live') return;              // the Live API handles turn-taking itself
    const threshold = { balanced: 0.55, soft: 0.40, noisy: 0.70 }[$('vw-sensitivity').value];
    channelSend({ type: 'session.update', session: { type: 'realtime', audio: { input: { noise_reduction: { type: $('vw-microphone').value },
      turn_detection: { type: 'server_vad', threshold, prefix_padding_ms: 300, silence_duration_ms: 700, create_response: true, interrupt_response: true } } } } });
  }

  // ------------------------------------------------------------------ chrome
  function transcript(open) {
    const t = $('vw-transcript-toggle'); $('vw-transcript').hidden = !open; t.setAttribute('aria-expanded', String(open)); t.setAttribute('aria-label', open ? 'Hide transcript' : 'Show transcript');
    if (open) $('vw-caption').hidden = true; layout();
    try { localStorage.setItem('pw-transcript', open ? '1' : '0'); } catch (_) { /* optional */ }
  }
  function theme(value) { document.documentElement.dataset.theme = value; $('vw-theme').value = value; try { localStorage.setItem('pw-theme', value); } catch (_) { /* optional */ } }
  async function reset() { await stopVoice(); for (const c of state.cards.values()) c.node.remove(); state.cards.clear(); state.history = []; state.handled.clear();
    $('vw-messages').replaceChildren(el('p', 'vw-welcome', 'What you and the assistant say appears here. Nothing is saved: it disappears when you reload.')); $('vw-caption').hidden = true; alertBar(''); show(); }

  for (const [name, label, tool, args] of QUICK) { const b = el('button', 'vw-chip'); b.type = 'button'; b.appendChild(icon(name)); b.appendChild(document.createTextNode(label)); b.addEventListener('click', () => manual(tool, args)); $('vw-quick').appendChild(b); }
  { const b = el('button', 'primary', 'Take the tour'); b.type = 'button'; b.addEventListener('click', tour); $('vw-suggestions').appendChild(b); }
  for (const [prompt, tool, args] of STARTERS) { const b = el('button', null, prompt); b.type = 'button'; b.addEventListener('click', () => starter(prompt, tool, args)); $('vw-suggestions').appendChild(b); }

  $('vw-connect').addEventListener('click', () => connect(new URLSearchParams(location.search).has('listen')));
  $('vw-composer').addEventListener('submit', (e) => { e.preventDefault(); ask($('vw-input').value); });
  $('vw-mute').addEventListener('click', () => { state.muted = !state.muted; state.stream?.getAudioTracks().forEach((t) => { t.enabled = !state.muted; }); if (state.protocol === 'live') channelSend({ type: state.muted ? 'session.input_audio.mute' : 'session.input_audio.unmute' }); $('vw-mute').setAttribute('aria-pressed', String(state.muted)); $('vw-mute').textContent = state.muted ? 'Unmute' : 'Mute'; status(state.muted ? 'Microphone muted' : 'Listening'); });
  $('vw-interrupt').addEventListener('click', () => { interrupt(); status(state.listenOnly ? 'Type your question' : 'Listening'); });
  $('vw-play').addEventListener('click', () => { $('vw-audio').play().then(() => { $('vw-play').hidden = true; alertBar(''); }).catch(() => {}); });
  $('vw-alert-close').addEventListener('click', () => alertBar(''));
  $('vw-clear').addEventListener('click', clearCards);
  $('vw-reset').addEventListener('click', reset);
  $('vw-layout').addEventListener('change', layout);
  $('vw-theme').addEventListener('change', () => theme($('vw-theme').value));
  $('vw-sensitivity').addEventListener('change', updateAudio); $('vw-microphone').addEventListener('change', updateAudio);
  $('vw-transcript-toggle').addEventListener('click', () => transcript($('vw-transcript').hidden));
  $('vw-transcript-close').addEventListener('click', () => { transcript(false); $('vw-transcript-toggle').focus(); });
  document.addEventListener('click', (e) => { const m = $('vw-tools'); if (m.open && !m.contains(e.target)) m.open = false; });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { $('vw-tools').open = false; if ($('vw-stage').classList.contains('has-solo')) setSolo('', false); } });
  window.addEventListener('resize', layout);
  window.addEventListener('pagehide', () => { if (state.peer) navigator.sendBeacon && stopVoice(); });

  let saved = null; try { saved = localStorage.getItem('pw-theme'); } catch (_) { /* optional */ }
  theme(saved || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
  try { if (localStorage.getItem('pw-transcript') === '1' && innerWidth > 1180) transcript(true); } catch (_) { /* starts closed */ }
  startMeter(null); show();
  fetch('/api/state').then((r) => r.json()).then((config) => {
    state.config = config; state.protocol = config.protocol || 'realtime';
    if (state.protocol === 'live') { for (const id of ['vw-sensitivity', 'vw-microphone']) { $(id).hidden = true; $(id).previousElementSibling.hidden = true; } $('vw-tools').querySelector('p').hidden = true; }
    if (config.problems?.length) alertBar(config.problems.join(' '));
    else if (!config.configured) alertBar('No OpenAI key on the server: the assistant is off, but the quick cards and the tour work.');
    $('vw-privacy').textContent = (config.configured ? 'AI voice (' + config.realtime_model + '). ' : '') + 'Your questions and summary numbers go to OpenAI. Tracking data and positions stay on this computer.';
  }).catch(() => alertBar('The workspace server is not answering. Start it with: python workspace/server.py'));
  const params = new URLSearchParams(location.search);
  if (params.get('theme') === 'dark' || params.get('theme') === 'light') theme(params.get('theme'));
  if (params.has('tour')) tour();
  if (params.has('gallery')) (async () => { for (const [name, args] of GALLERY) await manual(name, args); $('vw-stage').scrollTo({ top: 0 }); })();                      // /?tour opens with the guided cards already on the canvas
  window.Workspace = { manual, ask, tour, state };   // for the console and for tests
})();
