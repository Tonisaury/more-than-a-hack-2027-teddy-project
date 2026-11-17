/* Possession Review.
   One clock (the tactical video frame), three things locked to it: the picture, the head-up display that says
   what the model thinks, and the timeline. On practice matches a fourth: your own labels.
   The challenge match is watch-only -- there is no code path here, or in the server, that stores a label for it. */
(() => {
'use strict';

const $ = (selector) => document.querySelector(selector);
const root = $('#rv'), stage = $('#stage'), video = $('#video');
const LETTER = ['H', 'A', 'D', 'X'];
const GUTTER = 128, AXIS = 20, LANE_GAP = 4;
const store = {
  get(key, fallback) { try { const v = localStorage.getItem('rv.' + key); return v === null ? fallback : JSON.parse(v); } catch (e) { return fallback; } },
  set(key, value) { try { localStorage.setItem('rv.' + key, JSON.stringify(value)); } catch (e) { /* private window */ } },
};
const params = new URLSearchParams(location.search);
const S = {
  matches: [], m: null, A: null, n: 0, fps: 25, P: 0,
  moments: [], visible: [], byTime: [], cur: null, watched: new Set(),
  frame: 0, playing: false, speed: 1, source: 'pitch', pendingSeek: false, last: 0, acc: 0,
  overlay: true, map: true, loop: false, reel: false, blind: false, follow: true, tlFollow: true,
  view: { f0: 0, f1: 1 }, lanes: [], rle: {}, redraw: true,
  L: null, Lrle: null, undo: [], redo: [], openEnd: null, sawModel: false,
  veil: params.has('veil'), drag: null, scrub: false, pendingScrub: null, hudKey: '', lastStored: 0,
};
let C = {};

// ---------------------------------------------------------------- small helpers
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const pad2 = (v) => String(v).padStart(2, '0');
function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') node.className = v; else if (k === 'text') node.textContent = v; else node.setAttribute(k, v);
  }
  for (const child of children) if (child !== null && child !== undefined) node.append(child);
  return node;
}
async function getJSON(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}
const post = (url, body) => getJSON(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
function debounce(fn, ms) { let t = 0; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); }; }
function readColours() {
  const css = getComputedStyle(root);
  const v = (name) => css.getPropertyValue(name).trim();
  C = { home: v('--rv-home'), away: v('--rv-away'), dead: v('--rv-dead'), none: v('--rv-none'), doubt: v('--rv-doubt'), lane: v('--rv-lane'),
        pitch: v('--rv-pitch'), pitchLine: v('--rv-pitch-line'), ink: v('--ink'), muted: v('--muted'), line: v('--line'), surface: v('--surface') };
}
const colourOf = (code) => (code === 0 ? C.home : code === 1 ? C.away : code === 2 ? C.dead : null);
function rle(codes) {
  const starts = [0], labels = [codes[0]];
  for (let i = 1, last = codes[0]; i < codes.length; i++) if (codes[i] !== last) { last = codes[i]; starts.push(i); labels.push(last); }
  return { starts: Int32Array.from(starts), labels: Uint8Array.from(labels) };
}
function runIndex(r, frame) {                       // the run that contains `frame`
  let lo = 0, hi = r.starts.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (r.starts[mid] <= frame) lo = mid; else hi = mid - 1; }
  return lo;
}
function who(code, short) {
  if (code === 0 || code === 1) return S.m.teams[code] + (short ? '' : ' has the ball');
  return code === 2 ? 'Dead ball' : (short ? 'Not graded' : 'Outside the graded match');
}
function halfOf(frame) { return S.m.halves.findIndex(([a, b]) => frame >= a && frame < b); }
function clockLabel(frame, short) {
  const k = halfOf(frame);
  if (k < 0) return short ? '–' : 'Outside the halves';
  const s = Math.floor((frame - S.m.halves[k][0]) / S.fps);
  return (short ? (k + 1) + 'H ' : (k === 0 ? '1st half ' : '2nd half ')) + Math.floor(s / 60) + ':' + pad2(s % 60);
}
function videoTime(frame) { const s = Math.floor(frame / S.fps); return Math.floor(s / 3600) + ':' + pad2(Math.floor(s / 60) % 60) + ':' + pad2(s % 60); }
function toast(text) { const out = $('#save-state'); if (out) { out.textContent = text; } }
function stageMessage(text) { const box = $('#stage-msg'); box.hidden = !text; box.textContent = text || ''; }

// ---------------------------------------------------------------- loading
async function init() {
  root.dataset.theme = store.get('theme', 'dark');
  $('#theme').textContent = root.dataset.theme === 'dark' ? 'Light' : 'Dark';
  readColours();
  bind();
  bindExtras();
  loadExplanations();
  try {
    S.matches = (await getJSON('api/matches')).matches;
  } catch (error) { stageMessage('Could not reach the review server: ' + error.message); return; }
  if (!S.matches.length) { stageMessage('No review data yet. Run  python review/build.py  and reload.'); return; }
  const wanted = params.get('match') || store.get('match');
  await openMatch(S.matches.some((m) => m.id === wanted) ? wanted : S.matches[0].id);
  getJSON('api/notebook').then((r) => { $('#notes').value = r.text || ''; }).catch(() => {});
  requestAnimationFrame(tick);
}

function renderMatches() {
  const nav = $('#matches');
  nav.replaceChildren(...S.matches.map((m) => el('button', { type: 'button', class: 'rv-match', 'data-id': m.id, 'aria-pressed': String(S.m && S.m.id === m.id) },
    el('b', { text: m.title }), el('small', { text: m.subtitle }),
    el('span', { class: 'rv-badge' + (m.annotatable ? ' ok' : ''), text: m.annotatable ? 'Can be labelled' : 'Watch only' }))));
}

async function openMatch(id) {
  pause();
  stageMessage('Loading…');
  let meta, buffer, moments, record = { points: [] };
  try {
    meta = await getJSON('api/match/' + id + '/meta');
    buffer = await (await fetch('api/match/' + id + '/bundle')).arrayBuffer();
    moments = await getJSON('api/match/' + id + '/moments');
    if (meta.annotatable) record = await getJSON('api/match/' + id + '/labels');
  } catch (error) { stageMessage('Could not load this match: ' + error.message); return; }
  const types = { u1: Uint8Array, i1: Int8Array, i2: Int16Array, i4: Int32Array, f4: Float32Array };
  const A = {};
  for (const a of meta.arrays) A[a.name] = new types[a.dtype](buffer, a.offset, a.bytes / types[a.dtype].BYTES_PER_ELEMENT);
  Object.assign(S, { m: meta, A, n: meta.n_frames, fps: meta.fps, P: meta.slots.length, moments, cur: null, undo: [], redo: [], openEnd: null, hudKey: '' });
  S.rle = { final: rle(A.final), prev: rle(A.prev), ball_view: rle(A.ball_view), player_view: rle(A.player_view), seen: rle(A.seen) };
  if (A.events) S.rle.events = rle(A.events);
  S.L = new Uint8Array(S.n).fill(3);
  paintPoints(record.points || []);
  S.Lrle = rle(S.L);
  S.watched = new Set(store.get('watched.' + id, []));
  S.sawModel = store.get('sawModel.' + id, false);
  store.set('match', id);

  const practice = meta.annotatable;
  S.blind = practice && store.get('blind', false);
  $('#blind').checked = S.blind;
  root.classList.toggle('blind', S.blind);
  $('#blind-wrap').hidden = !practice;
  $('#labelbar').hidden = !practice;
  $('#tab-labels').hidden = !practice;
  $('#help-labels').hidden = !practice;
  $('#mine').hidden = !practice;
  $('#l-by').value = record.labelled_by || store.get('labelledBy', '');
  $('#lab-home').textContent = $('#lg-home').textContent = meta.teams[0];
  $('#lab-away').textContent = $('#lg-away').textContent = meta.teams[1];
  $('#l-score-out').replaceChildren();
  $('#l-export-out').textContent = '';
  if (!practice && $('#tab-labels').getAttribute('aria-selected') === 'true') showTab('moments');
  showBanner();
  $('#notes-hint').textContent = practice
    ? 'Free notes. Saved on this computer, next to the match data.'
    : 'Write patterns, not verdicts: “a keeper holding the ball is read as possession” helps you design a feature. A list of frames the model got wrong would be hand-labelling this match, which the rules forbid.';
  $('#veil').hidden = !(S.veil && meta.kind === 'challenge');

  const sources = [];
  if (meta.has_video) sources.push(['tactical', 'Tactical camera']);
  if (meta.has_tv && A.bc_frame) sources.push(['tv', 'TV broadcast']);
  sources.push(['pitch', 'Pitch map']);
  $('#source').replaceChildren(...sources.map(([key, name]) => el('button', { type: 'button', 'data-source': key, 'aria-pressed': 'false', text: name })));
  $('#source').hidden = sources.length < 2;
  if (!meta.has_video && meta.kind === 'challenge') stageMessage('');
  buildFilter();
  buildLanes();
  renderMatches();
  const start = clamp(store.get('frame.' + id, meta.halves[0][0]), 0, S.n - 1);
  S.frame = start;
  setView(start - 90 * S.fps, start + 90 * S.fps);
  refreshMoments();
  refreshLabelPanel();
  $('#note-filter').value = 'all';
  loadNotes();
  setSource(sources[0][0]);
  layout();
}

function showBanner() {
  const practice = S.m.annotatable, key = 'banner.' + S.m.kind;
  $('#banner').hidden = store.get(key, false);
  $('#banner').dataset.tone = practice ? 'info' : 'warning';
  $('#banner').dataset.key = key;
  $('#banner-text').textContent = practice
    ? 'Practice match: this is another game, so labelling it is allowed. It has no video, only the tracked positions of players and ball. ' + S.m.model_note
    : 'Watch only. The competition rules forbid hand labels on this match “in any amount and for any purpose”, validation included, so nothing here can be marked right or wrong. Watching to understand and debug is allowed. To label, open a practice match.';
}

// ---------------------------------------------------------------- the clock: video when there is one, a timer otherwise
const usingVideo = () => S.source === 'tactical' || S.source === 'tv';
const videoFrameFor = (t) => (S.source === 'tv' ? S.A.bc_frame[t] : t);
function tacticalFor(vf) {
  if (S.source !== 'tv') return vf;
  const b = S.A.bc_frame, here = S.frame;
  if (b[here] >= 0) {
    const guess = vf + (here - b[here]);
    for (let d = 0; d <= 12; d++) { if (b[guess + d] === vf) return guess + d; if (b[guess - d] === vf) return guess - d; }
  }
  for (let t = 0; t < S.n; t++) if (b[t] === vf) return t;
  return -1;
}

function setSource(source) {
  const wasPlaying = S.playing;
  pause();
  S.source = source;
  stage.classList.toggle('no-video', !usingVideo());
  for (const button of $('#source').children) button.setAttribute('aria-pressed', String(button.dataset.source === source));
  $('#ov-wrap').hidden = $('#ov-info').hidden = source !== 'tactical';
  $('#map-wrap').hidden = !usingVideo();
  if (usingVideo()) {
    const url = 'video/' + source;
    if (video.dataset.src !== url) { video.dataset.src = url; video.src = url; video.load(); }
  }
  layout();
  seek(S.frame);
  if (wasPlaying) play();
}

function play() {
  if (!S.m) return;
  if (usingVideo()) {
    if (videoFrameFor(S.frame) < 0) return;
    video.playbackRate = S.speed;
    video.play().catch(() => {});
  } else {
    S.playing = true; S.acc = 0; S.last = performance.now();
    updatePlayButton();
  }
}
function pause() {
  if (usingVideo() && !video.paused) video.pause();
  S.playing = false;
  updatePlayButton();
}
function updatePlayButton() { const b = $('#play'); b.textContent = S.playing ? 'Pause' : 'Play'; b.setAttribute('aria-label', S.playing ? 'Pause' : 'Play'); }

function seek(target) {
  if (!S.m) return;
  S.frame = clamp(Math.round(target), 0, S.n - 1);
  if (usingVideo()) {
    const vf = videoFrameFor(S.frame);
    if (vf < 0) {
      stageMessage('The TV recording has no picture for this moment. Switch to the tactical camera or the pitch map.');
      if (!video.paused) video.pause();
    } else {
      stageMessage('');
      S.pendingSeek = true;
      video.currentTime = (vf + 0.5) / S.fps;
    }
  } else stageMessage('');
  renderFrame(true);
}

function step(delta) {
  pause();
  const from = S.frame, to = clamp(from + delta, 0, S.n - 1);
  if (delta > 0 && delta <= 2 * S.fps) followLabels(from, to);
  seek(to);
}

/* Every new frame arrives here, from the video or from the timer. */
function advanceTo(t) {
  const from = S.frame;
  if (t >= S.n - 1) { t = S.n - 1; pause(); }
  if (S.playing) {
    if (S.cur && from < S.cur.end && t >= S.cur.end) markWatched(S.cur, true);
    if (S.loop && S.cur && from < S.cur.play_end && t >= S.cur.play_end) { seek(S.cur.play_start); return; }
    if (S.reel) {
      const inside = S.byTime.find((m) => t >= m.play_start && t < m.play_end);
      if (!inside) {
        const next = S.byTime.find((m) => m.play_start > t);
        if (next) { setCurrent(next, true); seek(next.play_start); } else pause();
        return;
      }
      if (inside !== S.cur) setCurrent(inside, true);
    }
    if (t > from && t - from <= 2 * S.fps) followLabels(from, t);
  }
  S.frame = t;
  renderFrame(false);
}

function onVideoFrame(now, info) {
  video.requestVideoFrameCallback(onVideoFrame);
  S.lastVideoFrame = performance.now();
  if (!S.m || !usingVideo() || S.pendingSeek) return;
  const t = tacticalFor(Math.round(info.mediaTime * S.fps));
  if (t >= 0 && t !== S.frame) advanceTo(t);
}
const hasFrameCallback = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

function tick(now) {
  if (S.m) {
    if (S.playing && !usingVideo()) {
      S.acc += (now - S.last) / 1000 * S.fps * S.speed;
      const whole = Math.floor(S.acc);
      if (whole > 0) { S.acc -= whole; advanceTo(S.frame + Math.min(whole, 2 * S.fps)); }
    } else if (S.playing && usingVideo() && !S.pendingSeek && now - (S.lastVideoFrame || 0) > 250) {
      // the browser is not reporting presented frames (older browser, or the picture is covered): fall back to the media clock
      const t = tacticalFor(Math.floor(video.currentTime * S.fps + 0.001));
      if (t >= 0 && t !== S.frame) advanceTo(t);
    }
    S.last = now;
    if (S.pendingScrub !== null) { const f = S.pendingScrub; S.pendingScrub = null; seek(f); }
    if (S.redraw) { S.redraw = false; drawLanes(); drawOverviewCache(); paintTimeline(); }
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- one frame on screen
function momentAt(frame) {
  const list = S.moments;
  let lo = 0, hi = list.length - 1;
  while (lo <= hi) { const mid = (lo + hi) >> 1; if (list[mid].end <= frame) lo = mid + 1; else if (list[mid].start > frame) hi = mid - 1; else return list[mid]; }
  return null;
}

function renderFrame(fromSeek) {
  const f = S.frame, A = S.A, m = S.m;
  const label = A.final[f], here = momentAt(f);
  const key = [label, A.ball_view[f], A.player_view[f], A.prev[f], A.events ? A.events[f] : 9, A.seen[f], here ? here.id : 0, S.L[f], S.blind].join();
  if (key !== S.hudKey) {
    S.hudKey = key;
    $('#call').dataset.label = label;
    $('#call-who').textContent = 'Model: ' + who(label);
    const chips = [['Ball view', A.ball_view[f], 'ball_view'], ['Player view', A.player_view[f], 'player_view']];
    if (m.has_previous) chips.push(['Previous version', A.prev[f], 'prev']);
    if (A.events) chips.push(['Event data', A.events[f], 'events']);
    const views = chips.map(([name, code, topic]) => el('button', { type: 'button', class: 'rv-view' + (label < 3 && code !== label && code < 3 ? ' differs' : ''),
      'data-label': code, 'data-explain': topic, title: 'What is this?' }, el('i'), name + ': ' + who(code, true)));
    views.push(el('button', { type: 'button', class: 'rv-view', 'data-label': A.seen[f] ? 2 : 3, 'data-explain': 'seen', title: 'What is this?' }, el('i'),
      A.seen[f] === 0 ? 'Ball not in the data' : A.seen[f] === 2 && m.kind === 'challenge' ? 'Ball filled in (TV or bridged)' : 'Ball in the data'));
    if (A.ball_model) views.push(el('span', { class: 'rv-view ring', title: 'The model was given a deliberately degraded ball, to imitate our tracking' }, el('i'), 'Dashed ring: the ball as the model saw it'));
    $('#views').replaceChildren(...views);
    $('#doubt-flag').hidden = !here;
    if (here) { $('#doubt-title').textContent = 'Doubtful moment · no. ' + here.rank + ' of ' + S.moments.length; $('#doubt-why').textContent = here.reasons.join(' · '); }
    if (m.annotatable) {
      $('#mine').dataset.label = S.L[f];
      $('#mine-who').textContent = S.L[f] === 3 ? 'not labelled' : who(S.L[f], true);
      for (const b of document.querySelectorAll('.rv-btn.lab')) b.setAttribute('aria-pressed', String(Number(b.dataset.lab) === S.L[f]));
    }
  }
  if (label < 3) {
    const p = [A.post[f * 3], A.post[f * 3 + 1], A.post[f * 3 + 2]];
    $('#call-sure').textContent = Math.round(p[label] / 2.55) + '% sure';
    $('#bar-h').style.height = (p[0] / 2.55) + '%'; $('#bar-a').style.height = (p[1] / 2.55) + '%'; $('#bar-d').style.height = (p[2] / 2.55) + '%';
  } else $('#call-sure').textContent = '';
  stage.dataset.doubt = S.blind ? 0 : A.doubt[f] > 128 ? 2 : (here || A.doubt[f] > 64) ? 1 : 0;
  $('#clock').textContent = clockLabel(f);
  $('#frame-no').textContent = 'frame ' + f.toLocaleString('en-US') + (m.has_video ? ' · video ' + videoTime(f) : '');

  const hidden = S.veil && m.kind === 'challenge';
  if (!hidden) {
    if (usingVideo()) { drawOverlay(); if (S.map) drawPitch($('#minimap'), false); }
    else drawPitch($('#pitch'), true);
  }
  $('#minimap').style.visibility = S.map && usingVideo() && !hidden ? 'visible' : 'hidden';

  if (S.tlFollow && !S.drag && !S.scrub) {
    const span = S.view.f1 - S.view.f0;
    if (f < S.view.f0 || f >= S.view.f1) setView(fromSeek ? f - span / 2 : f - span * 0.1, fromSeek ? f + span / 2 : f + span * 0.9);
  }
  paintTimeline();
  const now = performance.now();
  if (now - S.lastStored > 2000) { S.lastStored = now; store.set('frame.' + m.id, f); }
}

function sizeCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1, w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return dpr;
}

function drawPitch(canvas, big) {
  const dpr = sizeCanvas(canvas), ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  if (!w || !h) return;
  const A = S.A, f = S.frame, [LEN, WID] = S.m.pitch, P = S.P;
  const margin = big ? h * 0.13 : 5 * dpr, scale = Math.min((w - 2 * margin) / LEN, (h - 2 * margin) / WID);   // big: room for the HUD above and below
  const ox = (w - LEN * scale) / 2, oy = (h - WID * scale) / 2;
  const X = (cm) => ox + cm / 100 * scale, Y = (cm) => oy + cm / 100 * scale;
  ctx.clearRect(0, 0, w, h);
  if (big) { ctx.fillStyle = C.pitch; ctx.fillRect(0, 0, w, h); }
  ctx.strokeStyle = big ? C.pitchLine : '#ffffff66';
  ctx.lineWidth = Math.max(1, (big ? 1.5 : 1) * dpr);
  ctx.strokeRect(ox, oy, LEN * scale, WID * scale);
  ctx.beginPath(); ctx.moveTo(ox + LEN * scale / 2, oy); ctx.lineTo(ox + LEN * scale / 2, oy + WID * scale); ctx.stroke();
  ctx.beginPath(); ctx.arc(ox + LEN * scale / 2, oy + WID * scale / 2, 9.15 * scale, 0, Math.PI * 2); ctx.stroke();
  for (const side of [0, 1]) {
    for (const [depth, width] of [[16.5, 40.32], [5.5, 18.32]]) {
      ctx.strokeRect(side ? ox + (LEN - depth) * scale : ox, oy + (WID - width) / 2 * scale, depth * scale, width * scale);
    }
  }
  const r = big ? Math.max(5 * dpr, 0.95 * scale) : 2.6 * dpr;
  for (let p = 0; p < P; p++) {
    const x = A.players[(f * P + p) * 2], y = A.players[(f * P + p) * 2 + 1];
    if (x === -32768) continue;
    const team = S.m.slots[p];
    ctx.beginPath(); ctx.arc(X(x), Y(y), r, 0, Math.PI * 2);
    ctx.fillStyle = team === 0 ? C.home : team === 1 ? C.away : '#9aa7b5'; ctx.fill();
    if (big) { ctx.strokeStyle = '#0008'; ctx.lineWidth = dpr; ctx.stroke(); }
  }
  if (big && A.ball_model && !S.blind) {
    const gx = A.ball_model[f * 2], gy = A.ball_model[f * 2 + 1];
    if (gx !== -32768) { ctx.beginPath(); ctx.arc(X(gx), Y(gy), r * 1.25, 0, Math.PI * 2); ctx.strokeStyle = C.doubt; ctx.lineWidth = 1.5 * dpr; ctx.setLineDash([4 * dpr, 3 * dpr]); ctx.stroke(); ctx.setLineDash([]); }
  }
  const bx = A.ball[f * 2], by = A.ball[f * 2 + 1];
  if (bx !== -32768) {
    if (big) {
      ctx.beginPath();
      let started = false;
      for (let k = Math.max(0, f - S.fps); k <= f; k++) {
        const tx = A.ball[k * 2], ty = A.ball[k * 2 + 1];
        if (tx === -32768) { started = false; continue; }
        if (started) ctx.lineTo(X(tx), Y(ty)); else { ctx.moveTo(X(tx), Y(ty)); started = true; }
      }
      ctx.strokeStyle = '#ffffff88'; ctx.lineWidth = 1.5 * dpr; ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(X(bx), Y(by), big ? r * 0.62 : 2.2 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = '#fff'; ctx.fill(); ctx.strokeStyle = '#000'; ctx.lineWidth = dpr; ctx.stroke();
  }
}

function drawOverlay() {
  const canvas = $('#overlay'), dpr = sizeCanvas(canvas), ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const A = S.A, f = S.frame;
  if (!S.overlay || S.source !== 'tactical' || !A.homography || !A.homography_ok[f]) return;
  const hm = A.homography, o = f * 9, [LEN, WID] = S.m.pitch, sx = canvas.width / S.m.image[0], sy = canvas.height / S.m.image[1];
  const project = (xcm, ycm) => {
    const x = xcm / 100 / LEN, y = ycm / 100 / WID, d = hm[o + 6] * x + hm[o + 7] * y + hm[o + 8];
    return [(hm[o] * x + hm[o + 1] * y + hm[o + 2]) / d * sx, (hm[o + 3] * x + hm[o + 4] * y + hm[o + 5]) / d * sy];
  };
  const bx = A.ball[f * 2], by = A.ball[f * 2 + 1], hasBall = bx !== -32768;
  let nearest = -1, best = 250 * 250;                                  // 2.5 m, the model's control radius
  ctx.lineWidth = 2 * dpr;
  for (let p = 0; p < S.P; p++) {
    const x = A.players[(f * S.P + p) * 2], y = A.players[(f * S.P + p) * 2 + 1], team = S.m.slots[p];
    if (x === -32768 || (team !== 0 && team !== 1)) continue;
    const [u, v] = project(x, y);
    ctx.beginPath(); ctx.ellipse(u, v, 11 * dpr, 4.5 * dpr, 0, 0, Math.PI * 2);
    ctx.strokeStyle = team === 0 ? '#3f93ea' : '#f0703a'; ctx.stroke();
    if (hasBall) { const d = (x - bx) * (x - bx) + (y - by) * (y - by); if (d < best) { best = d; nearest = p; } }
  }
  if (hasBall) {
    const [u, v] = project(bx, by);
    ctx.beginPath(); ctx.arc(u, v, 9 * dpr, 0, Math.PI * 2); ctx.strokeStyle = '#ffe14d'; ctx.lineWidth = 2.5 * dpr; ctx.stroke();
    if (nearest >= 0) {
      const [pu, pv] = project(A.players[(f * S.P + nearest) * 2], A.players[(f * S.P + nearest) * 2 + 1]);
      ctx.beginPath(); ctx.ellipse(pu, pv, 17 * dpr, 7 * dpr, 0, 0, Math.PI * 2);
      ctx.strokeStyle = S.m.slots[nearest] === 0 ? '#3f93ea' : '#f0703a'; ctx.lineWidth = 3.5 * dpr; ctx.stroke();
    }
  }
}

// ---------------------------------------------------------------- timeline
function buildLanes() {
  const m = S.m, lanes = [];
  if (m.annotatable) lanes.push({ key: 'mine', name: 'Your labels', h: 26, kind: 'mine' });
  if (!S.blind) {
    lanes.push({ key: 'final', name: 'Model', h: 22, kind: 'seg' });
    if (m.has_previous) lanes.push({ key: 'prev', name: 'Previous version', h: 11, kind: 'seg' });
    lanes.push({ key: 'ball_view', name: 'Ball view alone', h: 11, kind: 'seg' });
    lanes.push({ key: 'player_view', name: 'Player view alone', h: 11, kind: 'seg' });
    if (S.A.events) lanes.push({ key: 'events', name: 'Event data', h: 11, kind: 'seg' });
    lanes.push({ key: 'doubt', name: 'Doubt', h: 16, kind: 'heat' });
  }
  if (!(S.veil && m.kind === 'challenge')) {
    lanes.push({ key: 'seen', name: 'Ball in the data', h: 7, kind: 'seen' });
    if (m.tags && m.tags.length) lanes.push({ key: 'tags', name: 'Tagged restarts', h: 9, kind: 'tags' });
  }
  let y = AXIS;
  for (const lane of lanes) { lane.y = y; y += lane.h + LANE_GAP; }
  S.lanes = lanes;
  $('#timeline').style.height = y + 'px';
  S.redraw = true;
}

function setView(f0, f1) {
  let span = clamp(f1 - f0, 5 * S.fps, S.n);
  f0 = clamp(f0, 0, S.n - span);
  S.view = { f0, f1: f0 + span };
  S.redraw = true;
}
function zoom(factor, anchor) {
  const { f0, f1 } = S.view, a = anchor === undefined ? S.frame : anchor, t = clamp((a - f0) / (f1 - f0), 0, 1);
  const span = clamp((f1 - f0) * factor, 5 * S.fps, S.n);
  setView(a - t * span, a + (1 - t) * span);
}
const TL = { cache: document.createElement('canvas'), over: document.createElement('canvas') };
const tlWidth = () => $('#timeline').clientWidth;
const xOf = (frame) => GUTTER + (frame - S.view.f0) / (S.view.f1 - S.view.f0) * (tlWidth() - GUTTER);
const frameAt = (x) => Math.round(S.view.f0 + (x - GUTTER) / (tlWidth() - GUTTER) * (S.view.f1 - S.view.f0));

function drawRuns(ctx, r, y, h, w) {
  const { f0, f1 } = S.view;
  for (let i = runIndex(r, Math.max(0, Math.floor(f0))); i < r.starts.length && r.starts[i] < f1; i++) {
    const colour = colourOf(r.labels[i]);
    if (!colour) continue;
    const a = Math.max(GUTTER, xOf(r.starts[i])), b = Math.min(w, xOf(i + 1 < r.starts.length ? r.starts[i + 1] : S.n));
    ctx.fillStyle = colour;
    ctx.fillRect(a, y, Math.max(1, b - a), h);
  }
}

function drawLanes() {
  const canvas = $('#timeline'), dpr = sizeCanvas(canvas), cache = TL.cache;
  cache.width = canvas.width; cache.height = canvas.height;
  const ctx = cache.getContext('2d'), w = canvas.clientWidth, h = canvas.clientHeight, { f0, f1 } = S.view, A = S.A;
  if (!w) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = C.surface; ctx.fillRect(0, 0, w, h);
  ctx.font = '11px Poppins, Arial'; ctx.textBaseline = 'middle';
  // time axis
  const perPx = (f1 - f0) / S.fps / (w - GUTTER);
  const stepS = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1800].find((s) => s / perPx >= 78) || 3600;
  ctx.fillStyle = C.muted; ctx.textAlign = 'left';
  for (const [a, b] of S.m.halves) {
    const first = Math.max(a, a + Math.ceil((f0 - a) / (stepS * S.fps)) * stepS * S.fps);
    for (let f = first; f < Math.min(b, f1); f += stepS * S.fps) {
      const x = xOf(f);
      if (x < GUTTER) continue;
      ctx.fillStyle = C.line; ctx.fillRect(x, AXIS - 5, 1, h);
      ctx.fillStyle = C.muted; ctx.fillText(clockLabel(f, true), x + 4, AXIS / 2);
    }
  }
  for (const lane of S.lanes) {
    ctx.fillStyle = C.lane; ctx.fillRect(GUTTER, lane.y, w - GUTTER, lane.h);
    ctx.fillStyle = lane.kind === 'mine' ? C.ink : C.muted; ctx.textAlign = 'right';
    ctx.fillText(lane.name, GUTTER - 8, lane.y + lane.h / 2);
    if (EX.articles.has(lane.key)) {                        // the name is a button: a dotted underline says so
      const tw = ctx.measureText(lane.name).width, ty = Math.min(lane.y + lane.h / 2 + 7, lane.y + lane.h + 2);
      ctx.strokeStyle = C.muted; ctx.lineWidth = 1; ctx.setLineDash([1, 2]);
      ctx.beginPath(); ctx.moveTo(GUTTER - 8 - tw, ty); ctx.lineTo(GUTTER - 8, ty); ctx.stroke(); ctx.setLineDash([]);
    }
    if (lane.kind === 'seg') drawRuns(ctx, S.rle[lane.key], lane.y, lane.h, w);
    else if (lane.kind === 'mine') {
      drawRuns(ctx, S.Lrle, lane.y, lane.h, w);
      ctx.fillStyle = C.ink;
      for (let i = 1; i < S.Lrle.starts.length; i++) {
        const x = xOf(S.Lrle.starts[i]);
        if (x >= GUTTER && x <= w) ctx.fillRect(x - 1, lane.y - 2, 2, lane.h + 4);
      }
      if (S.Lrle.starts.length === 1) { ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.fillText('Press 1, 2 or 3 to start labelling at the playhead', GUTTER + 8, lane.y + lane.h / 2); }
    } else if (lane.kind === 'seen') {
      const r = S.rle.seen;
      for (let i = runIndex(r, Math.max(0, Math.floor(f0))); i < r.starts.length && r.starts[i] < f1; i++) {
        if (!r.labels[i]) continue;
        const a = Math.max(GUTTER, xOf(r.starts[i])), b = Math.min(w, xOf(i + 1 < r.starts.length ? r.starts[i + 1] : S.n));
        ctx.fillStyle = r.labels[i] === 2 ? C.muted : C.ink; ctx.globalAlpha = r.labels[i] === 2 ? 0.55 : 0.8;
        ctx.fillRect(a, lane.y, Math.max(1, b - a), lane.h);
      }
      ctx.globalAlpha = 1;
    } else if (lane.kind === 'heat') {
      const framesPerPx = (f1 - f0) / (w - GUTTER), stride = Math.max(1, Math.floor(framesPerPx / 6));
      ctx.fillStyle = C.doubt;
      for (let x = GUTTER; x < w; x++) {
        const a = Math.max(0, Math.floor(f0 + (x - GUTTER) * framesPerPx)), b = Math.min(S.n, Math.max(a + 1, Math.floor(f0 + (x + 1 - GUTTER) * framesPerPx)));
        let peak = 0;
        for (let k = a; k < b; k += stride) if (A.doubt[k] > peak) peak = A.doubt[k];
        if (peak > 12) { ctx.globalAlpha = Math.min(1, peak / 150); ctx.fillRect(x, lane.y, 1, lane.h); }
      }
      ctx.globalAlpha = 1;
    } else if (lane.kind === 'tags') {
      for (const tag of S.m.tags) {
        if (tag.frame < f0 || tag.frame > f1) continue;
        ctx.fillStyle = tag.team === 0 ? C.home : tag.team === 1 ? C.away : C.muted;
        ctx.fillRect(xOf(tag.frame) - 1, lane.y, 3, lane.h);
      }
    }
  }
  // doubtful moments: a bracket over the lanes; the selected one is filled
  if (!S.blind) {
    const bottom = S.lanes.length ? S.lanes[S.lanes.length - 1].y + S.lanes[S.lanes.length - 1].h : h;
    for (const m of S.moments) {
      if (m.end < f0 || m.start > f1) continue;
      const a = Math.max(GUTTER, xOf(m.start)), b = Math.min(w, xOf(m.end));
      ctx.fillStyle = C.doubt;
      ctx.fillRect(a, AXIS - 4, Math.max(2, b - a), 3);
      if (S.cur === m) { ctx.globalAlpha = 0.16; ctx.fillRect(a, AXIS - 1, Math.max(2, b - a), bottom - AXIS + 1); ctx.globalAlpha = 1; }
    }
  }
  // your notes: a flag on the time axis
  ctx.fillStyle = C.ink;
  for (const note of N.list) {
    if (note.frame < f0 || note.frame > f1) continue;
    const x = xOf(note.frame);
    ctx.beginPath(); ctx.moveTo(x, 2); ctx.lineTo(x + 7, 5.5); ctx.lineTo(x, 9); ctx.closePath(); ctx.fill();
    ctx.fillRect(x - 0.5, 2, 1, AXIS - 6);
  }
  // outside the graded halves
  ctx.fillStyle = C.surface; ctx.globalAlpha = 0.72;
  let edge = 0;
  for (const [a, b] of S.m.halves.concat([[S.n, S.n]])) {
    const xa = Math.max(GUTTER, xOf(edge)), xb = Math.min(w, xOf(a));
    if (xb > xa) ctx.fillRect(xa, AXIS, xb - xa, h - AXIS);
    edge = b;
  }
  ctx.globalAlpha = 1;
}

function drawOverviewCache() {
  const canvas = $('#overview'), dpr = sizeCanvas(canvas), cache = TL.over;
  cache.width = canvas.width; cache.height = canvas.height;
  const ctx = cache.getContext('2d'), w = canvas.clientWidth, h = canvas.clientHeight, A = S.A;
  if (!w) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = C.lane; ctx.fillRect(0, 0, w, h);
  const per = S.n / w, lane = S.m.annotatable && S.blind ? S.L : A.final;
  for (let x = 0; x < w; x++) {
    const a = Math.floor(x * per), b = Math.max(a + 1, Math.floor((x + 1) * per));
    const colour = colourOf(lane[(a + b) >> 1]);
    if (colour) { ctx.fillStyle = colour; ctx.fillRect(x, 8, 1, h - 8); }
    if (!S.blind) {
      let peak = 0;
      for (let k = a; k < b; k += 5) if (A.doubt[k] > peak) peak = A.doubt[k];
      if (peak > 90) { ctx.fillStyle = C.doubt; ctx.fillRect(x, 0, 1, 6); }
    }
  }
}

function paintTimeline() {
  const canvas = $('#timeline'), ctx = canvas.getContext('2d'), dpr = window.devicePixelRatio || 1;
  if (TL.cache.width !== canvas.width || !canvas.width) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(TL.cache, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const x = xOf(S.frame);
  if (x >= GUTTER && x <= canvas.clientWidth) {
    ctx.fillStyle = C.ink; ctx.fillRect(x - 1, 0, 2, canvas.clientHeight);
    ctx.beginPath(); ctx.moveTo(x - 6, 0); ctx.lineTo(x + 6, 0); ctx.lineTo(x, 8); ctx.closePath(); ctx.fill();
  }
  const over = $('#overview'), octx = over.getContext('2d');
  if (TL.over.width !== over.width || !over.width) return;
  octx.setTransform(1, 0, 0, 1, 0, 0);
  octx.drawImage(TL.over, 0, 0);
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = over.clientWidth, a = S.view.f0 / S.n * w, b = S.view.f1 / S.n * w;
  octx.strokeStyle = C.ink; octx.lineWidth = 1.5; octx.strokeRect(a + 0.75, 0.75, Math.max(3, b - a) - 1.5, over.clientHeight - 1.5);
  octx.fillStyle = C.ink; octx.fillRect(S.frame / S.n * w - 0.5, 0, 1.5, over.clientHeight);
}

function layout() { if (!S.m) return; S.redraw = true; S.hudKey = ''; renderFrame(false); }

// ---------------------------------------------------------------- doubtful moments
function buildFilter() {
  const counts = {};
  for (const m of S.moments) for (const r of m.reasons) counts[r] = (counts[r] || 0) + 1;
  const options = [['all', 'All moments (' + S.moments.length + ')'], ['unwatched', 'Not watched yet']];
  for (const [reason, n] of Object.entries(counts).sort((a, b) => b[1] - a[1])) options.push([reason, reason[0].toUpperCase() + reason.slice(1) + ' (' + n + ')']);
  $('#m-filter').replaceChildren(...options.map(([value, text]) => el('option', { value, text })));
}

function momentState(m) {
  if (!S.m.annotatable) return S.watched.has(m.id) ? ['seen', 'Watched'] : null;
  let labelled = 0, agree = 0;
  for (let k = m.start; k < m.end; k++) if (S.L[k] < 3) { labelled++; if (S.L[k] === S.A.final[k]) agree++; }
  if (labelled < 0.9 * (m.end - m.start)) return S.watched.has(m.id) ? ['seen', 'Watched'] : null;
  if (S.blind) return ['seen', 'Labelled'];
  return agree >= 0.9 * labelled ? ['ok', 'Model agrees'] : ['no', 'Model differs'];
}

function refreshMoments() {
  const filter = $('#m-filter').value || 'all', sort = $('#m-sort').value;
  const list = S.moments.filter((m) => filter === 'all' || (filter === 'unwatched' ? !S.watched.has(m.id) : m.reasons.includes(filter)));
  S.byTime = list;
  S.visible = sort === 'rank' ? list.slice().sort((a, b) => a.rank - b.rank) : list;
  const items = S.visible.map((m) => {
    const state = momentState(m);
    const says = el('span', { class: 'rv-says', title: 'What the model says across this moment' });
    if (!S.blind) for (const c of m.says) says.append(el('i', { class: 'c' + LETTER.indexOf(c) }));
    const mine = notesOf(m).length;
    return el('li', { class: 'rv-moment-row' }, el('button', { type: 'button', class: 'rv-note-btn' + (mine ? ' has' : ''), 'data-note-moment': m.id,
      title: 'Take a note about this moment', text: mine ? mine + (mine > 1 ? ' notes' : ' note') : 'Note' }),
    el('button', { type: 'button', class: 'rv-moment' + (S.watched.has(m.id) ? ' watched' : ''), 'data-id': m.id, 'aria-current': String(S.cur === m) },
      el('span', { class: 'rv-rank', text: m.rank, title: 'Rank by amount of doubt' }),
      el('span', { class: 'rv-moment-when' }, clockLabel(m.start) + ' ', el('small', { text: '· ' + ((m.end - m.start) / S.fps).toFixed(1) + ' s ' }), says),
      el('span', { class: 'rv-moment-why', text: m.reasons.join(' · ') }),
      state ? el('span', { class: 'rv-moment-state ' + state[0], text: state[1] }) : null));
  });
  $('#moments').replaceChildren(...(items.length ? items : [el('li', { class: 'rv-empty', text: 'Nothing matches this filter.' })]));
  const done = S.moments.filter((m) => S.watched.has(m.id)).length;
  $('#m-progress').textContent = S.moments.length
    ? done + ' of ' + S.moments.length + ' watched · ' + (S.moments.reduce((sum, m) => sum + m.end - m.start, 0) / S.fps / 60).toFixed(1) + ' minutes of doubt in total'
    : 'No doubtful moments found.';
}
const refreshMomentsSoon = debounce(refreshMoments, 400);

function setCurrent(m, scroll) {
  S.cur = m;
  for (const button of document.querySelectorAll('.rv-moment')) {
    const on = m && Number(button.dataset.id) === m.id;
    button.setAttribute('aria-current', String(Boolean(on)));
    if (on && scroll) button.scrollIntoView({ block: 'nearest' });
  }
  if ($('#note-filter').value === 'moment') renderNotes();
  S.redraw = true;
}
function openMoment(m) {
  if (!m) return;
  setCurrent(m, true);
  const span = Math.max(S.view.f1 - S.view.f0, 0);
  if (span > 4 * 60 * S.fps || m.play_start < S.view.f0 || m.play_end > S.view.f1) {
    const mid = (m.play_start + m.play_end) / 2, half = Math.max(20 * S.fps, (m.play_end - m.play_start) * 1.6);
    setView(mid - half, mid + half);
  }
  seek(m.play_start);
  play();
}
function neighbour(direction) {
  if (!S.visible.length) return;
  const i = S.cur ? S.visible.indexOf(S.cur) : -1;
  openMoment(S.visible[clamp(i + direction, 0, S.visible.length - 1)] || S.visible[0]);
}
function markWatched(m, on) {
  const had = S.watched.has(m.id);
  if (on === undefined) on = !had;
  if (on === had) return;
  if (on) S.watched.add(m.id); else S.watched.delete(m.id);
  store.set('watched.' + S.m.id, [...S.watched]);
  refreshMomentsSoon();
}

// ---------------------------------------------------------------- your labels (practice matches only)
function paintPoints(points) {
  S.L.fill(3);
  points.forEach(([frame, label], i) => { S.L.fill(LETTER.indexOf(label), frame, i + 1 < points.length ? points[i + 1][0] : S.n); });
}
function pointsFromLabels() {
  const r = S.Lrle, points = [];
  for (let i = 0; i < r.starts.length; i++) if (i > 0 || r.labels[i] !== 3) points.push([r.starts[i], LETTER[r.labels[i]]]);
  return points;
}
function snapshot() { S.undo.push(pointsFromLabels()); if (S.undo.length > 300) S.undo.shift(); S.redo = []; }
function labelsChanged(light) {
  S.Lrle = rle(S.L);
  if (!S.blind) { S.sawModel = true; store.set('sawModel.' + S.m.id, true); }
  S.redraw = true; S.hudKey = '';
  saveSoon(); statsSoon(); refreshMomentsSoon();
  if (!light) renderFrame(false);
}
function runEnd(frame) { const code = S.L[frame]; let e = frame; while (e < S.n && S.L[e] === code) e++; return e; }

function applyLabel(code) {
  if (!S.m || !S.m.annotatable) return;
  const f = S.frame, current = S.L[f];
  if (current === code) return;
  snapshot();
  if (current === 3) S.L[f] = code;                       // start a new stretch; it grows as you watch
  else S.L.fill(code, f, runEnd(f));                       // from here to the next change
  const end = runEnd(f);
  S.openEnd = code !== 3 && end < S.n && S.L[end] === 3 ? end : null;
  labelsChanged();
}
function followLabels(from, to) {
  if (!S.m.annotatable || !S.follow || S.openEnd === null || from + 1 !== S.openEnd || S.L[from] === 3) return;
  let k = from + 1;
  while (k <= to && k < S.n && S.L[k] === 3) S.L[k++] = S.L[from];
  S.openEnd = k < S.n && S.L[k] === 3 ? k : null;
  labelsChanged(true);
}
function fillFromModel() {
  if (!S.m.annotatable || S.blind) { toast(S.blind ? 'Turn off “Hide model” first' : ''); return; }
  const f = S.frame;
  if (S.L[f] !== 3) { toast('Nothing to fill here'); return; }
  let a = f;
  while (a > 0 && f - a < 30 * S.fps && S.L[a - 1] === 3) a--;
  snapshot();
  for (let k = a; k <= f; k++) if (S.A.final[k] < 3) S.L[k] = S.A.final[k];
  S.openEnd = f + 1 < S.n && S.L[f + 1] === 3 && S.L[f] !== 3 ? f + 1 : null;
  labelsChanged();
}
function removeBoundary() {
  if (!S.m.annotatable) return;
  const r = S.Lrle, i = runIndex(r, Math.min(S.n - 1, S.frame + 12));
  if (i === 0) return;
  const b = r.starts[i];
  if (S.frame - b > 5 * S.fps) { toast('No change within 5 s before the playhead'); return; }
  snapshot();
  S.L.fill(S.L[b - 1], b, i + 1 < r.starts.length ? r.starts[i + 1] : S.n);
  S.openEnd = null;
  labelsChanged();
}
function undo(redo) {
  const from = redo ? S.redo : S.undo, to = redo ? S.undo : S.redo;
  if (!from.length) return;
  to.push(pointsFromLabels());
  paintPoints(from.pop());
  S.openEnd = null;
  labelsChanged();
}
function boundaryNear(frame, tolerance) {
  const r = S.Lrle;
  let best = null;
  for (const i of [runIndex(r, clamp(frame, 0, S.n - 1)), runIndex(r, clamp(frame, 0, S.n - 1)) + 1]) {
    if (i > 0 && i < r.starts.length && Math.abs(r.starts[i] - frame) <= tolerance && (best === null || Math.abs(r.starts[i] - frame) < Math.abs(best - frame))) best = r.starts[i];
  }
  return best;
}
function startDrag(boundary) {
  snapshot();
  const r = S.Lrle, i = runIndex(r, boundary);
  S.drag = { b0: boundary, left: S.L[boundary - 1], right: S.L[boundary], lo: r.starts[i - 1] + 1, hi: (i + 1 < r.starts.length ? r.starts[i + 1] : S.n) - 1, base: S.L.slice() };
  pause();
}
function dragTo(frame) {
  const d = S.drag, b = clamp(frame, d.lo, d.hi);
  S.L.set(d.base);
  if (b > d.b0) S.L.fill(d.left, d.b0, b); else S.L.fill(d.right, b, d.b0);
  S.Lrle = rle(S.L); S.redraw = true; S.hudKey = '';
  S.pendingScrub = b;
}

const saveSoon = debounce(async () => {
  if (!S.m || !S.m.annotatable) return;
  const id = S.m.id;
  toast('Saving…');
  try {
    await post('api/match/' + id + '/labels', { points: pointsFromLabels(), blind: !S.sawModel, labelled_by: $('#l-by').value });
    const now = new Date();
    toast('Saved ' + pad2(now.getHours()) + ':' + pad2(now.getMinutes()));
  } catch (error) { toast('Not saved: ' + error.message); }
}, 700);

function refreshLabelPanel() {
  if (!S.m || !S.m.annotatable) return;
  const L = S.L, r = S.Lrle, A = S.A;
  let labelled = 0, possessions = 0, stoppages = 0, live = 0;
  for (let i = 0; i < r.starts.length; i++) {
    const length = (i + 1 < r.starts.length ? r.starts[i + 1] : S.n) - r.starts[i];
    if (r.labels[i] === 3) continue;
    labelled += length;
    if (r.labels[i] === 2) stoppages++; else { possessions++; live += length; }
  }
  const covered = S.moments.filter((m) => { let c = 0; for (let k = m.start; k < m.end; k++) if (L[k] < 3) c++; return c >= 0.9 * (m.end - m.start); }).length;
  const minutes = (frames) => (frames / S.fps / 60).toFixed(1) + ' min';
  const rows = [['Labelled', minutes(labelled)], ['Possessions', possessions], ['Mean possession', possessions ? (live / possessions / S.fps).toFixed(1) + ' s' : '–'],
                ['Stoppages', stoppages], ['Doubtful moments covered', covered + ' of ' + S.moments.length]];
  $('#l-stats').replaceChildren(...rows.flatMap(([k, v]) => [el('dt', { text: k }), el('dd', { text: v })]));
  $('#l-score').disabled = S.blind;
  const table = $('#l-agree');
  if (S.blind) { table.replaceChildren(el('caption', { class: 'rv-hint', text: 'Hidden while “Hide model” is on.' })); return; }
  const sources = [['final', 'Model'], ['prev', 'Previous version'], ['ball_view', 'Ball view alone'], ['player_view', 'Player view alone'], ['events', 'Event data']].filter(([k]) => A[k]);
  const body = sources.map(([key, name]) => {
    let agree = 0;
    for (let k = 0; k < S.n; k++) if (L[k] < 3 && A[key][k] === L[k]) agree++;
    return el('tr', null, el('td', { text: name }), el('td', { class: 'n', text: labelled ? (100 * agree / labelled).toFixed(1) + '%' : '–' }));
  });
  table.replaceChildren(el('tr', null, el('th', { text: 'Source' }), el('th', { class: 'n', text: 'Same as you' })), ...body);
}
const statsSoon = debounce(refreshLabelPanel, 400);

async function scoreLabels() {
  const out = $('#l-score-out'), button = $('#l-score');
  button.disabled = true; out.replaceChildren(el('p', { class: 'rv-hint', text: 'Scoring…' }));
  try {
    const result = await post('api/match/' + S.m.id + '/score', { points: pointsFromLabels() });
    if (result.error) { out.replaceChildren(el('p', { class: 'rv-hint', text: result.error })); return; }
    const best = Math.max(...result.rows.slice(0, 5).map((row) => row.spell_f1));
    const fmt = (v) => (v === null || v === undefined ? '–' : v.toFixed(3));
    out.replaceChildren(
      el('table', { class: 'rv-table' },
        el('tr', null, el('th', { text: 'Against your labels' }), el('th', { class: 'n', text: 'spellF1' }), el('th', { class: 'n', text: 'macroF1' }), el('th', { class: 'n', text: 'Found' })),
        ...result.rows.map((row) => el('tr', { class: row.spell_f1 === best ? 'best' : '' },
          el('td', null, row.name, el('small', { text: row.note })),
          el('td', { class: 'n' }, fmt(row.spell_f1), el('small', { text: row.spell_low === null ? '' : fmt(row.spell_low) + '–' + fmt(row.spell_high) })),
          el('td', { class: 'n', text: fmt(row.macro_f1) }), el('td', { class: 'n', text: row.found + ' / ' + row.true })))),
      el('p', { class: 'rv-hint', text: result.possessions + ' possessions in ' + (result.labelled_seconds / 60).toFixed(1) + ' labelled minutes. ' + result.note
        + (result.possessions < 40 ? ' With this few possessions the ranges are wide: differences of a few hundredths mean nothing yet.' : '') }));
  } catch (error) { out.replaceChildren(el('p', { class: 'rv-hint', text: 'Could not score: ' + error.message })); }
  finally { button.disabled = false; }
}

// ---------------------------------------------------------------- explanations: pop-up first, the full story on request
const EX = { articles: new Map(), key: null };
async function loadExplanations() {
  try {
    const page = new DOMParser().parseFromString(await (await fetch('static/explain.html')).text(), 'text/html');
    for (const article of page.querySelectorAll('article')) EX.articles.set(article.dataset.key, document.importNode(article, true));
  } catch (error) { /* the page still works without them */ }
}
function topicsHere() {
  const practice = S.m && S.m.annotatable, skip = practice ? ['tags', 'overlay'] : ['events', 'mine'];
  return [...EX.articles.keys()].filter((key) => !skip.includes(key));
}
function nearestOf(team) {
  const A = S.A, f = S.frame, bx = A.ball[f * 2], by = A.ball[f * 2 + 1];
  if (bx === -32768) return null;
  let best = Infinity;
  for (let p = 0; p < S.P; p++) {
    if (S.m.slots[p] !== team) continue;
    const x = A.players[(f * S.P + p) * 2], y = A.players[(f * S.P + p) * 2 + 1];
    if (x !== -32768) best = Math.min(best, Math.hypot(x - bx, y - by) / 100);
  }
  return best;
}
/* The numbers behind the current frame, for the pop-up and the top of the full explanation. */
function liveFacts(key) {
  const A = S.A, f = S.frame, facts = [], pct = (v) => Math.round(v / 2.55) + '%';
  const three = (arr) => pct(arr[f * 3]) + ' / ' + pct(arr[f * 3 + 1]) + ' / ' + pct(arr[f * 3 + 2]);
  const graded = A.final[f] < 3;
  if (key === 'final' || key === 'confidence') {
    facts.push(['Model says', who(A.final[f], true)]);
    if (graded) {
      facts.push(['Sure of home / away / dead', three(A.post)]);
      facts.push(['A change of team here costs', A.seen[f] ? '4 (ball in the data)' : '12 (no ball)']);
    }
  } else if (key === 'prev') {
    facts.push(['Previous version says', who(A.prev[f], true)], ['Current model says', who(A.final[f], true)]);
    if (graded) facts.push(['Here they', A.prev[f] === A.final[f] ? 'agree' : 'disagree']);
  } else if (key === 'ball_view' || key === 'overlay' || key === 'keeper') {
    if (key === 'ball_view') facts.push(['Ball view alone says', who(A.ball_view[f], true)]);
    const d = [nearestOf(0), nearestOf(1)];
    if (d[0] === null) facts.push(['Ball', 'not in the data at this frame']);
    else for (const t of [0, 1]) {
      facts.push(['Nearest ' + S.m.teams[t] + ' player', isFinite(d[t]) ? d[t].toFixed(1) + ' m from the ball, grip up to ' + Math.max(0, 1 - d[t] / 2.5).toFixed(2) : 'none tracked']);
    }
  } else if (key === 'player_view') {
    facts.push(['Player view alone says', who(A.player_view[f], true)]);
    if (graded) facts.push(['p(home) / p(away) / p(dead)', three(A.pview)], ['Its weight on “which side” here', A.seen[f] ? '0.0625 (ball in the data)' : '0.25 (no ball)']);
  } else if (key === 'doubt') {
    const names = { 1: 'low confidence', 2: 'views disagree', 4: 'versions disagree', 8: 'very short possession', 16: 'ball not in the data' };
    const on = Object.entries(names).filter(([bit]) => A.reasons[f] & Number(bit)).map(([, name]) => name);
    facts.push(['Doubt here', (A.doubt[f] / 255).toFixed(2) + ' (a moment starts at 0.30)'], ['Signs present', on.join(', ') || 'none']);
    const m = momentAt(f);
    if (m) facts.push(['Inside', 'doubtful moment no. ' + m.rank + ' of ' + S.moments.length + ' (' + m.weight.toFixed(1) + ' doubt-seconds)']);
  } else if (key === 'seen') {
    facts.push(['Ball at this frame', A.seen[f] === 0 ? 'not in the data' : A.seen[f] === 2 && S.m.kind === 'challenge' ? 'filled in from TV or bridged' : 'seen by the tracker']);
  } else if (key === 'events' && A.events) {
    facts.push(['Event data says', who(A.events[f], true)], ['Model says', who(A.final[f], true)]);
  } else if (key === 'mine' && S.m.annotatable) {
    facts.push(['Your label here', S.L[f] === 3 ? 'none' : who(S.L[f], true)]);
  }
  return facts;
}
function fillFacts(list, key) {
  const facts = S.m ? liveFacts(key) : [];
  list.replaceChildren(...facts.flatMap(([k, v]) => [el('dt', { text: k }), el('dd', { text: v })]));
  list.hidden = !facts.length;
  return facts.length;
}
function closePop() { $('#pop').hidden = true; }
function openPop(key, x, y) {
  const article = EX.articles.get(key);
  if (!article) return;
  closeMenu();
  EX.key = key;
  const pop = $('#pop');
  $('#pop-title').textContent = article.dataset.title;
  $('#pop-text').textContent = article.querySelector('.short').textContent;
  $('#pop-when').hidden = !fillFacts($('#pop-facts'), key);
  if (S.m) $('#pop-when').textContent = 'At this frame (' + clockLabel(S.frame) + ')';
  pop.hidden = false;
  const w = pop.offsetWidth, h = pop.offsetHeight;
  pop.style.left = clamp(x - w / 2, 8, innerWidth - w - 8) + 'px';
  pop.style.top = (y + 14 + h > innerHeight - 8 ? Math.max(8, y - h - 14) : y + 14) + 'px';
  $('#pop-more').focus({ preventScroll: true });
}
function showExplain(key) {
  const article = EX.articles.get(key);
  if (!article) return;
  closePop();
  EX.key = key;
  $('#explain-nav').replaceChildren(...topicsHere().map((k) => el('button', { type: 'button', 'data-topic': k, 'aria-current': String(k === key), text: EX.articles.get(k).dataset.title })));
  $('#explain-title').textContent = article.dataset.title;
  $('#explain-when').hidden = !fillFacts($('#explain-facts'), key);
  if (S.m) $('#explain-when').textContent = 'At the frame on screen (' + clockLabel(S.frame) + ')';
  const copy = article.cloneNode(true);
  copy.querySelector('.short').className = 'lead';
  $('#explain-content').replaceChildren(...copy.childNodes);
  $('#explain-content').scrollTop = 0;
  if (!$('#explain').open) { pause(); $('#explain').showModal(); }
}

// ---------------------------------------------------------------- a small context menu
function closeMenu() { $('#menu').hidden = true; }
function openMenu(x, y, items) {
  closePop();
  const menu = $('#menu');
  menu.replaceChildren(...items.filter(Boolean).map(([label, run]) => {
    const button = el('button', { type: 'button', role: 'menuitem', text: label });
    button.addEventListener('click', () => { closeMenu(); run(); });
    return button;
  }));
  menu.hidden = false;
  menu.style.left = clamp(x, 8, innerWidth - menu.offsetWidth - 8) + 'px';
  menu.style.top = clamp(y, 8, innerHeight - menu.offsetHeight - 8) + 'px';
  menu.firstChild.focus({ preventScroll: true });
}
function momentMenu(m, x, y) {
  const mine = notesOf(m).length;
  openMenu(x, y, [
    ['Take a note about this moment', () => newNote(m)],
    mine ? ['Show its notes (' + mine + ')', () => { showTab('notes'); $('#note-filter').value = 'moment'; setCurrent(m, false); renderNotes(); }] : null,
    [S.watched.has(m.id) ? 'Mark as not watched' : 'Mark as watched', () => markWatched(m)],
    ['Why is it doubtful?', () => { seek(m.start); showExplain('doubt'); }],
  ]);
}

// ---------------------------------------------------------------- notes: an observation, a moment, optionally a picture you can draw on
const N = { list: [], topics: [], editing: null, base: null, strokes: [], tool: 'pen', colour: '#ffe14d', drawing: null, dirty: false };
const PICTURE = [1280, 720];
async function loadNotes() {
  try { const r = await getJSON('api/match/' + S.m.id + '/notes'); N.list = r.notes; N.topics = r.topics; }
  catch (error) { N.list = []; }
  $('#note-topic').replaceChildren(...N.topics.map((t) => el('option', { value: t, text: t[0].toUpperCase() + t.slice(1) })));
  renderNotes(); refreshMoments(); S.redraw = true;
}
function notesOf(m) { return N.list.filter((n) => n.moment === m.id || (n.frame >= m.start && n.frame < m.end)); }
function renderNotes() {
  const only = $('#note-filter').value === 'moment' && S.cur;
  const list = only ? notesOf(S.cur) : N.list;
  $('#tab-notes').textContent = 'Notebook' + (N.list.length ? ' (' + N.list.length + ')' : '');
  const cards = list.map((n) => {
    const m = n.moment ? S.moments.find((x) => x.id === n.moment) : momentAt(n.frame);
    const head = el('button', { type: 'button', class: 'rv-note-go', 'data-go': n.id, title: 'Go to this frame' },
      el('strong', { text: clockLabel(n.frame) }), el('span', { class: 'rv-chip', text: n.topic }),
      m ? el('span', { class: 'rv-chip doubt', text: 'moment no. ' + m.rank }) : null);
    const picture = n.image ? el('img', { class: 'rv-note-img', loading: 'lazy', alt: 'Picture attached to this note',
      src: 'api/match/' + S.m.id + '/note-image/' + n.image + '?v=' + (n.image_version || 0), 'data-edit': n.id }) : null;
    return el('li', { class: 'rv-note' }, head, n.text ? el('p', { text: n.text }) : null, picture,
      el('div', { class: 'rv-note-foot' }, el('small', { text: (n.author && n.author !== 'local' ? n.author + ' · ' : '') + n.updated.replace('T', ' ').slice(0, 16) }),
        el('button', { type: 'button', class: 'rv-btn ghost small', 'data-edit': n.id, text: 'Edit' })));
  });
  $('#note-list').replaceChildren(...(cards.length ? cards : [el('li', { class: 'rv-empty', text: only ? 'No notes on this moment yet.' : 'No notes yet. Press T, or right-click the picture or a doubtful moment.' })]));
}
function captureFrame() {
  const base = document.createElement('canvas'), f = S.frame;
  [base.width, base.height] = PICTURE;
  const ctx = base.getContext('2d'), hidden = S.veil && S.m.kind === 'challenge';
  ctx.fillStyle = '#05090e'; ctx.fillRect(0, 0, base.width, base.height);
  if (!hidden) {
    if (usingVideo()) {
      ctx.drawImage(video, 0, 0, base.width, base.height);
      if (S.overlay && S.source === 'tactical') ctx.drawImage($('#overlay'), 0, 0, base.width, base.height);
    } else ctx.drawImage($('#pitch'), 0, 0, base.width, base.height);
  }
  ctx.fillStyle = '#0b1522d9'; ctx.fillRect(0, base.height - 34, base.width, 34);
  ctx.fillStyle = '#fff'; ctx.font = '15px Poppins, Arial'; ctx.textBaseline = 'middle';
  const says = S.blind ? '' : '   ·   model: ' + who(S.A.final[f], true) + '   ·   ball view: ' + who(S.A.ball_view[f], true) + '   ·   player view: ' + who(S.A.player_view[f], true);
  ctx.fillText(S.m.title + '   ·   ' + clockLabel(f) + '   ·   frame ' + f + says, 14, base.height - 17);
  return base;
}
function paintPicture() {
  const canvas = $('#note-canvas'), ctx = canvas.getContext('2d');
  [canvas.width, canvas.height] = PICTURE;
  if (N.base) ctx.drawImage(N.base, 0, 0);
  ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = 5;
  for (const s of N.strokes.concat(N.drawing ? [N.drawing] : [])) {
    ctx.strokeStyle = ctx.fillStyle = s.colour;
    const [a, b] = [s.points[0], s.points[s.points.length - 1]];
    ctx.beginPath();
    if (s.tool === 'pen') { ctx.moveTo(a[0], a[1]); for (const p of s.points) ctx.lineTo(p[0], p[1]); ctx.stroke(); }
    else if (s.tool === 'circle') { ctx.ellipse((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, Math.max(4, Math.abs(b[0] - a[0]) / 2), Math.max(4, Math.abs(b[1] - a[1]) / 2), 0, 0, Math.PI * 2); ctx.stroke(); }
    else {
      const angle = Math.atan2(b[1] - a[1], b[0] - a[0]), head = 22;
      ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(b[0], b[1]);
      ctx.lineTo(b[0] - head * Math.cos(angle - 0.45), b[1] - head * Math.sin(angle - 0.45));
      ctx.lineTo(b[0] - head * Math.cos(angle + 0.45), b[1] - head * Math.sin(angle + 0.45));
      ctx.closePath(); ctx.fill();
    }
  }
}
function showPicture(on) {
  $('#note-canvas').hidden = !on;
  $('#draw-tools').hidden = !on;
  $('#note-snap').textContent = on ? 'Retake from the frame on screen' : 'Add a picture of this frame';
  if (on) paintPicture();
}
function openNote(note, frame, moment) {
  pause();
  N.editing = note ? { id: note.id, frame: note.frame, moment: note.moment } : { id: null, frame, moment: moment ? moment.id : null };
  N.base = null; N.strokes = []; N.drawing = null; N.dirty = false; N.remove = false;
  const m = N.editing.moment ? S.moments.find((x) => x.id === N.editing.moment) : null;
  $('#note-title').textContent = note ? 'Edit note' : 'New note';
  $('#note-where').textContent = S.m.title + ' · ' + clockLabel(N.editing.frame) + ' · frame ' + N.editing.frame.toLocaleString('en-US')
    + (m ? ' · doubtful moment no. ' + m.rank + ' (' + m.reasons.join(', ') + ')' : '');
  $('#note-topic').value = note ? note.topic : 'other';
  $('#note-text').value = note ? note.text : '';
  $('#note-delete').hidden = !note;
  $('#note-rule').textContent = S.m.annotatable ? '' : 'Describe what is happening and what may be confusing the model. Do not record who really has the ball: on this match that would be hand-labelling, which the rules forbid.';
  $('#note-state').textContent = '';
  showPicture(false);
  if (note && note.image) {
    const image = new Image();
    image.onload = () => { N.base = image; showPicture(true); };
    image.src = 'api/match/' + S.m.id + '/note-image/' + note.image + '?v=' + (note.image_version || 0);
  }
  $('#note-dialog').showModal();
  $('#note-text').focus();
}
function newNote(moment) {
  if (!S.m) return;
  if (moment && !(S.frame >= moment.play_start && S.frame < moment.play_end)) seek(moment.start);
  openNote(null, S.frame, moment || momentAt(S.frame));
}
async function saveNote() {
  const body = { id: N.editing.id, frame: N.editing.frame, moment: N.editing.moment, topic: $('#note-topic').value, text: $('#note-text').value.trim() };
  if (!$('#note-canvas').hidden && N.dirty) body.image = $('#note-canvas').toDataURL('image/jpeg', 0.86);
  else if (N.remove) body.image = 'remove';
  if (!body.text && !body.image && !N.editing.id) { $('#note-state').textContent = 'Write something or add a picture first.'; return; }
  $('#note-save').disabled = true; $('#note-state').textContent = 'Saving…';
  try {
    await post('api/match/' + S.m.id + '/notes', body);
    $('#note-dialog').close();
    await loadNotes();
    showTab('notes');
  } catch (error) { $('#note-state').textContent = 'Not saved: ' + error.message; }
  finally { $('#note-save').disabled = false; }
}
function bindExtras() {
  // explanations
  document.addEventListener('click', (e) => {
    const goto = e.target.closest('[data-goto]');
    if (goto) { e.preventDefault(); showExplain(goto.dataset.goto); return; }
    const topic = e.target.closest('[data-topic]');
    if (topic) { showExplain(topic.dataset.topic); return; }
    const trigger = e.target.closest('[data-explain]');
    if (trigger) { const r = trigger.getBoundingClientRect(); openPop(trigger.dataset.explain, r.left + r.width / 2, r.bottom - 8); e.stopPropagation(); return; }
    if (!e.target.closest('#pop')) closePop();
    if (!e.target.closest('#menu')) closeMenu();
  });
  $('#pop-close').addEventListener('click', closePop);
  $('#pop-more').addEventListener('click', () => showExplain(EX.key));
  $('#explain-open').addEventListener('click', () => showExplain(EX.key && topicsHere().includes(EX.key) ? EX.key : 'final'));
  window.addEventListener('resize', () => { closePop(); closeMenu(); });

  // context menus
  $('#stage').addEventListener('contextmenu', (e) => {
    if (!S.m) return;
    e.preventDefault();
    const m = momentAt(S.frame);
    openMenu(e.clientX, e.clientY, [
      ['Take a note about this frame', () => newNote(null)],
      m ? ['Take a note about doubtful moment no. ' + m.rank, () => newNote(m)] : null,
      ['What is the model doing here?', () => showExplain('final')],
      S.source === 'tactical' ? ['What do the rings mean?', () => showExplain('overlay')] : null,
    ]);
  });
  $('#moments').addEventListener('contextmenu', (e) => {
    const row = e.target.closest('[data-id], [data-note-moment]');
    if (!row) return;
    e.preventDefault();
    momentMenu(S.moments.find((m) => m.id === Number(row.dataset.id || row.dataset.noteMoment)), e.clientX, e.clientY);
  });
  $('#timeline').addEventListener('contextmenu', (e) => {
    if (!S.m) return;
    e.preventDefault();
    const r = $('#timeline').getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    const lane = S.lanes.find((l) => y >= l.y - 2 && y <= l.y + l.h + 2);
    if (x >= GUTTER) { pause(); seek(frameAt(x)); }
    const m = momentAt(S.frame);
    openMenu(e.clientX, e.clientY, [
      ['Take a note at ' + clockLabel(S.frame), () => newNote(null)],
      m ? ['Take a note about doubtful moment no. ' + m.rank, () => newNote(m)] : null,
      lane && EX.articles.has(lane.key) ? ['What is “' + lane.name + '”?', () => showExplain(lane.key)] : null,
    ]);
  });

  // notes
  $('#note-new').addEventListener('click', () => newNote(null));
  $('#note-filter').addEventListener('change', renderNotes);
  $('#note-list').addEventListener('click', (e) => {
    const edit = e.target.closest('[data-edit]'), go = e.target.closest('[data-go]');
    const note = N.list.find((n) => n.id === (edit || go || {}).dataset?.[edit ? 'edit' : 'go']);
    if (!note) return;
    if (edit) { seek(note.frame); openNote(note); }
    else { const m = note.moment ? S.moments.find((x) => x.id === note.moment) : momentAt(note.frame); if (m) setCurrent(m, false); pause(); seek(note.frame); }
  });
  $('#note-snap').addEventListener('click', () => { N.base = captureFrame(); N.strokes = []; N.dirty = true; N.remove = false; showPicture(true); });
  $('#note-unsnap').addEventListener('click', () => { N.base = null; N.strokes = []; N.dirty = false; N.remove = true; showPicture(false); });
  $('#note-undo').addEventListener('click', () => { N.strokes.pop(); paintPicture(); });
  $('#draw-tools').addEventListener('click', (e) => {
    const tool = e.target.closest('[data-tool]'), colour = e.target.closest('[data-colour]');
    if (tool) N.tool = tool.dataset.tool;
    if (colour) N.colour = colour.dataset.colour;
    for (const b of document.querySelectorAll('[data-tool]')) b.setAttribute('aria-pressed', String(b.dataset.tool === N.tool));
    for (const b of document.querySelectorAll('[data-colour]')) b.setAttribute('aria-pressed', String(b.dataset.colour === N.colour));
  });
  const canvas = $('#note-canvas');
  const at = (e) => { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * PICTURE[0], (e.clientY - r.top) / r.height * PICTURE[1]]; };
  canvas.addEventListener('pointerdown', (e) => { canvas.setPointerCapture(e.pointerId); N.drawing = { tool: N.tool, colour: N.colour, points: [at(e)] }; });
  canvas.addEventListener('pointermove', (e) => { if (!N.drawing) return; N.drawing.points.push(at(e)); paintPicture(); });
  const finish = () => { if (!N.drawing) return; if (N.drawing.points.length > 1) { N.strokes.push(N.drawing); N.dirty = true; } N.drawing = null; paintPicture(); };
  canvas.addEventListener('pointerup', finish);
  canvas.addEventListener('pointercancel', finish);
  $('#note-save').addEventListener('click', saveNote);
  $('#note-cancel').addEventListener('click', () => $('#note-dialog').close());
  $('#note-delete').addEventListener('click', async () => {
    if (!confirm('Delete this note and its picture?')) return;
    try { await post('api/match/' + S.m.id + '/notes-delete', { id: N.editing.id }); $('#note-dialog').close(); await loadNotes(); }
    catch (error) { $('#note-state').textContent = 'Not deleted: ' + error.message; }
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closePop(); closeMenu(); } });
}

// ---------------------------------------------------------------- wiring
function showTab(name) {
  for (const tab of document.querySelectorAll('.rv-tab')) tab.setAttribute('aria-selected', String(tab.dataset.tab === name));
  for (const panel of document.querySelectorAll('.rv-panel')) panel.hidden = panel.id !== 'panel-' + name;
  if (name === 'labels') refreshLabelPanel();
}
function setSpeed(value) { S.speed = value; $('#speed').value = String(value); if (usingVideo()) video.playbackRate = value; }
function cycle(select, direction) { const i = clamp(select.selectedIndex + direction, 0, select.options.length - 1); select.selectedIndex = i; select.dispatchEvent(new Event('change')); }

function bind() {
  if (hasFrameCallback) video.requestVideoFrameCallback(onVideoFrame);
  video.addEventListener('seeked', () => { S.pendingSeek = false; });
  video.addEventListener('play', () => { if (usingVideo()) { S.playing = true; S.pendingSeek = false; updatePlayButton(); } });
  video.addEventListener('pause', () => {
    if (!usingVideo()) return;
    S.playing = false; updatePlayButton();
    if (S.m && !S.pendingSeek) {                              // land exactly on the frame that is left on screen
      const t = tacticalFor(Math.floor(video.currentTime * S.fps + 0.001));
      if (t >= 0 && t !== S.frame) { S.frame = t; renderFrame(false); }
    }
  });
  video.addEventListener('error', () => { S.pendingSeek = false; if (usingVideo()) stageMessage('The video could not be played. Use the pitch map, or start the server with --tactical-video.'); });

  $('#matches').addEventListener('click', (e) => { const b = e.target.closest('.rv-match'); if (b && (!S.m || b.dataset.id !== S.m.id)) openMatch(b.dataset.id); });
  $('#play').addEventListener('click', () => (S.playing ? pause() : play()));
  $('#back-1s').addEventListener('click', () => step(-S.fps));
  $('#back-1f').addEventListener('click', () => step(-1));
  $('#fwd-1f').addEventListener('click', () => step(1));
  $('#fwd-1s').addEventListener('click', () => step(S.fps));
  $('#speed').addEventListener('change', (e) => setSpeed(Number(e.target.value)));
  $('#source').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) setSource(b.dataset.source); });
  $('#ov').addEventListener('change', (e) => { S.overlay = e.target.checked; renderFrame(false); });
  $('#map').addEventListener('change', (e) => { S.map = e.target.checked; renderFrame(false); });
  $('#theme').addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    store.set('theme', root.dataset.theme);
    $('#theme').textContent = root.dataset.theme === 'dark' ? 'Light' : 'Dark';
    readColours(); layout();
  });
  $('#help-open').addEventListener('click', () => $('#help').showModal());
  $('#banner-close').addEventListener('click', () => { store.set($('#banner').dataset.key, true); $('#banner').hidden = true; layout(); });
  $('#blind').addEventListener('change', (e) => {
    S.blind = e.target.checked; store.set('blind', S.blind);
    root.classList.toggle('blind', S.blind);
    buildLanes(); refreshMoments(); refreshLabelPanel(); layout();
  });
  for (const tab of document.querySelectorAll('.rv-tab')) tab.addEventListener('click', () => showTab(tab.dataset.tab));

  $('#m-filter').addEventListener('change', refreshMoments);
  $('#m-sort').addEventListener('change', refreshMoments);
  $('#m-prev').addEventListener('click', () => neighbour(-1));
  $('#m-next').addEventListener('click', () => neighbour(1));
  $('#loop').addEventListener('change', (e) => { S.loop = e.target.checked; });
  $('#reel').addEventListener('change', (e) => { S.reel = e.target.checked; if (S.reel && !S.playing) play(); });
  $('#moments').addEventListener('click', (e) => {
    const noteButton = e.target.closest('[data-note-moment]');
    if (noteButton) { newNote(S.moments.find((m) => m.id === Number(noteButton.dataset.noteMoment))); return; }
    const b = e.target.closest('.rv-moment');
    if (b) openMoment(S.moments.find((m) => m.id === Number(b.dataset.id)));
  });

  for (const b of document.querySelectorAll('.rv-btn.lab')) b.addEventListener('click', () => applyLabel(Number(b.dataset.lab)));
  $('#fill').addEventListener('click', fillFromModel);
  $('#del-boundary').addEventListener('click', removeBoundary);
  $('#undo').addEventListener('click', () => undo(false));
  $('#redo').addEventListener('click', () => undo(true));
  $('#follow').addEventListener('change', (e) => { S.follow = e.target.checked; });
  $('#l-score').addEventListener('click', scoreLabels);
  $('#l-by').addEventListener('change', (e) => { store.set('labelledBy', e.target.value); saveSoon(); });
  $('#l-export').addEventListener('click', async () => {
    try { const r = await post('api/match/' + S.m.id + '/export', {}); $('#l-export-out').textContent = 'Wrote ' + r.path + ' (' + (r.labelled_frames / S.fps / 60).toFixed(1) + ' labelled minutes).'; }
    catch (error) { $('#l-export-out').textContent = 'Could not write the file: ' + error.message; }
  });
  $('#l-clear').addEventListener('click', () => {
    if (!confirm('Remove every label on this practice match? You can undo it afterwards with Ctrl + Z.')) return;
    snapshot(); S.L.fill(3); S.openEnd = null; labelsChanged();
  });
  const saveNotes = debounce(async () => {
    try { await post('api/notebook', { text: $('#notes').value }); $('#notes-state').textContent = 'Saved'; } catch (error) { $('#notes-state').textContent = 'Not saved: ' + error.message; }
  }, 800);
  $('#notes').addEventListener('input', () => { $('#notes-state').textContent = '…'; saveNotes(); });

  $('#z-in').addEventListener('click', () => zoom(0.5));
  $('#z-out').addEventListener('click', () => zoom(2));
  $('#z-fit').addEventListener('click', () => setView(0, S.n));
  $('#z-min').addEventListener('click', () => setView(S.frame - 30 * S.fps, S.frame + 30 * S.fps));
  $('#tl-follow').addEventListener('change', (e) => { S.tlFollow = e.target.checked; });

  const lanes = $('#timeline');
  const local = (e, node) => { const r = node.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  lanes.addEventListener('pointerdown', (e) => {
    if (!S.m || e.button !== 0) return;
    const [x, y] = local(e, lanes);
    if (x < GUTTER) return;
    lanes.setPointerCapture(e.pointerId);
    const lane = S.lanes.find((l) => y >= l.y - 2 && y <= l.y + l.h + 2);
    const perPx = (S.view.f1 - S.view.f0) / (tlWidth() - GUTTER);
    const boundary = lane && lane.kind === 'mine' ? boundaryNear(frameAt(x), Math.max(1, 6 * perPx)) : null;
    if (boundary !== null) { startDrag(boundary); return; }
    pause(); S.scrub = true; seek(frameAt(x));
  });
  lanes.addEventListener('pointermove', (e) => {
    if (!S.m) return;
    const [x, y] = local(e, lanes);
    if (S.drag) { dragTo(frameAt(x)); return; }
    if (S.scrub) { S.pendingScrub = clamp(frameAt(x), 0, S.n - 1); return; }
    const lane = S.lanes.find((l) => y >= l.y - 2 && y <= l.y + l.h + 2);
    const perPx = (S.view.f1 - S.view.f0) / (tlWidth() - GUTTER);
    lanes.style.cursor = x < GUTTER ? (lane && EX.articles.has(lane.key) ? 'help' : 'default')
      : lane && lane.kind === 'mine' && boundaryNear(frameAt(x), Math.max(1, 6 * perPx)) !== null ? 'ew-resize' : 'crosshair';
  });
  lanes.addEventListener('click', (e) => {                 // a lane's name explains the lane
    const [x, y] = local(e, lanes);
    const lane = x < GUTTER && S.lanes.find((l) => y >= l.y - 2 && y <= l.y + l.h + 2);
    if (lane && EX.articles.has(lane.key)) { openPop(lane.key, e.clientX, e.clientY); e.stopPropagation(); }
  });
  const release = () => {
    if (S.drag) { S.drag = null; labelsChanged(); }
    S.scrub = false;
  };
  lanes.addEventListener('pointerup', release);
  lanes.addEventListener('pointercancel', release);
  lanes.addEventListener('wheel', (e) => {
    if (!S.m) return;
    e.preventDefault();
    const [x] = local(e, lanes), span = S.view.f1 - S.view.f0;
    if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) { const d = (e.deltaX || e.deltaY) / (tlWidth() - GUTTER) * span; setView(S.view.f0 + d, S.view.f1 + d); }
    else zoom(Math.exp(clamp(e.deltaY, -240, 240) * 0.0022), frameAt(Math.max(x, GUTTER)));
  }, { passive: false });
  const over = $('#overview');
  const jump = (e) => { const [x] = local(e, over), f = clamp(Math.round(x / over.clientWidth * S.n), 0, S.n - 1), span = S.view.f1 - S.view.f0; setView(f - span / 2, f + span / 2); S.pendingScrub = f; };
  over.addEventListener('pointerdown', (e) => { if (!S.m) return; over.setPointerCapture(e.pointerId); pause(); S.scrub = true; jump(e); });
  over.addEventListener('pointermove', (e) => { if (S.scrub && over.hasPointerCapture(e.pointerId)) jump(e); });
  over.addEventListener('pointerup', () => { S.scrub = false; });
  over.addEventListener('pointercancel', () => { S.scrub = false; });

  new ResizeObserver(() => layout()).observe($('#stage'));
  new ResizeObserver(() => layout()).observe(lanes);

  document.addEventListener('keyup', (e) => { if (e.code === 'Space' && e.target.tagName === 'BUTTON') e.preventDefault(); });
  document.addEventListener('keydown', (e) => {
    if (!S.m || document.querySelector('dialog[open]')) return;
    const tag = e.target.tagName;
    if (tag === 'INPUT' && e.target.type === 'text' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key, ctrl = e.ctrlKey || e.metaKey;
    const can = S.m.annotatable;
    let used = true;
    if (e.code === 'Space') S.playing ? pause() : play();
    else if (key === 'ArrowLeft' || key === 'ArrowRight') step((key === 'ArrowLeft' ? -1 : 1) * (ctrl ? 5 * S.fps : e.shiftKey ? S.fps : 1));
    else if (ctrl && key === 'z') can && undo(e.shiftKey);
    else if (ctrl && key === 'y') can && undo(true);
    else if (ctrl || e.altKey) used = false;
    else if (key === '[') cycle($('#speed'), -1);
    else if (key === ']') cycle($('#speed'), 1);
    else if (key === 'n') neighbour(1);
    else if (key === 'p') neighbour(-1);
    else if (key === 'r') { $('#loop').checked = S.loop = !S.loop; }
    else if (key === 'w') { const m = S.cur || momentAt(S.frame); if (m) markWatched(m); }
    else if (key === 'v') { const buttons = [...$('#source').children]; if (buttons.length > 1) setSource(buttons[(buttons.findIndex((b) => b.dataset.source === S.source) + 1) % buttons.length].dataset.source); }
    else if (key === 'o') { $('#ov').checked = S.overlay = !S.overlay; renderFrame(false); }
    else if (key === 'm') { $('#map').checked = S.map = !S.map; renderFrame(false); }
    else if (key === '+' || key === '=') zoom(0.5);
    else if (key === '-') zoom(2);
    else if (key === 'f') setView(0, S.n);
    else if (key === '?') $('#help').showModal();
    else if (key === 't') newNote(null);
    else if (key === 'i') showExplain(EX.key && topicsHere().includes(EX.key) ? EX.key : 'final');
    else if (can && (key === '1' || key === 'h')) applyLabel(0);
    else if (can && (key === '2' || key === 'a')) applyLabel(1);
    else if (can && (key === '3' || key === 'd')) applyLabel(2);
    else if (can && (key === '0' || key === 'x')) applyLabel(3);
    else if (can && key === 'Enter' && tag !== 'BUTTON') fillFromModel();
    else if (can && key === 'Backspace') removeBoundary();
    else used = false;
    if (used) e.preventDefault();
  });
  window.addEventListener('beforeunload', () => { if (S.m) store.set('frame.' + S.m.id, S.frame); });
}

init();
})();
