/* Block renderers for workspace cards. Every card is a list of blocks; each block type below draws one kind of
   football-shaped visual from data the local server sent. Text goes in with textContent; nothing here evaluates
   strings as code. Blue = Barcelona, orange = Athletic, grey = dead ball. */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const NAMES = { H: 'Barcelona', A: 'Athletic', D: 'Dead ball' };
  const ICONS = {
    mic: 'M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3ZM6 11a6 6 0 0 0 12 0M12 17v4M8.5 21h7',
    send: 'M4 12 20 4l-6 16-3-7-7-1Z', chat: 'M4 5h16v11H9l-5 4V5Z', settings: 'M4 7h10M18 7h2M4 17h2M10 17h10M16 4.5v5M8 14.5v5',
    trash: 'M5 7h14M9 7V4h6v3M7 7l1 13h8l1-13', plus: 'M12 5v14M5 12h14', close: 'M6 6l12 12M18 6 6 18', alert: 'M12 4 2.5 20h19L12 4ZM12 10v5M12 17.6v.2',
    pin: 'M9 4h6l-1 6 3 3H7l3-3-1-6ZM12 13v7', expand: 'M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5', check: 'M5 12.5 10 17l9-10', lock: 'M6 11h12v9H6zM8.5 11V8a3.5 3.5 0 0 1 7 0v3',
    info: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18ZM12 11v6M12 7.6v.2', play: 'M7 5v14l12-7-12-7Z', pause: 'M8 5v14M16 5v14',
    chart: 'M4 20V10M10 20V4M16 20v-8M2 20h20', overview: 'M4 4h7v7H4zM13 4h7v4h-7zM13 10h7v10h-7zM4 13h7v7H4z', trophy: 'M8 4h8v5a4 4 0 0 1-8 0V4ZM8 6H4.5a3.5 3.5 0 0 0 3.6 3.5M16 6h3.5a3.5 3.5 0 0 1-3.6 3.5M12 13v4M8.5 20h7',
    timeline: 'M3 7h8M13 7h8M3 12h4M9 12h12M3 17h11M16 17h5', layers: 'm12 4 9 4.5-9 4.5-9-4.5L12 4ZM3 13l9 4.5 9-4.5M3 17l9 4.5 9-4.5', grid: 'M4 4h16v16H4zM4 10h16M4 15h16M10 4v16M15 4v16',
    histogram: 'M3 20h18M5 20v-5h3v5M10.5 20V8h3v12M16 20v-9h3v9', link: 'M9 15 15 9M10 6l1.5-1.5a4.2 4.2 0 0 1 6 6L16 12M14 18l-1.5 1.5a4.2 4.2 0 0 1-6-6L8 12',
    'eye-off': 'M3 3l18 18M10.6 6.2A9.6 9.6 0 0 1 12 6c5 0 8.5 4 9.5 6a12 12 0 0 1-3 3.8M6.6 7.6A12.4 12.4 0 0 0 2.5 12c1 2 4.5 6 9.5 6 1.3 0 2.5-.3 3.6-.7M9.9 9.9a3 3 0 0 0 4.2 4.2',
    pitch: 'M3 5h18v14H3zM12 5v14M12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5ZM3 9h3v6H3M21 9h-3v6h3', tag: 'M20.6 13.4 13.4 20.6a2 2 0 0 1-2.8 0L3 13V3h10l7.6 7.6a2 2 0 0 1 0 2.8ZM7.5 7.4v.2',
    shield: 'M12 3 4.5 6v6c0 4.5 3.2 7.8 7.5 9 4.3-1.2 7.5-4.5 7.5-9V6L12 3ZM8.5 12l2.5 2.5 4.5-5', book: 'M4 5c3-1.3 5.5-1.3 8 0v15c-2.5-1.3-5-1.3-8 0V5ZM12 5c2.5-1.3 5-1.3 8 0v15c-3-1.3-5.5-1.3-8 0',
    target: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18ZM12 7.5a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9ZM12 11.8v.4', flow: 'M4 5h6v5H4zM14 14h6v5h-6zM10 7.5h4a3 3 0 0 1 3 3V14',
    spark: 'M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3ZM18.5 16l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2Z', note: 'M5 4h14v12l-4 4H5V4ZM15 20v-4h4M8 9h8M8 13h5',
  };

  function el(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined && text !== null) n.textContent = text; return n; }
  function svg(tag, attrs) { const n = document.createElementNS(NS, tag); for (const k in attrs || {}) n.setAttribute(k, attrs[k]); return n; }
  function icon(name) { const s = svg('svg', { viewBox: '0 0 24 24', class: 'ui-icon', 'aria-hidden': 'true' }); s.appendChild(svg('path', { d: ICONS[name] || ICONS.chart })); return s; }
  function mmss(seconds) { const neg = seconds < 0; const s = Math.abs(Math.round(seconds)); return (neg ? '-' : '') + Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); }
  function fmt(v, f) {
    if (v === null || v === undefined || Number.isNaN(v)) return '–';
    if (f === 'percent') return Math.round(v * 100) + '%';
    if (f === 'minutes') return v.toFixed(1) + ' min';
    if (f === 'int') return String(Math.round(v));
    if (f === 'fixed2') return v.toFixed(2);
    return String(Math.round(v * 100) / 100);
  }
  function rich(text) {                                   // paragraphs with **bold**, built from text nodes only
    const box = el('div', 'b-text');
    for (const para of String(text || '').split(/\n+/)) {
      if (!para.trim()) continue;
      const p = el('p');
      para.split('**').forEach((part, i) => { if (!part) return; p.appendChild(i % 2 ? el('strong', null, part) : document.createTextNode(part)); });
      box.appendChild(p);
    }
    return box;
  }
  function legend(items) { const l = el('div', 'b-legend'); for (const [tone, label] of items) { const s = el('span'); s.appendChild(el('i', 'tone-' + tone)); s.appendChild(document.createTextNode(label)); l.appendChild(s); } return l; }

  // ------------------------------------------------------------------ kpis, flow, checks, table
  function kpis(b) {
    const box = el('div', 'b-kpis');
    for (const k of b.items) { const t = el('div', 'b-kpi'); t.appendChild(el('span', null, k.label)); t.appendChild(el('b', null, k.value)); if (k.note) t.appendChild(el('small', null, k.note)); box.appendChild(t); }
    return box;
  }
  function flow(b) {
    const box = el('div', 'b-flow');
    for (const s of b.steps) { const t = el('div', 'flow-step'); t.appendChild(el('b', null, s.title)); t.appendChild(el('p', null, s.text)); box.appendChild(t); }
    return box;
  }
  function checks(b) {
    const label = { pass: 'Holds', warn: 'Look closer', info: 'Context', locked: 'Off limits' }, ic = { pass: 'check', warn: 'alert', info: 'info', locked: 'lock' };
    const box = el('div', 'b-checks');
    for (const c of b.items) {
      const row = el('div', 'check ' + c.status);
      const badge = el('span', 'check-badge'); badge.appendChild(icon(ic[c.status] || 'info')); badge.appendChild(document.createTextNode(c.badge || label[c.status] || c.status));
      row.appendChild(badge); row.appendChild(el('strong', null, c.name)); row.appendChild(el('em', null, c.value || '')); row.appendChild(el('p', null, c.detail || ''));
      box.appendChild(row);
    }
    return box;
  }
  function table(b) {
    const wrap = el('div', 'b-table'), t = el('table'), head = el('tr');
    for (const c of b.columns) head.appendChild(el('th', c.align === 'right' ? 'right' : '', c.label));
    const thead = el('thead'); thead.appendChild(head); t.appendChild(thead);
    const body = el('tbody');
    for (const r of b.rows) { const tr = el('tr', r._highlight ? 'em' : ''); for (const c of b.columns) tr.appendChild(el('td', c.align === 'right' ? 'right' : '', r[c.key] === undefined ? '' : String(r[c.key]))); body.appendChild(tr); }
    t.appendChild(body); wrap.appendChild(t);
    if (b.note) wrap.appendChild(el('p', 'b-note', b.note));
    return wrap;
  }

  // ------------------------------------------------------------------ bars and heatmap
  function bars(b) {
    const horizontal = b.orientation === 'horizontal';
    const box = el('div', 'b-bars ' + (horizontal ? 'horizontal' : 'vertical'));
    if (b.title) box.appendChild(el('p', 'b-title', b.title));
    const totals = b.categories.map((_, i) => b.series.reduce((a, s) => a + (s.values[i] || 0), 0));
    const max = b.max || Math.max(1e-9, ...totals);
    const many = b.series.length > 1;
    const tone = (s, i) => 'tone-' + (s.tone || (many ? ['H', 'A', 'D', 'neutral'][b.series.indexOf(s) % 4] : (b.highlight === undefined || b.highlight === i ? 'brand' : 'neutral')));
    if (horizontal) {
      b.categories.forEach((name, i) => {
        const row = el('div', 'hbar' + (b.highlight === i ? ' em' : ''));
        const n = el('span', 'hbar-name', name); n.title = name; row.appendChild(n);
        const track = el('div', 'hbar-track');
        for (const s of b.series) { const seg = el('div', tone(s, i)); seg.style.width = ((s.values[i] || 0) / max * 100) + '%'; seg.title = s.name + ': ' + fmt(s.values[i], b.format); track.appendChild(seg); }
        if (b.whisker && b.whisker.index === i) { const w = el('div', 'whisker'); w.style.left = (b.whisker.low / max * 100) + '%'; w.style.width = ((b.whisker.high - b.whisker.low) / max * 100) + '%'; w.title = 'Plausible range ' + fmt(b.whisker.low, b.format) + ' to ' + fmt(b.whisker.high, b.format); track.appendChild(w); }
        row.appendChild(track);
        row.appendChild(el('span', 'hbar-value', many && b.format === 'percent' ? '' : fmt(totals[i], b.format)));
        box.appendChild(row);
      });
    } else {
      const plot = el('div', 'bars-plot'), labels = el('div', 'bars-labels');
      b.categories.forEach((name, i) => {
        const col = el('div', 'bars-col');
        if (!(many && b.format === 'percent')) col.appendChild(el('span', null, fmt(totals[i], b.format)));
        const stack = b.series.slice().reverse();
        for (const s of stack) { const seg = el('div', tone(s, i)); seg.style.height = ((s.values[i] || 0) / max * 100) + '%'; seg.title = name + ' · ' + s.name + ': ' + fmt(s.values[i], b.format); col.appendChild(seg); }
        const label = col.querySelector('span'); if (label) label.style.bottom = (totals[i] / max * 100) + '%', label.style.transform = 'none', label.style.marginBottom = '2px';
        plot.appendChild(col); labels.appendChild(el('span', null, name));
      });
      box.appendChild(plot); box.appendChild(labels);
    }
    if (many) box.appendChild(legend(b.series.map((s, i) => [s.tone || ['H', 'A', 'D', 'neutral'][i % 4], s.name])));
    return box;
  }
  function heatmap(b) {
    const box = el('div', 'b-heat');
    box.style.gridTemplateColumns = 'minmax(110px,auto) repeat(' + b.cols.length + ',minmax(0,1fr))';
    box.appendChild(el('div', 'heat-h'));
    for (const c of b.cols) box.appendChild(el('div', 'heat-h', c));
    const max = b.format === 'percent' ? 1 : Math.max(1e-9, ...b.values.flat());
    b.rows.forEach((r, i) => {
      box.appendChild(el('div', 'heat-h heat-r', r));
      b.cols.forEach((c, j) => {
        const v = b.values[i][j], k = Math.max(0, Math.min(1, v / max));
        const cell = el('div', 'heat-c' + (k > 0.55 ? ' hi' : '')); cell.style.setProperty('--k', (k * 0.9).toFixed(3));
        cell.appendChild(el('span', null, fmt(v, b.format)));
        if (b.counts) cell.appendChild(el('small', null, b.counts[i][j] + ' ' + (b.count_unit || '')));
        cell.title = r + ' · ' + c + ': ' + fmt(v, b.format);
        box.appendChild(cell);
      });
    });
    return box;
  }

  // ------------------------------------------------------------------ the possession timeline with evidence lanes
  function strip(segments, start, end) {
    const s = el('div', 'tl-strip'), span = end - start;
    for (const [a, e, l] of segments) {
      if (l === 'X') continue;
      const w = (e - a) / span * 100, m = el('b', l + (w > 0.7 ? ' gap' : ''));
      m.style.left = ((a - start) / span * 100) + '%'; m.style.width = w + '%'; s.appendChild(m);
    }
    return s;
  }
  function lanePath(values, min, max, area) {
    const n = values.length, y = (v) => 100 - (Math.max(min, Math.min(max, v)) - min) / (max - min) * 100;
    let d = '', open = false, first = 0;
    for (let i = 0; i <= n; i++) {
      const v = i < n ? values[i] : null;
      if (v === null || v === undefined) { if (open && area) d += 'L' + i + ' 100L' + first + ' 100Z'; open = false; continue; }
      if (!open) { d += 'M' + i + ' ' + y(v).toFixed(1); first = i; open = true; if (area) d += 'L' + (i + 1) + ' ' + y(v).toFixed(1); }
      else d += 'L' + (i + (area ? 1 : 0.5)) + ' ' + y(v).toFixed(1);
    }
    return d;
  }
  function timeline(b) {
    const box = el('div', 'b-timeline'), span = b.end - b.start, total = span / b.fps;
    if (b.title) box.appendChild(el('p', 'b-title', b.title));
    const plots = [];
    for (const r of b.rows) {
      const row = el('div', 'tl-row'), name = el('div', 'tl-name', r.name);
      const turns = r.segments.filter((s) => s[2] === 'H' || s[2] === 'A').length;
      name.appendChild(el('small', null, turns + (turns === 1 ? ' possession' : ' possessions')));
      const s = strip(r.segments, b.start, b.end);
      row.appendChild(name); row.appendChild(s); box.appendChild(row); plots.push(s);
    }
    for (const lane of b.lanes || []) {
      const row = el('div', 'tl-row tl-lane'), name = el('div', 'tl-name', lane.name);
      if (lane.unit) name.appendChild(el('small', null, lane.unit));
      const n = lane.series[0].values.length;
      const g = svg('svg', { viewBox: '0 0 ' + n + ' 100', preserveAspectRatio: 'none' });
      g.appendChild(svg('rect', { class: 'frame', x: 0, y: 0, width: n, height: 100 }));
      if (lane.mid !== undefined && lane.mid !== null) { const y = 100 - (lane.mid - lane.min) / (lane.max - lane.min) * 100; g.appendChild(svg('line', { class: 'mid', x1: 0, x2: n, y1: y, y2: y })); }
      for (const s of lane.series) {
        if (lane.kind === 'area') g.appendChild(svg('path', { class: 'area s-' + (s.tone || 'neutral'), d: lanePath(s.values, lane.min, lane.max, true), stroke: 'none' }));
        g.appendChild(svg('path', { class: 'line s-' + (s.tone || 'neutral'), d: lanePath(s.values, lane.min, lane.max, false), fill: 'none' }));
      }
      const holder = el('div'); holder.appendChild(g);
      row.appendChild(name); row.appendChild(holder); box.appendChild(row); plots.push(holder);
    }
    const axisRow = el('div', 'tl-row'), axis = el('div', 'tl-axis');
    const step = [5, 10, 15, 30, 60, 120, 300, 600].find((s) => total / s <= 8) || 900;
    for (let t = Math.ceil(b.clock0 / step) * step; t <= b.clock0 + total + 1e-6; t += step) {
      const p = (t - b.clock0) / total * 100, tick = el('span', p > 96 ? 'last' : '', mmss(t));
      tick.style.left = p + '%'; if (p < 3) tick.style.transform = 'none';
      axis.appendChild(tick);
    }
    axisRow.appendChild(el('div', 'tl-name')); axisRow.appendChild(axis); box.appendChild(axisRow);
    const items = [['H', 'Barcelona has the ball'], ['A', 'Athletic has the ball'], ['D', 'Dead ball']];
    for (const lane of b.lanes || []) if (lane.series.length > 1) for (const s of lane.series) items.push([s.tone || 'neutral', s.name]);
    box.appendChild(legend(items));

    // one crosshair for every row: read all answers and curves at the same instant
    const cursor = el('div', 'tl-cursor'), tip = el('div', 'tl-tip'); cursor.hidden = tip.hidden = true; box.appendChild(cursor); box.appendChild(tip);
    const at = (segments, f) => { for (const s of segments) if (f >= s[0] && f < s[1]) return s[2]; return null; };
    function move(e) {
      const ref = plots[0].getBoundingClientRect(), host = box.getBoundingClientRect();
      const frac = (e.clientX - ref.left) / ref.width;
      if (frac < 0 || frac > 1) { cursor.hidden = tip.hidden = true; return; }
      const frame = Math.min(b.end - 1, Math.floor(b.start + frac * span));
      cursor.hidden = tip.hidden = false; cursor.style.left = (e.clientX - host.left) + 'px';
      tip.replaceChildren(el('b', null, mmss(b.clock0 + (frame - b.start) / b.fps)));
      for (const r of b.rows) { const l = at(r.segments, frame); const line = el('div'); if (l && l !== 'X') line.appendChild(el('i', 'tone-' + l)); line.appendChild(document.createTextNode(r.name + ': ' + (NAMES[l] || 'no label'))); tip.appendChild(line); }
      for (const lane of b.lanes || []) for (const s of lane.series) { const v = s.values[Math.min(s.values.length - 1, Math.floor((frame - b.start) / lane.step))]; tip.appendChild(el('div', null, (lane.series.length > 1 ? s.name : lane.name.split(':')[0]) + ': ' + (v === null || v === undefined ? 'no data' : lane.max === 1 && lane.min === 0 ? Math.round(v * 100) + '%' : v.toFixed(2) + (lane.unit ? ' ' + lane.unit : '')))); }
      const x = e.clientX - host.left + 14, w = tip.offsetWidth;
      tip.style.left = Math.max(0, x + w > host.width ? x - w - 28 : x) + 'px'; tip.style.top = Math.max(0, e.clientY - host.top - 20) + 'px';
    }
    box.addEventListener('pointermove', move); box.addEventListener('pointerdown', move);
    box.addEventListener('pointerleave', () => { cursor.hidden = tip.hidden = true; });
    return box;
  }

  // ------------------------------------------------------------------ turn matching: true possessions linked to predicted ones
  function turns(b) {
    const box = el('div', 'b-turns'), span = b.end - b.start, pct = (f) => (f - b.start) / span * 100;
    const g = svg('svg', { viewBox: '0 0 100 150', preserveAspectRatio: 'none', role: 'img', 'aria-label': 'True possessions linked to predicted possessions' });
    const rect = (a, e, y, cls, title) => { const r = svg('rect', { x: pct(a), width: Math.max(0.15, pct(e) - pct(a) - 0.12), y, height: 30, rx: 0.4, class: cls }); if (title) { const t = svg('title'); t.textContent = title; r.appendChild(t); } g.appendChild(r); return r; };
    for (const [a, e] of b.dead_truth) rect(a, e, 16, 'f-D', 'Dead ball');
    for (const [a, e] of b.dead_pred) rect(a, e, 104, 'f-D', 'Dead ball (predicted)');
    b.truth.forEach(([a, e, l, iou, pi], i) => {
      if (pi !== null && pi !== undefined) { const p = b.pred[pi]; g.appendChild(svg('polygon', { class: 'link f-' + l, points: [pct(a), 46, pct(e), 46, pct(p[1]), 104, pct(p[0]), 104].join(' ') })); }
      rect(a, e, 16, 'f-' + l, '#' + (i + 1) + ' ' + NAMES[l] + ' · ' + ((e - a) / b.fps).toFixed(1) + ' s · ' + (pi === null || pi === undefined ? 'missed' : 'overlap ' + iou.toFixed(2)));
      if (pi === null || pi === undefined) g.appendChild(svg('rect', { x: pct(a), width: Math.max(0.3, pct(e) - pct(a) - 0.12), y: 13, height: 36, class: 'miss' }));
    });
    for (const [a, e, l, used] of b.pred) rect(a, e, 104, 'f-' + l, NAMES[l] + ' (predicted) · ' + ((e - a) / b.fps).toFixed(1) + ' s' + (used ? '' : ' · matches no true possession'));
    const wrap = el('div'); wrap.style.position = 'relative';
    const top = el('p', 'b-title', 'The truth (dashed red = missed)'), bottom = el('p', 'b-title', 'Our possessions');
    wrap.appendChild(top); wrap.appendChild(g); bottom.style.marginTop = '4px'; wrap.appendChild(bottom);
    box.appendChild(wrap);
    box.appendChild(legend([['H', 'Barcelona'], ['A', 'Athletic'], ['D', 'Dead ball']]));
    return box;
  }

  // ------------------------------------------------------------------ the pitch, with a short replay
  function pitch(b) {
    const box = el('div', 'b-pitch'), L = b.length, W = b.width;
    const g = svg('svg', { viewBox: [-3, -5, L + 6, W + 8].join(' '), role: 'img', 'aria-label': 'Player positions on the pitch' });
    const ln = (tag, a) => g.appendChild(svg(tag, Object.assign({ class: 'ln' }, a)));
    ln('rect', { x: 0, y: 0, width: L, height: W }); ln('line', { x1: L / 2, x2: L / 2, y1: 0, y2: W }); ln('circle', { cx: L / 2, cy: W / 2, r: 9.15 });
    for (const s of [0, 1]) {
      const x = s ? L - 16.5 : 0, x6 = s ? L - 5.5 : 0;
      ln('rect', { x, y: W / 2 - 20.16, width: 16.5, height: 40.32 }); ln('rect', { x: x6, y: W / 2 - 9.16, width: 5.5, height: 18.32 });
      ln('rect', { x: s ? L : -2, y: W / 2 - 3.66, width: 2, height: 7.32 });
    }
    const arrow = svg('text', { class: 'arrow', x: L / 2, y: -1.6, 'text-anchor': 'middle' }); arrow.textContent = b.home_attacks === 'right' ? 'Barcelona attack →' : '← Barcelona attack'; g.appendChild(arrow);
    const layer = svg('g'); g.appendChild(layer);
    const bar = el('div', 'pitch-bar'), play = el('button', 'secondary'), slider = el('input'), tag = el('span', 'pitch-tag'), info = el('span', 'pitch-info');
    play.type = 'button'; slider.type = 'range'; slider.min = 0; slider.max = b.frames.length - 1; slider.step = 1; slider.value = b.centre; slider.setAttribute('aria-label', 'Moment in the replay');
    let timer = null;
    function draw(i) {
      const f = b.frames[i]; layer.replaceChildren();
      for (const [x, y, team] of f.players) layer.appendChild(svg('circle', { class: 'pl ' + (team === 0 ? 'f-H' : 'f-A'), cx: x, cy: y, r: 1.15 }));
      if (f.ball) layer.appendChild(svg('circle', { class: 'ball', cx: f.ball[0], cy: f.ball[1], r: 0.75 }));
      tag.className = 'pitch-tag tone-' + f.label + ' ' + f.label; tag.textContent = 'We say: ' + (NAMES[f.label] || 'not graded');
      info.textContent = f.t + ' · ball ' + (f.ball ? 'tracked' : 'NOT tracked') + ' · TV ' + (f.tv ? 'live shot' : 'away') + ' · players ' + (f.speed === null ? '–' : f.speed + ' m/s');
    }
    function stop() { clearInterval(timer); timer = null; play.replaceChildren(icon('play'), document.createTextNode('Play')); }
    play.addEventListener('click', () => {
      if (timer) return stop();
      if (Number(slider.value) >= b.frames.length - 1) slider.value = 0;
      play.replaceChildren(icon('pause'), document.createTextNode('Pause'));
      timer = setInterval(() => { if (!box.isConnected) return stop(); const n = Number(slider.value) + 1; if (n >= b.frames.length) return stop(); slider.value = n; draw(n); }, 200);
    });
    slider.addEventListener('input', () => draw(Number(slider.value)));
    stop(); draw(b.centre);
    bar.appendChild(play); bar.appendChild(slider); bar.appendChild(tag);
    if (b.frames.length < 2) { play.hidden = true; slider.hidden = true; }
    box.appendChild(g); box.appendChild(bar); box.appendChild(info);
    box.appendChild(legend([['H', 'Barcelona'], ['A', 'Athletic']]));
    return box;
  }

  // ------------------------------------------------------------------ assistant-designed static HTML, fully boxed in
  function frameCss() {
    const css = getComputedStyle(document.documentElement), v = (n) => css.getPropertyValue(n).trim();
    return ':root{--home:' + v('--vw-home') + ';--away:' + v('--vw-away') + ';--dead:' + v('--vw-dead') + ';--ink:' + v('--ink') + ';--muted:' + v('--muted') + ';--line:' + v('--line') + ';--surface:' + v('--surface') + ';--soft:' + v('--surface-soft') + ';--accent:' + v('--brand-bright') + '}' +
      '*{box-sizing:border-box}body{margin:0;padding:4px;background:var(--surface);color:var(--ink);font:13px/1.55 Poppins,Arial,Helvetica,sans-serif}h1,h2,h3{margin:0 0 8px;line-height:1.25}h1{font-size:20px}h2{font-size:16px}h3{font-size:14px}p{margin:0 0 8px}' +
      '.muted,.note{color:var(--muted);font-size:12px}.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:0 0 12px}.kpi{padding:10px 12px;border:1px solid var(--line);border-radius:12px;background:var(--soft)}.kpi b{display:block;font-size:24px;line-height:1.2}.kpi span{font-size:11px;color:var(--muted);font-weight:600}' +
      '.strip{display:flex;height:28px;border-radius:5px;overflow:hidden;background:var(--soft);margin:4px 0 10px}.strip span{display:block;height:100%}.strip .h,.h-bg{background:var(--home)}.strip .a,.a-bg{background:var(--away)}.strip .d,.d-bg{background:var(--dead)}' +
      '.bar-row{display:grid;grid-template-columns:34% 1fr 56px;gap:10px;align-items:center;margin:6px 0}.track{height:14px;background:var(--soft);border-radius:0 4px 4px 0}.fill{height:100%;background:var(--accent);border-radius:0 4px 4px 0}' +
      '.pill{display:inline-block;padding:2px 10px;border-radius:999px;background:var(--soft);font-size:11px;font-weight:600}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--line)}th{font-size:11px;color:var(--muted)}' +
      'svg{max-width:100%;height:auto}svg.pitch{background:#17603a;border-radius:10px}svg text{font-family:Poppins,Arial,sans-serif;fill:var(--ink)}';
  }
  function html(b) {
    const box = el('div');
    box.appendChild(el('span', 'vw-ai-label', 'AI-designed visual'));
    const clean = window.DOMPurify ? window.DOMPurify.sanitize(String(b.html), { FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form', 'input', 'button', 'link', 'meta', 'base', 'audio', 'video', 'img'], FORBID_ATTR: ['srcset', 'action', 'formaction', 'href', 'xlink:href', 'src'], WHOLE_DOCUMENT: false }) : '';
    const frame = el('iframe', 'vw-html-frame'); frame.setAttribute('sandbox', ''); frame.setAttribute('referrerpolicy', 'no-referrer'); frame.title = 'AI-designed visual'; frame.style.marginTop = '8px';
    frame.srcdoc = '<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src data:"><style>' + frameCss() + '</style></head><body>' + clean + '</body></html>';
    box.appendChild(frame);
    if (!window.DOMPurify) box.appendChild(el('p', 'b-note', 'The sanitiser did not load, so this visual was not drawn.'));
    return box;
  }

  const RENDER = { kpis, text: (b) => rich(b.text), timeline, bars, heatmap, table, checks, flow, turns, pitch, html };
  window.Blocks = {
    el, icon, mmss,
    render(block) { const fn = RENDER[block.type]; if (!fn) return el('p', 'b-note', 'This card has a part this page cannot draw (' + block.type + ').'); try { return fn(block); } catch (e) { return el('p', 'b-note', 'Could not draw this part: ' + e.message); } },
    wide(card) { return card.blocks.some((b) => ['timeline', 'turns', 'checks', 'flow'].includes(b.type)); },
    tall(card) { return card.blocks.some((b) => ['checks', 'turns'].includes(b.type)); },
  };
})();
