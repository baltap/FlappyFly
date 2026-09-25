import { BrainView, CLASS_STYLE } from './brainview.js';
import { GameView, EyeView } from './gameview.js';

const $ = (s) => document.querySelector(s);
const worker = new Worker('js/worker.js', { type: 'module' });
const state = {
  ready: false, running: false, frame: null, history: [], best: 0, daTrace: [], lastT: performance.now(),
  roles: null, trained: true, hasPretrained: false,
};
let brainView, gameView, eyeView;

function toast(t) { const el = $('#toast'); el.textContent = t; el.classList.add('show'); clearTimeout(toast.h); toast.h = setTimeout(() => el.classList.remove('show'), 2600); }

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'status') $('#stream-status').textContent = m.text;
  else if (m.type === 'error') { console.error(m.text); $('#stream-status').textContent = 'Error: ' + m.text.split('\n')[0]; }
  else if (m.type === 'ready') onReady(m);
  else if (m.type === 'frame') onFrame(m);
  else if (m.type === 'rebuilt') { state.history = m.history.slice(); state.best = Math.max(0, ...state.history.map(h => h.score)); refreshHud(); }
};

function onReady(m) {
  state.roles = m.roles;
  state.hasPretrained = m.hasPretrained;
  brainView = new BrainView($('#brain'), m);
  gameView = new GameView($('#game'));
  eyeView = new EyeView($('#eye'), m.dirs, m.dirSide);
  const legend = $('#legend');
  const keys = ['eye', 'optic', 'vpn', 'central', 'kc', 'mbon', 'pam', 'ppl1', 'dn', 'flap'];
  legend.innerHTML = keys.map(k => { const c = CLASS_STYLE[k].col.map(v => Math.round(v * 255)); return `<li><i style="background:rgb(${c})"></i>${CLASS_STYLE[k].label}</li>`; }).join('');
  if (!m.hasPretrained) { state.trained = false; $('#fly-trained').disabled = true; $('#fly-trained').title = 'No trained fly yet (run sim/train.mjs)'; setFly(false, true); }
  else setFly(true, true);
  state.ready = true;
  $('#play').disabled = false;
  setRunning(true);
  $('#stream-status').textContent = `${m.n.toLocaleString()} neurons`;
  requestAnimationFrame(draw);
}

function setRunning(on) {
  state.running = on;
  $('#play').textContent = on ? 'Pause' : 'Play';
  worker.postMessage({ type: 'run', running: on });
}

function setFly(trained, silent) {
  state.trained = trained;
  $('#fly-trained').classList.toggle('on', trained);
  $('#fly-newborn').classList.toggle('on', !trained);
  state.history = []; state.best = 0; state.daTrace = [];
  worker.postMessage({ type: 'rebuild', pretrained: trained, mode: $('#mode').value, shaping: $('#shaping').checked ? 1 : 0, difficulty: $('#difficulty').value });
  // the trained fly keeps its memories frozen by default (continued learning can unsettle it)
  $('#learn').checked = !trained;
  $('#explore').checked = !trained;
  worker.postMessage({ type: 'learning', on: !trained });
  worker.postMessage({ type: 'explore', on: !trained });
  if (!silent) toast(trained ? 'A trained fly: it has already learned from its crashes.' : 'A newborn fly: every crash teaches it something.');
  refreshHud();
}

function onFrame(f) {
  state.frame = f;
  for (const h of f.newEpisodes) { state.history.push(h); state.best = Math.max(state.best, h.score); }
  if (f.newEpisodes.length) refreshHud();
  if (brainView) brainView.update(f.act, Math.max(16, performance.now() - state.lastT));
  state.lastT = performance.now();
  for (const ev of f.events) {
    if (ev === 'crash') { gameView.flash = 1; brainView.highlight(state.roles.ppl1, 1); ring('pun'); }
    if (ev === 'pass') { gameView.passGlow = 1; brainView.highlight(state.roles.pam, 0.9); ring('rew'); }
  }
  if (f.daPam > 2) brainView.highlight(state.roles.pam, Math.min(1, f.daPam / 20));
  if (f.daPpl1 > 2) brainView.highlight(state.roles.ppl1, Math.min(1, f.daPpl1 / 20));
  if (f.action && performance.now() - (state.lastFlapHi || 0) > 60) { brainView.highlight(state.roles.flapDN, 1); state.lastFlapHi = performance.now(); }
  if (f.replayed) { const b = $('#replay-badge'); b.textContent = `replaying ${f.replayed.toLocaleString()} memories`; b.hidden = false; clearTimeout(state.rh); state.rh = setTimeout(() => { b.hidden = true; }, 1400); }
  state.daTrace.push(f.delta);
  if (state.daTrace.length > 240) state.daTrace.shift();
}

function ring(kind) {
  const r = $('#da-ring');
  r.classList.remove('rew', 'pun');
  void r.offsetWidth;
  r.classList.add(kind);
  clearTimeout(ring.h); ring.h = setTimeout(() => r.classList.remove(kind), 180);
}

function refreshHud() {
  const h = state.history;
  $('#life').textContent = h.length + 1;
  $('#best').textContent = state.best;
  const last = h.slice(-20);
  $('#avg').textContent = last.length ? (last.reduce((s, x) => s + x.score, 0) / last.length).toFixed(1) : '0.0';
  drawCurve();
}

function draw() {
  requestAnimationFrame(draw);
  const f = state.frame;
  if (!f) return;
  gameView.draw(f.game, performance.now());
  brainView.draw();
  eyeView.draw(f.eye);
  // decision bars: Q values around the centre, DNg02 rate from the left
  const qs = f.q, span = Math.max(0.5, Math.abs(qs[0]), Math.abs(qs[1]));
  for (const [i, id] of [[1, 'q1'], [0, 'q0']]) {
    const v = qs[i] / span, el = $('#' + id);
    el.style.left = v >= 0 ? '50%' : `${50 + v * 50}%`;
    el.style.width = `${Math.abs(v) * 50}%`;
    $('#' + id + 'v').textContent = qs[i].toFixed(2);
  }
  $('#q1').parentElement.parentElement.classList.toggle('active', f.action === 1);
  $('#q0').parentElement.parentElement.classList.toggle('active', f.action === 0);
  $('#dn').style.width = `${Math.min(100, f.dnRate * 2)}%`;
  $('#dnv').textContent = `${f.dnRate.toFixed(0)} Hz`;
  $('#explore-badge').hidden = !f.explored;
  $('#stream-status').textContent = `${(f.rate || 0).toFixed(1)}× real time · ε ${f.eps.toFixed(3)} · ${f.mode}`;
  drawDA();
  drawMB(f.mbon);
}

function fitCanvas(cv) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  return [ctx, w, h];
}

function drawDA() {
  const [ctx, w, h] = fitCanvas($('#da'));
  const tr = state.daTrace, mid = h / 2;
  ctx.strokeStyle = '#232838'; ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
  const n = 240, bw = w / n;
  for (let i = 0; i < tr.length; i++) {
    const v = Math.max(-1.5, Math.min(1.5, tr[i])) / 1.5;
    ctx.fillStyle = v >= 0 ? '#3ddc97' : '#ff5d73';
    const x = (n - tr.length + i) * bw;
    ctx.fillRect(x, v >= 0 ? mid - v * (mid - 2) : mid, Math.max(1, bw - 0.5), Math.abs(v) * (mid - 2));
  }
}

function drawMB(mbon) {
  const [ctx, w, h] = fitCanvas($('#mb'));
  const n = mbon.length, cols = Math.ceil(n / 3), cw = w / cols, ch = h / 3;
  for (let i = 0; i < n; i++) {
    const d = Math.max(-1, Math.min(1, (mbon[i] - 1) * 4));
    const c = d >= 0 ? [61, 220, 151] : [255, 93, 115];
    const a = Math.abs(d);
    ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.12 + 0.88 * a})`;
    ctx.fillRect((i % cols) * cw + 1, Math.floor(i / cols) * ch + 1, cw - 2, ch - 2);
  }
}

function drawCurve() {
  const cv = $('#curve');
  const [ctx, w, h] = fitCanvas(cv);
  const hist = state.history;
  const pad = { l: 34, r: 10, t: 10, b: 22 };
  const maxS = Math.max(5, ...hist.map(x => x.score));
  const N = Math.max(30, hist.length);
  const X = (i) => pad.l + (i + 0.5) / N * (w - pad.l - pad.r);
  const Y = (s) => h - pad.b - s / maxS * (h - pad.t - pad.b);
  ctx.fillStyle = '#8a90a6'; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'right';
  for (const s of [0, Math.round(maxS / 2), maxS]) { ctx.fillText(String(s), pad.l - 6, Y(s) + 4); ctx.strokeStyle = '#1c2130'; ctx.beginPath(); ctx.moveTo(pad.l, Y(s)); ctx.lineTo(w - pad.r, Y(s)); ctx.stroke(); }
  ctx.textAlign = 'center'; ctx.fillText('life →', w / 2, h - 5);
  for (let i = 0; i < hist.length; i++) {
    const s = hist[i].score;
    ctx.fillStyle = s > 0 ? 'rgba(255,209,102,0.8)' : 'rgba(255,93,115,0.45)';
    ctx.beginPath(); ctx.arc(X(i), Y(s), s > 0 ? 2.4 : 1.6, 0, Math.PI * 2); ctx.fill();
  }
  if (hist.length > 1) {
    ctx.strokeStyle = '#ffd166'; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i < hist.length; i++) {
      const a = hist.slice(Math.max(0, i - 19), i + 1);
      const m = a.reduce((s, x) => s + x.score, 0) / a.length;
      i ? ctx.lineTo(X(i), Y(m)) : ctx.moveTo(X(i), Y(m));
    }
    ctx.stroke(); ctx.lineWidth = 1;
  }
  if (!hist.length) { ctx.fillStyle = '#8a90a6'; ctx.fillText('Every dot will be one life. Watch the line climb as the fly learns.', w / 2, h / 2); }
}

// controls
$('#play').onclick = () => setRunning(!state.running);
document.querySelectorAll('[data-speed]').forEach(b => b.onclick = () => {
  document.querySelectorAll('[data-speed]').forEach(x => x.classList.toggle('on', x === b));
  worker.postMessage({ type: 'speed', speed: +b.dataset.speed });
});
$('#fly-trained').onclick = () => setFly(true);
$('#fly-newborn').onclick = () => setFly(false);
$('#learn').onchange = (e) => worker.postMessage({ type: 'learning', on: e.target.checked });
$('#explore').onchange = (e) => worker.postMessage({ type: 'explore', on: e.target.checked });
$('#shaping').onchange = (e) => worker.postMessage({ type: 'shaping', value: e.target.checked ? 1 : 0 });
$('#mode').onchange = (e) => { worker.postMessage({ type: 'mode', mode: e.target.value }); toast(e.target.value === 'bio' ? 'Biology only: the DNg02 neurons decide alone.' : 'Hybrid: the learned readout decides, nudged by DNg02.'); };
$('#difficulty').onchange = (e) => { worker.postMessage({ type: 'difficulty', value: e.target.value }); toast(e.target.value === 'original' ? 'Original pipes: much harder than what the fly trained on.' : 'Training pipes: 150 px gaps.'); };
$('#help-flap').onclick = () => worker.postMessage({ type: 'flap' });
window.addEventListener('keydown', (e) => { if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); worker.postMessage({ type: 'flap' }); } });
window.addEventListener('resize', () => state.ready && drawCurve());

worker.postMessage({ type: 'init' });
