// Simulation worker: owns the fly (brain + eye + game + learning) and streams frames.
import { parseNet } from './engine.js';
import { Agent } from './agent.js';
import { parseNpca } from './npca.js';

const DIFFICULTY = { training: { gap: 150, maxGapShift: 60 }, original: { gap: 100, maxGapShift: 140 } };
const D = (f) => new URL(`../data/${f}`, import.meta.url).href;
let agent = null;
let meta = null;
let running = false;
let speed = 1;             // 1 = real time, 0 = as fast as possible
let spikeCount = null;
let lastPost = 0;
let simMsSincePost = 0;
let newEpisodes = [];
let events = [];
let pendingFlap = false;
let rateWin = { sim: 0, wall: 0, t0: performance.now() };
let base = { net: null, gains: null, npca: {} };
let pretrained = null;

async function fetchBuf(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.arrayBuffer();
}

async function init(opts) {
  postMessage({ type: 'status', text: 'Loading the connectome (80k neurons, 4.3M synapses)…' });
  const [bin, js, gains] = await Promise.all([
    fetchBuf(D('brain.bin')), fetch(D('brain.json')).then(r => r.json()), fetchBuf(D('gains.f32')),
  ]);
  meta = js;
  base.net = parseNet(bin);
  base.gains = new Float32Array(gains);
  try { pretrained = await fetch(D('pretrained.json')).then(r => (r.ok ? r.json() : null)); } catch { pretrained = null; }
  const npcaName = pretrained?.npcaName ?? 'lam';
  base.npca[npcaName] = parseNpca(await fetchBuf(D(`npca_${npcaName}.bin`)));
  base.npcaName = npcaName;
  build(opts);
  postMessage({
    type: 'ready', n: meta.n, pos: meta.pos, cls: meta.class, roles: meta.roles, nFeat: agent.nFeat,
    dirs: agent.eye.dirs, dirSide: agent.eye.dirSide, hasPretrained: !!pretrained, pretrainedEpisodes: pretrained ? pretrained.episode : 0,
    featInfo: agent.featInfo, typeOf: meta.type, types: meta.types,
  });
}

function build(opts = {}) {
  // a newborn fly gets the same readout layout as the trained one, with blank weights
  const cfg = pretrained ? { pcaK: pretrained.pcaK, efference: pretrained.efference, conj: pretrained.conj } : {};
  agent = new Agent(base.net, meta, base.gains, { mode: opts.mode ?? 'hybrid', seed: opts.seed ?? (Math.random() * 1e6 | 0),
    shaping: opts.shaping ?? 1, npca: base.npca[base.npcaName], game: DIFFICULTY[opts.difficulty ?? 'training'], ...cfg });
  if (opts.pretrained && pretrained) agent.load(pretrained);
  spikeCount = new Uint8Array(agent.brain.n);
  agent.onEpisodeEnd = (h) => newEpisodes.push(h);
}

function simulate(ms) {
  for (let i = 0; i < ms; i++) {
    if (pendingFlap) { agent.game.flap(); pendingFlap = false; }
    agent.step();
    const b = agent.brain, sl = b.spikeList;
    for (let s = 0; s < b.nSpikes; s++) { const k = sl[s]; if (spikeCount[k] < 255) spikeCount[k]++; }
    if (agent.lastEvents && agent.lastEvents.length) for (const e of agent.lastEvents) events.push(e);
    if (agent.game.t === 1) events.push('start');
  }
  simMsSincePost += ms;
}

function post() {
  const g = agent.game;
  const act = spikeCount;
  spikeCount = new Uint8Array(agent.brain.n);
  const mbon = new Float32Array(agent.meta.roles.mbon.length);
  const cnt = new Float32Array(mbon.length);
  const f = agent.mb.f, post = agent.mb.post;
  for (let j = 0; j < f.length; j++) { mbon[post[j]] += f[j]; cnt[post[j]]++; }
  for (let m = 0; m < mbon.length; m++) mbon[m] = cnt[m] ? mbon[m] / cnt[m] : 1;
  const frame = {
    type: 'frame',
    game: { y: g.y, vy: g.vy, score: g.score, t: g.t, dead: g.dead, lastFlap: g.lastFlap, scroll: g.scroll,
      pipes: g.pipes.map(p => [p.x, p.gapY]), o: g.o },
    act, simMs: simMsSincePost,
    eye: Float32Array.from(agent.eye.P),
    q: [agent.q[0], agent.q[1]], delta: agent.delta, action: agent.lastAction, explored: agent.explored,
    eps: agent.epsilon, zDN: agent.zDN, dnRate: agent.dnRate, mode: agent.o.mode,
    daPam: agent.daLeft > 0 ? agent.daPam : 0, daPpl1: agent.daLeft > 0 ? agent.daPpl1 : 0,
    mbon, episode: agent.episode, learning: agent.learning,
    newEpisodes, events, rate: rateWin.rate || 0, replayed: agent.replaying,
  };
  newEpisodes = []; events = []; agent.replaying = 0;
  simMsSincePost = 0;
  postMessage(frame, [act.buffer]);
}

let lastTick = performance.now();
function loop() {
  if (!running || !agent) return;
  const now = performance.now();
  const wall = Math.min(100, now - lastTick);
  lastTick = now;
  // Real time: simulate as many ms as wall time passed (capped by compute). Fast: fill ~28 ms.
  const budgetEnd = now + (speed === 0 ? 28 : 24);
  let target = speed === 0 ? Infinity : Math.round(wall * speed);
  let done = 0;
  while (done < target && performance.now() < budgetEnd) { simulate(5); done += 5; }
  rateWin.sim += done; rateWin.wall += wall;
  if (rateWin.wall > 1000) { rateWin.rate = rateWin.sim / rateWin.wall; rateWin.sim = 0; rateWin.wall = 0; }
  if (now - lastPost > 33) { post(); lastPost = now; }
  setTimeout(loop, 0);
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') { await init(m); }
    else if (m.type === 'run') { running = m.running; lastTick = performance.now(); if (running) loop(); }
    else if (m.type === 'speed') { speed = m.speed; }
    else if (m.type === 'learning') { agent.learning = m.on; agent.mb.enabled = m.on; }
    else if (m.type === 'explore') { agent.explore = m.on; }
    else if (m.type === 'shaping') { agent.o.shaping = m.value; }
    else if (m.type === 'mode') { agent.o.mode = m.mode; }
    else if (m.type === 'difficulty') { Object.assign(agent.game.o, DIFFICULTY[m.value]); agent.game.reset(); agent.game.flap(); }
    else if (m.type === 'rebuild') { const was = running; running = false; build(m); postMessage({ type: 'rebuilt', episode: agent.episode, history: agent.history }); running = was; if (running) { lastTick = performance.now(); loop(); } }
    else if (m.type === 'flap') { pendingFlap = true; }
    else if (m.type === 'snapshot') { postMessage({ type: 'snapshot', data: agent.snapshot() }); }
  } catch (err) {
    postMessage({ type: 'error', text: String(err && err.stack || err) });
  }
};
