// The Flappy Fly: eye -> connectome -> flap, learning from its crashes.
//
// Every `decisionMs` the fly chooses between two actions: flap (a hard wing beat) or
// glide. Two readout channels, Q(flap) and Q(glide), read the normalized activity x of
// the brain's retinotopic visual populations and its MBONs (like competing MBON output
// channels). They learn by SARSA(lambda):
//
//   delta = r + gamma Q(s', a') - Q(s, a)        the dopamine signal
//
// delta is injected into the brain's PAM (delta > 0, reward) or PPL1 (delta < 0,
// punishment) dopamine neurons; their spikes gate plasticity at the connectome's
// KC -> MBON synapses (plasticity.js), and the same delta trains the readouts.
//
// Modes
//   bio     the connectome alone decides: flap when the DNg02 wing-power population
//           (Namiki et al. 2022) bursts above threshold; only KC->MBON synapses learn
//   hybrid  the learned readouts decide, biased by the DNg02 population. The readouts
//           stand in for the premotor circuits of the nerve cord, which this brain-only
//           model does not contain.
//
// Rewards: +1 for each pipe passed, -1 for a crash, plus an optional innate attraction
// to the opening (potential-based shaping on the distance to the gap, which provably
// leaves the optimal policy unchanged; Ng, Harada & Russell 1999). The teacher knows
// where the gap is; the fly only has its eyes.

import { Brain } from './engine.js';
import { Eye } from './eye.js';
import { Game } from './game.js';
import { buildPools, poolRates } from './pools.js';
import { MBPlasticity } from './plasticity.js';

export const AGENT_DEFAULTS = {
  mode: 'hybrid',          // 'bio' | 'hybrid'
  decisionMs: 40,
  gamma: 0.94,             // per decision
  lambda: 0.8,
  alpha: 0.1,              // normalized step size
  eps0: 0.1, eps1: 0.01, epsEpisodes: 300,
  exploreFlap: 0.15,       // flap probability of an exploratory action
  shaping: 1.0,            // innate attraction to the opening (0 = off)
  wBio: 0.05,              // DNg02 bias on the flap preference (hybrid)
  bioThreshold: -0.2,      // DNg02 z-score that triggers a flap (bio); calibrated to ~hovering flap rate
  rewardPass: 1.0,
  rewardCrash: -1.0,
  daGain: 14,              // mV of DAN drive per unit of |delta|
  daMs: 120,               // duration of the dopamine pulse
  normTau: 20000,          // ms, running feature normalization
  mbEta: 5e-7,
  featureMode: 'npca',     // 'raw' pools | 'npca' developmental components of retinotopic neurons
  pcaK: 128,
  efference: true,         // proprioceptive copy of the last wing beat
  effCenters: [0, 100, 200, 320, 480, 700],   // ms since last flap
  conj: true,              // conjunctions: visual components x motor state
  replaySize: 30000,       // transitions kept for replay between lives (0 = off)
  replayPerLife: 3000,     // replayed transitions after each crash
  replayAlpha: 0.03,
  seed: 1,
};

export class Agent {
  constructor(net, meta, gains, opts = {}) {
    this.o = { ...AGENT_DEFAULTS, ...opts, game: opts.game, eye: opts.eye };
    const o = this.o;
    this.meta = meta;
    this.brain = new Brain(net, meta, { seed: o.seed, ...(o.brain || {}) });
    if (gains) this.brain.gainIn.set(gains);
    this.mb = new MBPlasticity(this.brain, { eta: o.mbEta });
    this.brain.rebuildWeights(this.mb);
    this.eye = new Eye(meta, o.eye);
    this.rng = mulberry(o.seed * 7919 + 1);
    this.game = new Game(o.game, this.rng);
    // readout features: retinotopic visual pools (both eyes merged per region) + MBONs
    this.pools = buildPools(meta, {
      include: (i, cls) => cls === 'optic' || cls === 'vpn' || cls === 'mbon',
      mergeSides: true,
    });
    const keep = [];
    this.pools.info.forEach((p, k) => { if (p.region || p.cls === 'mbon') keep.push(k); });
    this.featIdx = Int32Array.from(keep);
    this.featInfo = keep.map(k => this.pools.info[k]);
    this.nRaw = keep.length + 1;            // + bias
    this.poolBuf = new Float32Array(this.pools.n);
    this.xRaw = new Float32Array(this.nRaw);
    this.mu = new Float32Array(this.nRaw);
    this.var = new Float32Array(this.nRaw).fill(25);
    this.pca = null;
    if (o.featureMode === 'npca') {
      if (!o.npca) throw new Error('featureMode npca needs opts.npca (web/data/npca_*.bin)');
      const K = Math.min(o.pcaK, o.npca.k);
      this.pca = { ...o.npca, K };
      this.pc = new Float32Array(K + 1);
      this.zbuf = new Float32Array(o.npca.n);
    }
    if (o.featureMode === 'truth') this.truthBuf = new Float32Array(4);   // diagnostic only
    const nVis = this.truthBuf ? 4 : this.pca ? this.pca.K + 1 : this.nRaw;
    const nEff = o.efference ? o.effCenters.length : 0;
    this.nVis = nVis; this.nEff = nEff;
    this.nFeat = o.efference && o.conj ? nVis * nEff : nVis + nEff;
    this.eff = new Float32Array(nEff);
    this.x = new Float32Array(this.nFeat);
    // replay memory: (vis, since, a, r, vis', since', terminal)
    this.rep = null;
    if (o.replaySize > 0 && this.pca) {
      const R = o.replaySize, nv = nVis;
      this.rep = { R, nv, n: 0, head: 0, vis: new Float32Array(R * nv), vis2: new Float32Array(R * nv),
        since: new Float32Array(R), since2: new Float32Array(R), a: new Uint8Array(R), r: new Float32Array(R), term: new Uint8Array(R) };
      this.xa = new Float32Array(this.nFeat); this.xb = new Float32Array(this.nFeat);
      this.prevVis = new Float32Array(nv); this.prevSince = 0;
    }
    this.replaying = 0;
    this.W = [new Float32Array(this.nFeat), new Float32Array(this.nFeat)];   // glide, flap
    this.E = [new Float32Array(this.nFeat), new Float32Array(this.nFeat)];
    this.flapDN = Int32Array.from(meta.roles.flapDN);
    this.dnMu = 5; this.dnVar = 25; this.zDN = 0; this.dnRate = 0;
    this.pam = Int32Array.from(meta.roles.pam);
    this.ppl1 = Int32Array.from(meta.roles.ppl1);
    this.daLeft = 0; this.daPam = 0; this.daPpl1 = 0;
    this.learning = true;
    this.explore = true;
    this.clock = 0;
    this.q = [0, 0]; this.delta = 0; this.lastAction = 0; this.explored = false;
    this.pendingReward = 0;
    this.episode = 0;
    this.history = [];
    this.brain.reset();
    this.eye.reset();
    this.hasPrev = false;
    this.onEpisodeEnd = null;
    this.game.flap();
    this.phiPrev = this.potential();
  }

  get epsilon() {
    const o = this.o;
    if (!this.explore) return 0;
    const f = Math.min(1, this.episode / o.epsEpisodes);
    return o.eps0 + (o.eps1 - o.eps0) * f;
  }

  potential() {
    const g = this.game;
    return -this.o.shaping * Math.abs(g.nextPipe().gapY - g.y) / 100;
  }

  features() {
    poolRates(this.pools, this.brain.rate, this.poolBuf);
    const { xRaw, mu, featIdx, poolBuf, o } = this;
    const a = o.decisionMs / o.normTau;
    const n = featIdx.length;
    for (let k = 0; k < n; k++) {
      const r = poolBuf[featIdx[k]];
      mu[k] += a * (r - mu[k]);
      const d = r - mu[k];
      this.var[k] += a * (d * d - this.var[k]);
      xRaw[k] = d / Math.sqrt(this.var[k] + 4);
    }
    xRaw[n] = 1;
    let vis = xRaw;
    if (this.pca) {
      const { K, n: nn, idx, mean, sd, comp, scale } = this.pca, pc = this.pc, z = this.zbuf, rate = this.brain.rate;
      for (let i = 0; i < nn; i++) z[i] = (rate[idx[i]] - mean[i]) / sd[i];
      for (let c = 0; c < K; c++) {
        let s = 0;
        const off = c * nn;
        for (let i = 0; i < nn; i++) s += comp[off + i] * z[i];
        const v = s / (scale[c] + 1e-6);
        pc[c] = v > 4 ? 4 : v < -4 ? -4 : v;
      }
      pc[K] = 1;
      vis = pc;
    }
    if (this.truthBuf) {   // diagnostic: game-state features instead of the brain
      const g = this.game, p = g.nextPipe(), t = this.truthBuf;
      const dy = (p.gapY - g.y) / 100, dx = (p.x - g.o.birdX) / 200;
      t[0] = dy; t[1] = dx; t[2] = dy * dx; t[3] = 1;
      vis = t;
    }
    const x = this.x, nv = this.nVis;
    if (o.efference) {
      const since = this.game.t - this.game.lastFlap, eff = this.eff, cs = o.effCenters;
      for (let j = 0; j < cs.length; j++) { const d = (since - cs[j]) / 90; eff[j] = Math.exp(-d * d); }
      if (o.conj) {
        let k = 0;
        for (let j = 0; j < cs.length; j++) { const e = eff[j]; for (let v = 0; v < nv; v++) x[k++] = vis[v] * e; }
      } else {
        x.set(vis.subarray(0, nv), 0); x.set(eff, nv);
      }
    } else if (vis !== x) x.set(vis.subarray(0, nv), 0);
    let s = 0;
    for (const i of this.flapDN) s += this.brain.rate[i];
    s /= this.flapDN.length;
    this.dnMu += a * (s - this.dnMu);
    this.dnVar += a * ((s - this.dnMu) ** 2 - this.dnVar);
    this.zDN = (s - this.dnMu) / Math.sqrt(this.dnVar + 1);
    this.dnRate = s;
    return x;
  }

  // conjunctive features from stored visual components and time since the last flap
  _buildX(out, vis, since) {
    const o = this.o, nv = this.nVis, cs = o.effCenters;
    if (!o.efference) { out.set(vis.subarray(0, nv)); return out; }
    if (o.conj) {
      let k = 0;
      for (let j = 0; j < cs.length; j++) { const d = (since - cs[j]) / 90, e = Math.exp(-d * d); for (let v = 0; v < nv; v++) out[k++] = vis[v] * e; }
    } else {
      out.set(vis.subarray(0, nv)); for (let j = 0; j < cs.length; j++) { const d = (since - cs[j]) / 90; out[nv + j] = Math.exp(-d * d); }
    }
    return out;
  }

  _store(r, terminal, vis2, since2) {
    const R = this.rep; if (!R || !this.hasPrev) return;
    const h = R.head, nv = R.nv;
    R.vis.set(this.prevVis, h * nv); R.since[h] = this.prevSince; R.a[h] = this.lastAction; R.r[h] = r; R.term[h] = terminal ? 1 : 0;
    if (!terminal) { R.vis2.set(vis2.subarray(0, nv), h * nv); R.since2[h] = since2; }
    R.head = (h + 1) % R.R; R.n = Math.min(R.n + 1, R.R);
  }

  // Offline consolidation between lives: one-step Q-learning on remembered transitions.
  replay(count = this.o.replayPerLife) {
    const R = this.rep; if (!R || R.n < 500 || !this.learning || this.o.mode !== 'hybrid') return;
    const o = this.o, nv = R.nv, xa = this.xa, xb = this.xb, [w0, w1] = this.W, nF = this.nFeat;
    for (let c = 0; c < count; c++) {
      const j = (this.rng() * R.n) | 0;
      this._buildX(xa, R.vis.subarray(j * nv, (j + 1) * nv), R.since[j]);
      let target = R.r[j];
      if (!R.term[j]) {
        this._buildX(xb, R.vis2.subarray(j * nv, (j + 1) * nv), R.since2[j]);
        let q0 = 0, q1 = 0;
        for (let k = 0; k < nF; k++) { q0 += w0[k] * xb[k]; q1 += w1[k] * xb[k]; }
        target += o.gamma * Math.max(q0, q1);
      }
      const w = R.a[j] ? w1 : w0;
      let q = 0, nx = 0;
      for (let k = 0; k < nF; k++) { q += w[k] * xa[k]; nx += xa[k] * xa[k]; }
      const d = Math.max(-2, Math.min(2, target - q)), st = o.replayAlpha / (nx + 1);
      for (let k = 0; k < nF; k++) w[k] += st * d * xa[k];
    }
    this.replaying = count;
  }

  qValues(x) {
    let q0 = 0, q1 = 0;
    const [w0, w1] = this.W;
    for (let k = 0; k < x.length; k++) { q0 += w0[k] * x[k]; q1 += w1[k] * x[k]; }
    this.q[0] = q0; this.q[1] = q1;
    return this.q;
  }

  choose(x) {
    const o = this.o;
    this.explored = false;
    if (o.mode === 'bio') return this.zDN > o.bioThreshold ? 1 : 0;
    const [q0, q1] = this.qValues(x);
    if (this.rng() < this.epsilon) { this.explored = true; return this.rng() < o.exploreFlap ? 1 : 0; }
    return q1 + o.wBio * this.zDN > q0 ? 1 : 0;
  }

  dopamine(delta) {
    this.delta = delta;
    this.daLeft = this.o.daMs;
    this.daPam = Math.min(30, Math.max(0, delta) * this.o.daGain);
    this.daPpl1 = Math.min(30, Math.max(0, -delta) * this.o.daGain);
  }

  // One millisecond of the world.
  step() {
    const g = this.game, br = this.brain, o = this.o;
    const ev = g.step(1);
    for (const e of ev) {
      if (e === 'pass') this.pendingReward += o.rewardPass;
      if (e === 'crash') this.pendingReward += o.rewardCrash;
    }
    this.eye.update(g, br, 1);
    const ext = br.ext;
    if (this.daLeft > 0) {
      this.daLeft -= 1;
      const on = this.daLeft > 0;
      for (const i of this.pam) ext[i] = on ? this.daPam : 0;
      for (const i of this.ppl1) ext[i] = on ? this.daPpl1 : 0;
    }
    br.step();
    this.mb.tick();
    this.clock += 1;
    this.lastEvents = ev;
    if (g.dead) { this._decide(true); this._endEpisode(); return; }
    if (this.clock % o.decisionMs === 0) this._decide(false);
  }

  _decide(terminal) {
    const o = this.o;
    const x = this.features();
    const phiN = terminal ? 0 : this.potential();
    const r = this.pendingReward + o.gamma * phiN - this.phiPrev;
    this.phiPrev = phiN;
    this.pendingReward = 0;
    const since = this.game.t - this.game.lastFlap;
    this._store(r, terminal, this.pc, since);
    const a = terminal ? 0 : this.choose(x);
    if (o.mode === 'bio') this.qValues(x);
    const Qn = terminal ? 0 : this.q[a];
    if (this.hasPrev) {
      const delta = r + o.gamma * Qn - this.Qprev;
      this.dopamine(delta);
      if (this.learning && o.mode === 'hybrid') this._learn(delta);
    }
    if (terminal) { this.hasPrev = false; return; }
    const gl = o.gamma * o.lambda;
    for (let b = 0; b < 2; b++) {
      const e = this.E[b];
      if (b === a) for (let k = 0; k < x.length; k++) e[k] = gl * e[k] + x[k];
      else for (let k = 0; k < x.length; k++) e[k] *= gl;
    }
    if (a) this.game.flap();
    if (this.rep) { this.prevVis.set(this.pc.subarray(0, this.nVis)); this.prevSince = since; }
    this.Qprev = this.q[a];
    this.lastAction = a;
    this.hasPrev = true;
  }

  _learn(delta) {
    const x = this.x;
    let nx = 0;
    for (let k = 0; k < x.length; k++) nx += x[k] * x[k];
    const step = this.o.alpha / (nx + 1);
    const d = Math.max(-2, Math.min(2, delta));
    for (let b = 0; b < 2; b++) {
      const w = this.W[b], e = this.E[b];
      for (let k = 0; k < w.length; k++) w[k] += step * d * e[k];
    }
  }

  _endEpisode() {
    const g = this.game;
    this.history.push({ ep: this.episode, score: g.score, t: g.t, cause: g.deathCause, flaps: g.flaps });
    this.episode++;
    const h = this.history[this.history.length - 1];
    this.E[0].fill(0); this.E[1].fill(0);
    this.replay();
    this.mb.reset();
    g.reset();
    g.flap();               // every game starts with a tap
    this.eye.reset();
    this.hasPrev = false;
    this.phiPrev = this.potential();
    if (this.onEpisodeEnd) this.onEpisodeEnd(h);
  }

  snapshot() {
    return {
      version: 3, mode: this.o.mode, featureMode: this.o.featureMode, npcaName: this.o.npcaName ?? 'lam', pcaK: this.pca ? this.pca.K : 0, efference: this.o.efference, conj: this.o.conj,
      W0: Array.from(this.W[0]), W1: Array.from(this.W[1]),
      mu: Array.from(this.mu), var: Array.from(this.var), dnMu: this.dnMu, dnVar: this.dnVar,
      mbF: Array.from(this.mb.f), episode: this.episode, history: this.history,
      featKeys: this.featInfo.map(f => f.key),
    };
  }

  load(s) {
    if (s.version !== 3 || s.W0.length !== this.nFeat) throw new Error('snapshot does not match this brain');
    this.W[0].set(s.W0); this.W[1].set(s.W1);
    this.mu.set(s.mu); this.var.set(s.var); this.dnMu = s.dnMu; this.dnVar = s.dnVar;
    this.mb.f.set(s.mbF); this.mb.apply(this.brain);
    this.episode = s.episode; this.history = s.history || [];
  }
}

function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
