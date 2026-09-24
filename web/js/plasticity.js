// Dopamine-gated three-factor plasticity at Kenyon cell -> MBON synapses.
//
//   eligibility  E_e  <- decays with tauE; on a KC spike: E_e += (r_MBON - rbar_MBON)
//   dopamine     DA_m  = sum_d C[d,m] (r_d - rbar_d) s_d / sum_d C[d,m]   (Hz; s = +1 PAM, -1 PPL1)
//   update       f_e  += eta * DA_m * E_e * dt,  f clipped to [fMin, fMax]
//   recovery     f_e relaxes to 1 with time constant tauRecover (forgetting)
//
// C is the connectome's DAN -> MBON compartment map (shared-KC overlap, see pipeline).
// f multiplies the synapse's base weight. This is node-perturbation learning: MBON
// fluctuations that preceded a better-than-expected outcome are reinforced.

export class MBPlasticity {
  constructor(brain, opts = {}) {
    const meta = brain.meta;
    this.b = brain;
    this.eta = opts.eta ?? 5e-7;
    this.tauRecover = opts.tauRecover ?? 120000;
    this.tauE = opts.tauE ?? 800;
    this.fMin = opts.fMin ?? 0.0;
    this.fMax = opts.fMax ?? 4.0;
    const isMbon = new Int32Array(brain.n).fill(-1);
    meta.roles.mbon.forEach((i, k) => { isMbon[i] = k; });
    const kcSet = new Uint8Array(brain.n);
    for (const i of meta.roles.kc) kcSet[i] = 1;
    // collect KC->MBON edges
    const edges = [], pre = [], post = [];
    for (const k of meta.roles.kc) {
      for (let e = brain.indptr[k]; e < brain.indptr[k + 1]; e++) {
        const m = isMbon[brain.indices[e]];
        if (m >= 0) { edges.push(e); pre.push(k); post.push(m); }
      }
    }
    this.edge = Int32Array.from(edges);
    this.pre = Int32Array.from(pre);
    this.post = Int32Array.from(post);
    this.f = new Float32Array(edges.length).fill(1);
    this.base = new Float32Array(edges.length);
    this.E = new Float32Array(edges.length);
    // edges grouped by presynaptic KC for spike-driven eligibility
    this.byPre = new Map();
    this.pre.forEach((k, j) => { if (!this.byPre.has(k)) this.byPre.set(k, []); this.byPre.get(k).push(j); });
    this.byPreArr = new Array(brain.n);
    for (const [k, js] of this.byPre) this.byPreArr[k] = Int32Array.from(js);
    this.mbon = Int32Array.from(meta.roles.mbon);
    this.dan = Int32Array.from(meta.danOrder);
    this.danSign = new Float32Array(this.dan.length);
    const pam = new Set(meta.roles.pam);
    this.dan.forEach((d, j) => { this.danSign[j] = pam.has(d) ? 1 : -1; });
    this.C = meta.danMbon;                 // [dan][mbon]
    this.Csum = new Float32Array(this.mbon.length);
    for (const row of this.C) row.forEach((c, m) => { this.Csum[m] += c; });
    this.rbarM = new Float32Array(this.mbon.length).fill(5);
    this.rbarD = new Float32Array(this.dan.length).fill(3);
    this.DA = new Float32Array(this.mbon.length);
    this.enabled = true;
    this.captureBase();
  }

  captureBase() {
    for (let j = 0; j < this.edge.length; j++) this.base[j] = this.b.w[this.edge[j]];
  }

  apply(brain) {  // after a weight rebuild
    this.captureBase();
    for (let j = 0; j < this.edge.length; j++) brain.w[this.edge[j]] = this.base[j] * this.f[j];
  }

  reset() { this.E.fill(0); }

  // once per brain step
  tick() {
    const b = this.b, dt = b.p.dt, rate = b.rate;
    const aE = Math.exp(-dt / this.tauE), aBar = dt / 2000;
    const { E, byPreArr, post, rbarM, rbarD, mbon, dan } = this;
    for (let j = 0; j < E.length; j++) E[j] *= aE;
    for (let m = 0; m < mbon.length; m++) rbarM[m] += aBar * (rate[mbon[m]] - rbarM[m]);
    for (let d = 0; d < dan.length; d++) rbarD[d] += aBar * (rate[dan[d]] - rbarD[d]);
    const sl = b.spikeList;
    for (let s = 0; s < b.nSpikes; s++) {
      const js = byPreArr[sl[s]];
      if (!js) continue;
      for (let q = 0; q < js.length; q++) {
        const j = js[q], m = post[j];
        E[j] += (rate[mbon[m]] - rbarM[m]) * 0.05;
      }
    }
    if (!this.enabled) return;
    // dopamine per MBON compartment
    const DA = this.DA, C = this.C;
    DA.fill(0);
    for (let d = 0; d < dan.length; d++) {
      const x = (rate[dan[d]] - rbarD[d] - 2) * this.danSign[d];   // only phasic bursts count
      if (x === 0 || rate[dan[d]] - rbarD[d] < 2) continue;
      const row = C[d];
      for (let m = 0; m < row.length; m++) if (row[m]) DA[m] += row[m] * x;
    }
    for (let m = 0; m < DA.length; m++) DA[m] = this.Csum[m] > 0 ? DA[m] / this.Csum[m] : 0;
    const eta = this.eta * dt, w = b.w, { edge, base, f, fMin, fMax } = this;
    const rec = dt / this.tauRecover;
    for (let j = 0; j < E.length; j++) {
      const da = DA[post[j]];
      let x = f[j] + rec * (1 - f[j]);
      if (da !== 0) x += eta * da * E[j];
      x = x < fMin ? fMin : x > fMax ? fMax : x;
      f[j] = x;
      w[edge[j]] = base[j] * x;
    }
  }

  stats() {
    let s = 0, s2 = 0;
    for (const x of this.f) { s += x; s2 += x * x; }
    const n = this.f.length;
    return { mean: s / n, sd: Math.sqrt(Math.max(0, s2 / n - (s / n) ** 2)) };
  }
}
