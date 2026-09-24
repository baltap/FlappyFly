// Leaky integrate-and-fire network on the connectome (Shiu et al. 2024 parameters),
// event-driven spike propagation over a CSR synapse table. Runs in the browser and Node.
//
//   dv/dt = (g - (v - vRest)) / tauM        dg/dt = -g / tauSyn
//   presynaptic spike: g_post += wSyn * count * sign * gain_post * plastic
//
// Every neuron has an intrinsic excitability offset (threshold shift) that a slow
// homeostatic rule sets during development (see calibrate()) and then freezes.

export const DEFAULTS = {
  dt: 1.0,            // ms
  vRest: -52, vReset: -52, vTh: -45,
  tauM: 20, tauSyn: 5, tRef: 2.2,
  wSyn: 0.275,        // mV per synapse
  noise: 0.5,         // mV/sqrt(ms) membrane noise (sd ~1.6 mV)
  maxRateHz: 400,
};

export class Brain {
  constructor(net, meta, params = {}) {
    this.p = { ...DEFAULTS, ...params };
    this.meta = meta;
    this.n = net.n;
    this.indptr = net.indptr;
    this.indices = net.indices;
    this.count = net.weight;                 // signed synapse counts (Int16)
    const n = this.n, m = this.indices.length;
    this.w = new Float32Array(m);            // effective mV per spike
    this.gainIn = new Float32Array(n).fill(1);
    this.v = new Float32Array(n).fill(this.p.vRest);
    this.g = new Float32Array(n);
    this.ext = new Float32Array(n);          // external drive, mV (steady-state depolarization)
    this.thOff = new Float32Array(n);        // homeostatic threshold offset
    this.ref = new Float32Array(n);
    this.spiked = new Uint8Array(n);
    this.spikeList = new Int32Array(n);
    this.nSpikes = 0;
    this.rate = new Float32Array(n);         // exponential rate estimate (Hz), tau 50 ms
    this.t = 0;
    this.rng = mulberry32(params.seed ?? 1);
    // Gaussian noise table (Box-Muller), read with a random offset and odd stride each step
    this.noiseTab = new Float32Array(1 << 16);
    for (let k = 0; k < this.noiseTab.length; k += 2) {
      const u = this.rng() || 1e-12, v = this.rng(), r = Math.sqrt(-2 * Math.log(u));
      this.noiseTab[k] = r * Math.cos(2 * Math.PI * v); this.noiseTab[k + 1] = r * Math.sin(2 * Math.PI * v);
    }
    this.rebuildWeights();
  }

  rebuildWeights(plastic = null) {
    const { indptr, indices, count, w, gainIn } = this;
    const ws = this.p.wSyn;
    for (let i = 0; i < this.n; i++) {
      for (let e = indptr[i]; e < indptr[i + 1]; e++) w[e] = ws * count[e] * gainIn[indices[e]];
    }
    if (plastic) plastic.apply(this);
  }

  reset() {
    this.v.fill(this.p.vRest);
    for (let i = 0; i < this.n; i++) this.v[i] += (this.rng() * 6 - 3);
    this.g.fill(0); this.ref.fill(0); this.rate.fill(0); this.ext.fill(0);
  }

  // Advance one dt step. Returns the number of spikes.
  step() {
    const p = this.p, n = this.n, dt = p.dt;
    const { v, g, ext, thOff, ref, spiked, spikeList, rate, indptr, indices, w } = this;
    const aM = Math.exp(-dt / p.tauM), aS = Math.exp(-dt / p.tauSyn);
    // exact update of v driven by an exponentially decaying g over dt
    const kG = (p.tauSyn / (p.tauSyn - p.tauM)) * (aS - aM);
    const sig = p.noise * Math.sqrt(dt);
    const aR = Math.exp(-dt / 50), rInc = 1000 / 50;
    const tab = this.noiseTab, mask = tab.length - 1;
    let nk = (this.rng() * tab.length) | 0;
    const stride = (((this.rng() * 4096) | 0) << 1) + 1;
    let ns = 0;
    for (let i = 0; i < n; i++) {
      spiked[i] = 0;
      if (ref[i] > 0) {
        ref[i] -= dt;
        g[i] *= aS;
        rate[i] *= aR;
        continue;
      }
      const target = p.vRest + ext[i];
      let vi = target + (v[i] - target) * aM + g[i] * kG + sig * tab[nk];
      nk = (nk + stride) & mask;
      g[i] *= aS;
      if (vi >= p.vTh + thOff[i]) {
        vi = p.vReset;
        ref[i] = p.tRef;
        spiked[i] = 1;
        spikeList[ns++] = i;
        rate[i] = rate[i] * aR + rInc;
      } else {
        rate[i] *= aR;
      }
      v[i] = vi;
    }
    for (let s = 0; s < ns; s++) {
      const i = spikeList[s];
      for (let e = indptr[i], end = indptr[i + 1]; e < end; e++) g[indices[e]] += w[e];
    }
    this.nSpikes = ns;
    this.t += dt;
    return ns;
  }
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Parse web/data/brain.bin (see pipeline/build_brain.py).
export function parseNet(buf) {
  const ab = buf instanceof ArrayBuffer ? buf : buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const head = new Uint32Array(ab, 0, 2);
  const n = head[0], m = head[1];
  let off = 8;
  const indptr = new Uint32Array(ab, off, n + 1); off += 4 * (n + 1);
  const indices = new Uint32Array(ab, off, m); off += 4 * m;
  const weight = new Int16Array(ab.slice(off, off + 2 * m));
  return { n, m, indptr, indices, weight };
}
