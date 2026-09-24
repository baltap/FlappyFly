// The compound eye. Each lamina column has a viewing direction (azimuth, elevation)
// taken from its hex coordinate in the connectome (pipeline/build_brain.py). We ray-cast
// the Flappy world in 3D from the fly's head: pipes are walls across the flight path
// with a horizontal slot, the ground is a striped plane, the sky is bright.
//
// Lamina neurons L1, L2 and L3 all depolarize to darkness. L1/L2 are transient
// (contrast of change), L3 is sustained. Their depolarization becomes external drive.

const DEG = Math.PI / 180;

export const EYE_DEFAULTS = {
  eyeDt: 8,           // ms between re-renders
  tauP: 8,            // photoreceptor low-pass, ms
  tauAdapt: 300,      // luminance adaptation, ms
  e0: 5.5,            // resting drive, mV above rest
  kTransient: 16,     // L1/L2 gain on (negative) contrast
  kSustained: 9,      // L3 gain on darkness
  kSustainedL12: 2.5,
  acceptance: 2.5,    // deg, offset of the 4 extra sample rays
  pillarHalfW: 100,    // px, lateral half-width of a pipe pillar
  lum: { sky: 0.95, pipe: 0.12, pipeInner: 0.07, groundA: 0.62, groundB: 0.38, haze: 0.75 },
};

export class Eye {
  constructor(meta, opts = {}) {
    this.o = { ...EYE_DEFAULTS, ...opts, lum: { ...EYE_DEFAULTS.lum, ...(opts.lum || {}) } };
    const dirKey = new Map();
    this.dirs = [];        // [az, el] deg
    this.dirSide = [];     // eye of the column ('R' | 'L')
    this.neuron = [];      // neuron index
    this.dirOf = [];       // direction index per eye neuron
    this.kind = [];        // 1=L1 2=L2 3=L3
    this.sideOf = [];      // 'R' | 'L'
    for (const [i, side, t, az, el] of meta.roles.eye) {
      const key = `${az}|${el}`;
      let d = dirKey.get(key);
      if (d === undefined) { d = this.dirs.length; dirKey.set(key, d); this.dirs.push([az, el]); this.dirSide.push(side); }
      this.neuron.push(i); this.dirOf.push(d); this.kind.push(t === 'L1' ? 1 : t === 'L2' ? 2 : 3);
      this.sideOf.push(side);
    }
    const nd = this.dirs.length;
    this.lum = new Float32Array(nd);
    this.P = new Float32Array(nd).fill(this.o.lum.sky);
    this.Pslow = new Float32Array(nd).fill(this.o.lum.sky);
    this.contrast = new Float32Array(nd);
    // precompute sample ray unit vectors: forward, up, lateral
    const a = this.o.acceptance;
    const offs = [[0, 0], [a, 0], [-a, 0], [0, a], [0, -a]];
    this.rays = new Float32Array(nd * offs.length * 3);
    this.nRay = offs.length;
    let k = 0;
    for (const [az, el] of this.dirs) {
      for (const [da, de] of offs) {
        const A = (az + da) * DEG, E = (el + de) * DEG;
        this.rays[k++] = Math.cos(E) * Math.cos(A);
        this.rays[k++] = Math.sin(E);
        this.rays[k++] = Math.cos(E) * Math.sin(A);
      }
    }
    this.clock = 0;
  }

  reset() {
    this.P.fill(this.o.lum.sky); this.Pslow.fill(this.o.lum.sky); this.contrast.fill(0); this.clock = 0;
  }

  // Luminance along one ray from the bird.
  _cast(fwd, up, lat, game) {
    const o = game.o, L = this.o.lum;
    const hb = o.groundY - game.y;
    let tHit = Infinity, lum = L.sky;
    if (up < -1e-4) {
      const tg = hb / -up;
      tHit = tg;
      const X = tg * fwd + game.scroll, Z = tg * lat;
      const stripe = Math.sin((X + Z * 0.6) * (2 * Math.PI / 28)) > 0 ? L.groundA : L.groundB;
      lum = stripe;
    }
    // Pipes are square pillars (pipeW deep, 2*halfW wide) with a horizontal slot.
    // Slab intersection in the ground plane (forward X, lateral Z), then height tests.
    const hw = this.o.pillarHalfW;
    const invF = Math.abs(fwd) > 1e-6 ? 1 / fwd : 1e9, invL = Math.abs(lat) > 1e-6 ? 1 / lat : 1e9;
    let tz0 = (-hw) * invL, tz1 = hw * invL;
    if (tz0 > tz1) { const s = tz0; tz0 = tz1; tz1 = s; }
    for (const p of game.pipes) {
      const xn = p.x - o.birdX, xf = xn + o.pipeW;
      let tx0 = xn * invF, tx1 = xf * invF;
      if (tx0 > tx1) { const s = tx0; tx0 = tx1; tx1 = s; }
      const t0 = Math.max(tx0, tz0, 0), t1 = Math.min(tx1, tz1);
      if (t1 <= t0 || t0 >= tHit) continue;
      const gTop = o.groundY - (p.gapY - o.gap / 2); // height of the gap's upper edge
      const gBot = o.groundY - (p.gapY + o.gap / 2);
      const h0 = hb + up * t0;
      if (h0 > gTop || h0 < gBot) { tHit = t0; lum = L.pipe; continue; }
      const h1 = hb + up * t1;
      if (h1 > gTop || h1 < gBot) {
        // ray enters through the slot and hits its ceiling or floor
        const tIn = up > 0 ? (gTop - hb) / up : (gBot - hb) / up;
        if (tIn < tHit) { tHit = tIn; lum = L.pipeInner; }
      }
    }
    if (tHit < Infinity) lum += (L.haze - lum) * (1 - Math.exp(-tHit / 900));
    return lum;
  }

  render(game) {
    const nr = this.nRay, rays = this.rays, lum = this.lum;
    for (let d = 0, k = 0; d < lum.length; d++) {
      let s = 0;
      for (let r = 0; r < nr; r++, k += 3) s += this._cast(rays[k], rays[k + 1], rays[k + 2], game);
      lum[d] = s / nr;
    }
  }

  // Advance photoreceptors by dt and write lamina drive into brain.ext.
  update(game, brain, dt = 1) {
    this.clock += dt;
    if (this.clock >= this.o.eyeDt || this.clock === dt) { this.render(game); if (this.clock >= this.o.eyeDt) this.clock = 0; }
    const o = this.o;
    const aP = 1 - Math.exp(-dt / o.tauP), aA = 1 - Math.exp(-dt / o.tauAdapt);
    const { P, Pslow, lum, contrast } = this;
    for (let d = 0; d < P.length; d++) {
      P[d] += (lum[d] - P[d]) * aP;
      Pslow[d] += (P[d] - Pslow[d]) * aA;
      contrast[d] = (P[d] - Pslow[d]) / (Pslow[d] + 0.1);
    }
    if (!brain) return;
    const ext = brain.ext;
    for (let j = 0; j < this.neuron.length; j++) {
      const d = this.dirOf[j];
      const dark = 1 - P[d];
      let v;
      if (this.kind[j] === 3) v = o.e0 + o.kSustained * dark - 3;
      else v = o.e0 + o.kTransient * -contrast[d] + o.kSustainedL12 * dark - 1;
      ext[this.neuron[j]] = v < 0 ? 0 : v > 30 ? 30 : v;
    }
  }
}
