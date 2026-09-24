// Development: homeostatic synaptic scaling (Turrigiano-style). While the fly watches
// flight scenes, each neuron multiplicatively scales all of its input synapses so that
// its average firing rate approaches a class-specific set point. After development the
// gains are frozen; learning during the game is done by dopamine plasticity only.

export const TARGET_HZ = {
  eye: 12, optic: 10, vpn: 8, central: 5, dn: 6, kc: 1.5, mbon: 8, dan: 3, mbother: 25,
};

export class Homeostasis {
  constructor(brain, opts = {}) {
    this.b = brain;
    this.eta = opts.eta ?? 0.25;
    this.tau = opts.tau ?? 1000;         // ms, rate averaging
    this.period = opts.period ?? 100;    // ms between gain updates
    this.minGain = opts.minGain ?? 0.05;
    this.maxGain = opts.maxGain ?? 60;
    const n = brain.n;
    this.target = new Float32Array(n);
    brain.meta.class.forEach((c, i) => { this.target[i] = TARGET_HZ[c] ?? 5; });
    this.avg = new Float32Array(n);
    this.clock = 0;
    // neurons driven only from outside (eye) are not scaled
    this.fixed = new Uint8Array(n);
    for (const [i] of brain.meta.roles.eye) this.fixed[i] = 1;
  }

  // call once per brain step
  tick() {
    const b = this.b, dt = b.p.dt, a = dt / this.tau, n = b.n;
    const avg = this.avg, sp = b.spiked, inst = 1000 / dt;
    for (let i = 0; i < n; i++) avg[i] += a * ((sp[i] ? inst : 0) - avg[i]);
    this.clock += dt;
    if (this.clock < this.period) return false;
    this.clock = 0;
    const g = b.gainIn, tg = this.target;
    for (let i = 0; i < n; i++) {
      if (this.fixed[i]) continue;
      const err = (tg[i] - avg[i]) / (tg[i] + 2);
      let x = g[i] * Math.exp(this.eta * Math.max(-1, Math.min(1, err)));
      g[i] = x < this.minGain ? this.minGain : x > this.maxGain ? this.maxGain : x;
    }
    b.rebuildWeights();
    return true;
  }
}
