// Record individual-neuron rates of retinotopic columnar cells (lamina + medulla/lobula
// types with a hex column) under a mixed pilot, for developmental PCA.
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { ROOT, loadBrainData } from './load.mjs';
import { Agent } from '../web/js/agent.js';

const seconds = +(process.argv[2] ?? 300);
const { net, meta } = loadBrainData();
const gains = new Float32Array(readFileSync(join(ROOT, 'web/data/gains.f32')).buffer.slice(0));
const agent = new Agent(net, meta, gains, { mode: 'hybrid', seed: 21, game: { gap: 150, maxGapShift: 60 }, featureMode: 'raw', efference: false });
agent.learning = false;
let mode = 0;
agent.choose = function () {
  const g = this.game, p = g.nextPipe();
  if (this.clock % 3000 < this.o.decisionMs) mode = this.rng() < 0.3 ? 1 : 0;
  if (mode === 1) return this.rng() < 0.12 ? 1 : 0;
  return g.y > p.gapY + (this.rng() - 0.5) * 60 && g.vy > -0.05 ? 1 : 0;
};
const pick = [];
meta.class.forEach((c, i) => { const rf = meta.rf[i]; if ((c === 'eye' || c === 'optic' || c === 'vpn') && rf && rf[2] < 6) pick.push(i); });
const rows = [], tgt = [];
while (agent.clock < seconds * 1000) {
  agent.step();
  if (agent.clock % 40 === 0 && agent.clock > 3000) {
    const r = new Float32Array(pick.length);
    for (let k = 0; k < pick.length; k++) r[k] = agent.brain.rate[pick[k]];
    rows.push(r);
    const g = agent.game, p = g.nextPipe();
    tgt.push([p.gapY - g.y, g.vy * 1000, p.x - g.o.birdX, g.y]);
  }
}
const X = new Float32Array(rows.length * pick.length);
rows.forEach((r, i) => X.set(r, i * pick.length));
writeFileSync(join(ROOT, 'runs/neu_X.f32'), Buffer.from(X.buffer));
writeFileSync(join(ROOT, 'runs/neu_meta.json'), JSON.stringify({ rows: rows.length, cols: pick.length, pick, tgt,
  types: pick.map(i => meta.types[meta.type[i]]) }));
console.log('rows', rows.length, 'neurons', pick.length);
