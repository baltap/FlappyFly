// Record the agent's readout features under a mixed pilot (scripted + random) for the
// developmental PCA. Usage: node sim/record_features.mjs [seconds]
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { ROOT, loadBrainData } from './load.mjs';
import { Agent } from '../web/js/agent.js';

const seconds = +(process.argv[2] ?? 240);
const { net, meta } = loadBrainData();
const gains = new Float32Array(readFileSync(join(ROOT, 'web/data/gains.f32')).buffer.slice(0));
const agent = new Agent(net, meta, gains, { mode: 'hybrid', seed: 21, game: { gap: 150, maxGapShift: 60 }, featureMode: 'raw', efference: false });
agent.learning = false;
let rnd = 0.5;
let mode = 0;
agent.choose = function () {
  const g = this.game, p = g.nextPipe();
  if (this.clock % 3000 === 0) mode = this.rng() < 0.3 ? 1 : 0;       // 30% of time random flapping
  if (mode === 1) return this.rng() < 0.12 ? 1 : 0;
  const target = p.gapY + (this.rng() - 0.5) * 60;
  return g.y > target && g.vy > -0.05 ? 1 : 0;
};
const rows = [], tgt = [];
const t0 = performance.now();
while (agent.clock < seconds * 1000) {
  agent.step();
  if (agent.clock % agent.o.decisionMs === 0 && agent.clock > 3000) {
    rows.push(Float32Array.from(agent.xRaw ?? agent.x));
    const g = agent.game, p = g.nextPipe();
    tgt.push([p.gapY - g.y, g.vy * 1000, p.x - g.o.birdX, g.y]);
  }
}
const n = agent.xRaw ? agent.xRaw.length : agent.x.length;
const X = new Float32Array(rows.length * n);
rows.forEach((r, i) => X.set(r, i * n));
writeFileSync(join(ROOT, 'runs/feat_X.f32'), Buffer.from(X.buffer));
writeFileSync(join(ROOT, 'runs/feat_meta.json'), JSON.stringify({ rows: rows.length, cols: n, tgt, keys: agent.featInfo.map(f => f.key), episodes: agent.episode }));
console.log('rows', rows.length, 'cols', n, 'episodes', agent.episode, `${((performance.now() - t0) / 1000).toFixed(0)}s`);
