// Record frozen-brain activity under the scripted pilot for decoding diagnostics.
import { writeFileSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ROOT } from './load.mjs';
import { makeWorld, oraclePolicy } from './common.mjs';
import { buildPools, poolRates } from '../web/js/pools.js';

const seconds = +(process.argv[2] ?? 60);
const { brain, eye, game, meta } = makeWorld({ brain: { seed: 11 } });
brain.gainIn.set(new Float32Array(readFileSync(join(ROOT, 'web/data/gains.f32')).buffer.slice(0)));
brain.rebuildWeights();
brain.reset();
const pools = buildPools(meta);
const pick = { length: pools.n };
const pr = new Float32Array(pools.n);
const every = 20, rows = [], tgt = [], pop = [];
let rng = 1;
for (let t = 0; t < seconds * 1000; t++) {
  let act = 0;
  if (t % 10 === 0 && oraclePolicy(game)) { game.flap(); }
  act = game.y > game.nextPipe().gapY + 8 ? 1 : 0;
  game.step(1);
  if (game.dead) { game.reset(); eye.reset(); }
  eye.update(game, brain, 1);
  const ns = brain.step();
  pop.push(ns);
  if (t > 500 && t % every === 0) {
    rows.push(Float32Array.from(poolRates(pools, brain.rate, pr)));
    const p = game.nextPipe();
    tgt.push([p.gapY - game.y, game.vy * 1000, p.x - game.o.birdX, game.y, act]);
  }
}
const X = new Float32Array(rows.length * pick.length);
rows.forEach((r, i) => X.set(r, i * pick.length));
writeFileSync(join(ROOT, 'runs/rec_X.f32'), Buffer.from(X.buffer));
writeFileSync(join(ROOT, 'runs/rec_meta.json'), JSON.stringify({ rows: rows.length, cols: pick.length, cls: pools.info.map(p => p.cls), types: pools.info.map(p => p.type), keys: pools.info.map(p => p.key), tgt, pop }));
console.log('rows', rows.length, 'cols', pick.length, 'max pop spikes/ms', pop.reduce((a, b) => Math.max(a, b), 0), 'mean', (pop.reduce((a, b) => a + b) / pop.length).toFixed(0));
