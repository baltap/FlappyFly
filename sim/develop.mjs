// Developmental phase: homeostatic scaling under natural flight scenes (scripted pilot),
// then freeze the gains to web/data/gains.f32.
import { writeFileSync, existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ROOT } from './load.mjs';
import { makeWorld, oraclePolicy, classRates } from './common.mjs';
import { Homeostasis } from '../web/js/homeostasis.js';

const seconds = +(process.argv[2] ?? 60);
const { brain, eye, game } = makeWorld({ brain: { seed: 3 } });
const eta = +(process.argv[3] ?? 0.08);
const gp = join(ROOT, 'web/data/gains.f32');
if (process.argv[4] === 'warm' && existsSync(gp)) {
  brain.gainIn.set(new Float32Array(readFileSync(gp).buffer.slice(0)));
  brain.rebuildWeights();
  console.log('warm start');
}
const homeo = new Homeostasis(brain, { eta, period: 200 });
brain.reset();
const t0 = performance.now();
for (let t = 0; t < seconds * 1000; t++) {
  if (t % 10 === 0 && oraclePolicy(game)) game.flap();
  game.step(1);
  if (game.dead) { game.reset(); eye.reset(); }
  eye.update(game, brain, 1);
  brain.step();
  homeo.tick();
  if (t % 5000 === 4999) {
    const g = [...brain.gainIn].sort((a, b) => a - b);
    console.log(`${((t + 1) / 1000).toFixed(0)}s`, JSON.stringify(classRates(brain)),
      'gain p10/50/90', g[g.length * 0.1 | 0].toFixed(2), g[g.length / 2 | 0].toFixed(2), g[g.length * 0.9 | 0].toFixed(2),
      `(${((performance.now() - t0) / 1000).toFixed(0)}s wall)`);
  }
}
writeFileSync(join(ROOT, 'web/data/gains.f32'), Buffer.from(brain.gainIn.buffer));
console.log('saved gains');
