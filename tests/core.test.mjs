import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ROOT, loadBrainData } from '../sim/load.mjs';
import { Game, GAME_DEFAULTS } from '../web/js/game.js';
import { Eye } from '../web/js/eye.js';
import { Brain } from '../web/js/engine.js';
import { buildPools } from '../web/js/pools.js';
import { Agent } from '../web/js/agent.js';
import { parseNpca } from '../web/js/npca.js';

const { net, meta } = loadBrainData();
const gains = new Float32Array(readFileSync(join(ROOT, 'web/data/gains.f32')).buffer.slice(0));
const npca = parseNpca(readFileSync(join(ROOT, 'web/data/npca_lam.bin')));

test('game: a flap peaks ~42 px higher after ~307 ms, like the original', () => {
  const g = new Game({ firstPipe: 5000 });
  const y0 = g.y;
  g.flap();
  let minY = y0, tMin = 0;
  for (let t = 1; t < 600; t++) { g.step(1); if (g.y < minY) { minY = g.y; tMin = t; } }
  assert.ok(Math.abs(y0 - minY - 42) < 2, `apex ${y0 - minY}`);
  assert.ok(Math.abs(tMin - 307) < 5, `t ${tMin}`);
});

test('game: scoring and crashing', () => {
  const g = new Game({ maxGapShift: 0 }, () => 0.5);
  const events = [];
  for (let t = 0; t < 20000 && !g.dead; t++) {
    const p = g.nextPipe();
    if (g.y > p.gapY + 8 && g.vy > -0.05 && t % 10 === 0) g.flap();
    events.push(...g.step(1));
  }
  assert.ok(g.score >= 5, `score ${g.score}`);
  const g2 = new Game();
  for (let t = 0; t < 3000 && !g2.dead; t++) g2.step(1);
  assert.equal(g2.deathCause, 'ground');
});

test('eye: the gap shows up above the horizon when it is above the fly', () => {
  const eye = new Eye(meta);
  const g = new Game({ maxGapShift: 0 }, () => 0.5);
  g.pipes[0].x = GAME_DEFAULTS.birdX + 60;
  const frontal = (lo, hi) => {
    let s = 0, n = 0;
    eye.dirs.forEach(([az, el], d) => { if (az < 5 && el > lo && el < hi) { s += eye.lum[d]; n++; } });
    return s / n;
  };
  g.pipes[0].gapY = g.y - 70;           // gap well above the fly
  eye.render(g);
  const up = frontal(20, 60), down = frontal(-60, -20);
  assert.ok(up > down + 0.1, `gap above: up ${up.toFixed(2)} down ${down.toFixed(2)}`);
  g.pipes[0].gapY = g.y + 70;           // gap below
  eye.render(g);
  assert.ok(frontal(-60, -20) > frontal(20, 60) + 0.1, 'gap below');
});

test('connectome: sizes, signs and roles are consistent', () => {
  assert.equal(net.n, meta.n);
  assert.equal(net.indptr[net.n], net.m);
  assert.ok(meta.roles.flapDN.length >= 10);
  assert.ok(meta.roles.kc.length > 500 && meta.roles.mbon.length > 50);
  let pos = 0, neg = 0;
  for (const w of net.weight) { if (w > 0) pos++; else if (w < 0) neg++; }
  assert.ok(pos > neg && neg > 0);
  assert.equal(meta.danMbon.length, meta.danOrder.length);
});

test('engine: silent without input, active with the developed gains and a lit eye', () => {
  const b = new Brain(net, meta, { noise: 0 });
  for (let i = 0; i < 50; i++) b.step();
  assert.equal(b.nSpikes, 0);
  const b2 = new Brain(net, meta, { seed: 3 });
  b2.gainIn.set(gains); b2.rebuildWeights(); b2.reset();
  for (const [i] of meta.roles.eye) b2.ext[i] = 12;
  let spikes = 0;
  for (let i = 0; i < 300; i++) spikes += b2.step();
  const rate = spikes / b2.n / 0.3;
  assert.ok(rate > 1 && rate < 60, `mean rate ${rate.toFixed(1)} Hz`);
});

test('pools: retinotopic pools carry regions, both eyes merge when asked', () => {
  const a = buildPools(meta, { include: (i, c) => c === 'optic' });
  const b = buildPools(meta, { include: (i, c) => c === 'optic', mergeSides: true });
  assert.ok(b.n < a.n);
  assert.ok(a.info.some(p => p.region));
});

test('agent: learns only when allowed, plasticity stays bounded, snapshots round-trip', () => {
  const ag = new Agent(net, meta, gains, { npca, pcaK: 16, seed: 4, mbEta: 5e-6 });
  for (let i = 0; i < 1500; i++) ag.step();
  const f = ag.mb.f;
  for (const x of f) assert.ok(x >= 0 && x <= 4);
  const snap = ag.snapshot();
  const ag2 = new Agent(net, meta, gains, { npca, pcaK: 16, seed: 5 });
  ag2.load(JSON.parse(JSON.stringify(snap)));
  assert.deepEqual(Array.from(ag2.W[1]), snap.W1);
  const frozen = new Agent(net, meta, gains, { npca, pcaK: 16, seed: 6 });
  frozen.learning = false;
  for (let i = 0; i < 1500; i++) frozen.step();
  assert.ok(frozen.W[0].every(v => v === 0) && frozen.W[1].every(v => v === 0));
});
