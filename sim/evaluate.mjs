// Evaluate a trained fly with learning and exploration off.
//   node sim/evaluate.mjs runs/fly.json [--lives 30] [--blind] [--seed 7] [--maxSec 120]
// --blind cuts the eye input (lamina drive = 0): the readout then gets no visual signal.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ROOT, loadBrainData } from './load.mjs';
import { Agent } from '../web/js/agent.js';
import { parseNpca } from '../web/js/npca.js';

const argv = process.argv.slice(2), args = {};
for (let i = 1; i < argv.length; i++) {
  if (!argv[i].startsWith('--')) continue;
  const nx = argv[i + 1];
  if (nx === undefined || nx.startsWith('--')) args[argv[i].slice(2)] = true; else { args[argv[i].slice(2)] = nx; i++; }
}
const snap = JSON.parse(readFileSync(argv[0], 'utf8'));
const lives = +(args.lives ?? 30), maxMs = +(args.maxSec ?? 120) * 1000;
const { net, meta } = loadBrainData();
const gains = new Float32Array(readFileSync(join(ROOT, 'web/data/gains.f32')).buffer.slice(0));
const npca = parseNpca(readFileSync(join(ROOT, `web/data/npca_${snap.npcaName ?? 'lam'}.bin`)));
const agent = new Agent(net, meta, gains, {
  mode: args.mode ?? snap.mode, seed: +(args.seed ?? 7), npca, pcaK: snap.pcaK, efference: snap.efference, conj: snap.conj,
  game: { gap: +(args.gap ?? 150), maxGapShift: +(args.shift ?? 60) },
});
agent.load(snap);
agent.history = [];
agent.learning = false; agent.mb.enabled = false; agent.explore = false;
if (args.blind) {
  const upd = agent.eye.update.bind(agent.eye);
  agent.eye.update = (g, b, dt) => { upd(g, null, dt); for (const i of agent.eye.neuron) b.ext[i] = 0; };
}
const start = agent.episode;
let lifeMs = 0;
while (agent.episode - start < lives) {
  agent.step();
  if (++lifeMs > maxMs) { agent.game.dead = true; agent.game.deathCause = 'timeout'; agent._endEpisode(); }
  if (agent.game.t === 0) lifeMs = 0;
}
const s = agent.history.map(h => h.score).sort((a, b) => a - b);
const mean = s.reduce((a, b) => a + b, 0) / s.length;
console.log(JSON.stringify({ snapshot: argv[0], blind: !!args.blind, lives: s.length, mean: +mean.toFixed(2),
  median: s[s.length >> 1], best: s[s.length - 1], zeroFrac: +(s.filter(x => x === 0).length / s.length).toFixed(2), scores: agent.history.map(h => h.score) }));
