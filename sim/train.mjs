// Headless training of one fly. Usage:
//   node sim/train.mjs --mode hybrid --episodes 300 --out runs/hybrid.json [--gap 150 --shift 60]
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { ROOT, loadBrainData } from './load.mjs';
import { Agent } from '../web/js/agent.js';
import { parseNpca } from '../web/js/npca.js';

const argv = process.argv.slice(2), args = {};
for (let i = 0; i < argv.length; i++) {
  if (!argv[i].startsWith('--')) continue;
  const next = argv[i + 1];
  if (next === undefined || next.startsWith('--')) args[argv[i].slice(2)] = true;
  else { args[argv[i].slice(2)] = next; i++; }
}
const mode = args.mode ?? 'hybrid';
const episodes = +(args.episodes ?? 300);
const out = args.out ?? `runs/${mode}.json`;
const game = { gap: +(args.gap ?? 150), maxGapShift: +(args.shift ?? 60) };
const opts = { mode, seed: +(args.seed ?? 1), game };
for (const k of ['replaySize', 'replayPerLife', 'replayAlpha', 'pcaK', 'alpha', 'gamma', 'lambda', 'wBio', 'mbEta', 'daGain', 'shaping', 'eps0', 'eps1', 'epsEpisodes', 'decisionMs', 'bioThreshold'])
  if (args[k] !== undefined) opts[k] = +args[k];
if (args.nomb) opts.mbEta = 0;
if (args.featureMode) opts.featureMode = args.featureMode;
if (args.noeff) opts.efference = false;
if (args.noconj) opts.conj = false;
if ((opts.featureMode ?? 'npca') === 'npca') opts.npca = parseNpca(readFileSync(join(ROOT, `web/data/npca_${args.npca ?? 'lam'}.bin`)));
opts.npcaName = args.npca ?? 'lam';

const { net, meta } = loadBrainData();
const gains = new Float32Array(readFileSync(join(ROOT, 'web/data/gains.f32')).buffer.slice(0));
const agent = new Agent(net, meta, gains, opts);
if (args.resume && existsSync(args.resume)) agent.load(JSON.parse(readFileSync(args.resume, 'utf8')));
if (args.frozen) agent.learning = false;
console.log(`mode ${mode}, ${agent.nFeat} readout features (${agent.o.featureMode}), ${agent.mb.edge.length} plastic KC->MBON synapses`, JSON.stringify({ ...opts, npca: undefined }));
const t0 = performance.now();
let best = 0;
agent.onEpisodeEnd = (h) => {
  best = Math.max(best, h.score);
  const last = agent.history.slice(-20);
  const avg = last.reduce((s, x) => s + x.score, 0) / last.length;
  const avgT = last.reduce((s, x) => s + x.t, 0) / last.length / 1000;
  if (h.ep % 10 === 9 || h.score >= 5) {
    const mb = agent.mb.stats();
    console.log(`ep ${h.ep + 1} score ${h.score} (${h.cause}) | last20 avg ${avg.toFixed(2)} life ${avgT.toFixed(1)}s | best ${best} | eps ${agent.epsilon.toFixed(3)} | MB f ${mb.mean.toFixed(3)}±${mb.sd.toFixed(3)} | ${((performance.now() - t0) / 1000).toFixed(0)}s`);
  }
  if (h.ep % 25 === 24) writeFileSync(join(ROOT, out), JSON.stringify(agent.snapshot()));
};
const maxMs = +(args.maxSimSec ?? 1e9) * 1000;
let ms = 0;
while (agent.episode < episodes && ms < maxMs) {
  // warm-up: let the normalizers settle before the first decisions count
  agent.step(); ms++;
}
writeFileSync(join(ROOT, out), JSON.stringify(agent.snapshot()));
const h = agent.history;
const q = (a, b) => { const s = h.slice(a, b); return s.length ? (s.reduce((x, y) => x + y.score, 0) / s.length).toFixed(2) : '-'; };
console.log(`done: ${h.length} episodes; mean score first 50 ${q(0, 50)}, last 50 ${q(-50)}; best ${best}`);
