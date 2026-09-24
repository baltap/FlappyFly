import { loadBrainData } from './load.mjs';
import { Brain } from '../web/js/engine.js';
import { Eye } from '../web/js/eye.js';
import { Game } from '../web/js/game.js';

// Scripted pilot used only for developmental visual experience and diagnostics.
export function oraclePolicy(game, rng = Math.random) {
  const p = game.nextPipe();
  return game.y > p.gapY + 8 + (rng() - 0.5) * 30 && game.vy > -0.05;
}

export function classRates(brain) {
  const out = {};
  brain.meta.class.forEach((c, i) => { (out[c] ??= [0, 0]); out[c][0]++; out[c][1] += brain.rate[i]; });
  return Object.fromEntries(Object.entries(out).map(([k, [n, s]]) => [k, +(s / n).toFixed(2)]));
}

export function makeWorld(params = {}) {
  const { net, meta } = loadBrainData();
  const brain = new Brain(net, meta, params.brain);
  const eye = new Eye(meta, params.eye);
  const game = new Game(params.game);
  return { brain, eye, game, meta };
}
