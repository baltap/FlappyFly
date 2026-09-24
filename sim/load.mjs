import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { parseNet } from '../web/js/engine.js';

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

export function loadBrainData() {
  const net = parseNet(readFileSync(join(ROOT, 'web/data/brain.bin')));
  const meta = JSON.parse(readFileSync(join(ROOT, 'web/data/brain.json'), 'utf8'));
  return { net, meta };
}
