// Minimal static server for web/ (no dependencies). PORT env, default 8350.
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { join, extname, normalize } from 'node:path';
import { ROOT } from './load.mjs';

const WEB = join(ROOT, 'web');
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json', '.bin': 'application/octet-stream', '.f32': 'application/octet-stream', '.svg': 'image/svg+xml', '.png': 'image/png' };
const port = +(process.env.PORT ?? 8350);
createServer(async (req, res) => {
  try {
    let p = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^(\.\.[/\\])+/, '');
    if (p.endsWith('/')) p += 'index.html';
    const file = join(WEB, p);
    if (!file.startsWith(WEB)) { res.writeHead(403); return res.end(); }
    await stat(file);
    res.writeHead(200, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream', 'Cache-Control': 'no-cache' });
    res.end(await readFile(file));
  } catch {
    res.writeHead(404); res.end('not found');
  }
}).listen(port, () => console.log(`Flappy Fly on http://localhost:${port}`));
