import { deflateSync } from 'node:zlib';
import { writeFileSync } from 'node:fs';
// Write an 8-bit grayscale or RGB PNG. pixels: Uint8Array of w*h*(1|3).
export function writePNG(path, w, h, pixels, channels = 1) {
  const crcT = new Int32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; });
  const crc = (b) => { let c = -1; for (const x of b) c = crcT[(c ^ x) & 255] ^ (c >>> 8); return (c ^ -1) >>> 0; };
  const chunk = (type, data) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type), data]); const c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([len, td, c]); };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = channels === 3 ? 2 : 0;
  const raw = Buffer.alloc((w * channels + 1) * h);
  for (let y = 0; y < h; y++) { raw[y * (w * channels + 1)] = 0; Buffer.from(pixels.buffer, pixels.byteOffset + y * w * channels, w * channels).copy(raw, y * (w * channels + 1) + 1); }
  writeFileSync(path, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]));
}
