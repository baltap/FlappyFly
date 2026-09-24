// Developmental principal components over individual retinotopic neurons
// (pipeline/develop_npca.py): the readout sees the early visual system through them.
export function parseNpca(buf) {
  const ab = buf instanceof ArrayBuffer ? buf : buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const [k, n] = new Uint32Array(ab, 0, 2);
  let off = 8;
  const idx = new Int32Array(ab, off, n); off += 4 * n;
  const mean = new Float32Array(ab, off, n); off += 4 * n;
  const sd = new Float32Array(ab, off, n); off += 4 * n;
  const scale = new Float32Array(ab, off, k); off += 4 * k;
  const comp = new Float32Array(ab, off, k * n);
  return { k, n, idx, mean, sd, scale, comp };
}
