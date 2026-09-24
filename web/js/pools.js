// Population pools: neurons grouped by cell type, side and (for retinotopic cells) the
// region of the eye their receptive field covers. Pool activity = mean firing rate.
// These are the inputs of the learned readout (the stand-in for the missing premotor
// circuits of the nerve cord) and of the value estimate.

const EL_EDGES = [-80, -40, -20, -8, 0, 8, 20, 40, 81];   // fine near the horizon
const AZ_EDGES = [-25, 20, 60, 180];                      // frontal, fronto-lateral, lateral/rear

function bin(x, edges) {
  for (let k = 0; k < edges.length - 1; k++) if (x < edges[k + 1]) return k;
  return edges.length - 2;
}

export function buildPools(meta, opts = {}) {
  const include = opts.include ?? (() => true);
  const wide = opts.wideDeg ?? 25;
  const merge = opts.mergeSides ?? false;   // both eyes see the same scene
  const keyOf = new Map();
  const members = [];
  const info = [];
  const poolOf = new Int32Array(meta.n).fill(-1);
  for (let i = 0; i < meta.n; i++) {
    const cls = meta.class[i];
    if (!include(i, cls)) continue;
    const type = meta.types[meta.type[i]];
    const side = meta.side[i];
    const rf = meta.rf ? meta.rf[i] : null;
    let key, region = null;
    if (rf && rf[2] < wide) {
      const e = bin(rf[1], EL_EDGES), a = bin(rf[0], AZ_EDGES);
      key = merge ? `${type}|e${e}|a${a}` : `${type}|${side}|e${e}|a${a}`;
      region = { e, a, el: (EL_EDGES[e] + EL_EDGES[e + 1]) / 2 };
    } else {
      key = `${type}|${side}`;
    }
    let p = keyOf.get(key);
    if (p === undefined) {
      p = members.length; keyOf.set(key, p); members.push([]);
      info.push({ key, type, side, cls, region });
    }
    members[p].push(i);
    poolOf[i] = p;
  }
  const flat = new Int32Array(members.reduce((s, m) => s + m.length, 0));
  const start = new Int32Array(members.length + 1);
  let k = 0;
  members.forEach((m, p) => { start[p] = k; for (const i of m) flat[k++] = i; });
  start[members.length] = k;
  return { n: members.length, flat, start, info, poolOf };
}

// Mean rate per pool (Hz) into out (Float32Array of pools.n).
export function poolRates(pools, rate, out) {
  const { flat, start, n } = pools;
  for (let p = 0; p < n; p++) {
    let s = 0;
    const a = start[p], b = start[p + 1];
    for (let k = a; k < b; k++) s += rate[flat[k]];
    out[p] = s / (b - a);
  }
  return out;
}
