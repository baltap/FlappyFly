"""Developmental PCA over individual retinotopic neurons (recorded by sim/record_neurons.mjs).
Writes web/data/npca_<name>.bin:
  u32 k, u32 n | i32 idx[n] | f32 mean[n] | f32 sd[n] | f32 scale[k] | f32 comp[k*n]
Usage: uv run python pipeline/develop_npca.py <name> <k> [type,type,...]"""
import json, sys, numpy as np
name, K = sys.argv[1], int(sys.argv[2])
types_sel = sys.argv[3].split(',') if len(sys.argv) > 3 else None
m = json.load(open('runs/neu_meta.json'))
X = np.fromfile('runs/neu_X.f32', dtype=np.float32).reshape(m['rows'], m['cols'])[100:]
Y = np.array(m['tgt'])[100:]
pick = np.array(m['pick']); types = np.array(m['types'])
msk = np.isin(types, types_sel) if types_sel else np.ones(len(pick), bool)
X = X[:, msk]; idx = pick[msk]
mean = X.mean(0); sd = X.std(0) + 1.0
Z = (X - mean) / sd
U, S, Vt = np.linalg.svd(Z, full_matrices=False)
comp = Vt[:K].astype(np.float32); scale = (S[:K] / np.sqrt(len(Z))).astype(np.float32)
F = Z @ comp.T / scale
n = len(F); folds = np.array_split(np.arange(n), 5); P = np.zeros_like(Y)
for te in folds:
    tr = np.setdiff1d(np.arange(n), te)
    W = np.linalg.solve(F[tr].T @ F[tr] + 10 * np.eye(K), F[tr].T @ (Y[tr] - Y[tr].mean(0)))
    P[te] = F[te] @ W + Y[tr].mean(0)
r2 = 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
print(name, 'neurons', len(idx), 'k', K, 'R2 gap_dy/vy/pipe_dx/y', r2.round(2))
with open(f'web/data/npca_{name}.bin', 'wb') as fh:
    fh.write(np.array([K, len(idx)], dtype=np.uint32).tobytes())
    fh.write(idx.astype(np.int32).tobytes())
    fh.write(mean.astype(np.float32).tobytes()); fh.write(sd.astype(np.float32).tobytes())
    fh.write(scale.tobytes()); fh.write(comp.tobytes())
