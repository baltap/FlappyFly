"""Developmental dimensionality reduction of the visual populations (PCA, the fixed point
of Hebbian/Oja learning) from recorded flight; writes web/data/pca.json.
Also reports how well task variables can be read out linearly (blocked CV)."""
import json, sys, numpy as np
K = int(sys.argv[1]) if len(sys.argv) > 1 else 64
m = json.load(open('runs/feat_meta.json'))
X = np.fromfile('runs/feat_X.f32', dtype=np.float32).reshape(m['rows'], m['cols'])[:, :-1]   # drop bias
Y = np.array(m['tgt'])
X = X[200:]; Y = Y[200:]                       # skip normalizer warm-up
mu = X.mean(0); Xc = X - mu
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
ev = S ** 2 / (S ** 2).sum()
print('variance explained by top 8/32/64/128:', [round(float(ev[:k].sum()), 3) for k in (8, 32, 64, 128)])
n = len(X); folds = np.array_split(np.arange(n), 5)
def cv(F, lam=1.0):
    P = np.zeros_like(Y)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        a, s = F[tr].mean(0), F[tr].std(0) + 1e-6
        Z = (F - a) / s
        W = np.linalg.solve(Z[tr].T @ Z[tr] + lam * np.eye(Z.shape[1]), Z[tr].T @ (Y[tr] - Y[tr].mean(0)))
        P[te] = Z[te] @ W + Y[tr].mean(0)
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
names = ['gap_dy', 'vy', 'pipe_dx', 'y']
for k in (8, 16, 32, 64, 128, 256):
    F = Xc @ Vt[:k].T
    print(f'PCs {k:4d}', ' '.join(f'{a}={b:.2f}' for a, b in zip(names, cv(F, 10.0))))
# sparse random expansion (Kenyon-cell-like) of the top K PCs
rng = np.random.default_rng(0)
F = Xc @ Vt[:K].T; F = F / (F.std(0) + 1e-6)
for nk in (1000, 2000):
    J = rng.normal(0, 1, (K, nk)) * (rng.random((K, nk)) < 0.15)
    H = F @ J
    th = np.percentile(H, 90, axis=0)
    A = np.maximum(H - th, 0)
    print(f'KC {nk} (10% active)', ' '.join(f'{a}={b:.2f}' for a, b in zip(names, cv(A, 30.0))))
comp = Vt[:K]
json.dump({'k': K, 'keys': m['keys'], 'mean': np.round(mu, 5).tolist(),
           'components': np.round(comp, 5).tolist(), 'scale': np.round(S[:K] / np.sqrt(n), 5).tolist()},
          open('web/data/pca.json', 'w'))
print('wrote web/data/pca.json', K)

# temporal-difference features (sustained + transient channel): PC(t) and PC(t) - PC(t-1)
for k in (64, 128):
    F = Xc @ Vt[:k].T
    D = np.vstack([np.zeros((1, k)), np.diff(F, axis=0)])
    D2 = np.vstack([np.zeros((2, k)), F[2:] - F[:-2]])
    print(f'PCs {k} + diff', ' '.join(f'{a}={b:.2f}' for a, b in zip(names, cv(np.hstack([F, D, D2]), 10.0))))
