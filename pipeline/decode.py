"""Ridge decoding of task variables from recorded brain activity (diagnostic)."""
import json, numpy as np, sys
R = 'runs/'
m = json.load(open(R + 'rec_meta.json'))
X = np.fromfile(R + 'rec_X.f32', dtype=np.float32).reshape(m['rows'], m['cols'])
Y = np.array(m['tgt']); names = ['gap_dy', 'vy', 'pipe_dx', 'y']
cls = np.array(m['cls']); types = np.array(m['types'])
n = len(X)
# blocked cross-validation: 5 contiguous folds
folds = np.array_split(np.arange(n), 5)
def fit(Xs, lam=10.0):
    keep = Xs.std(0) > 1e-3
    Xs = Xs[:, keep]
    P = np.zeros_like(Y)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-3
        Z = (Xs - mu) / sd
        if Z.shape[1] > 3000:   # dual form
            K = Z[tr] @ Z[tr].T
            A = np.linalg.solve(K + lam * np.eye(len(tr)), Y[tr] - Y[tr].mean(0))
            P[te] = Z[te] @ (Z[tr].T @ A) + Y[tr].mean(0)
        else:
            W = np.linalg.solve(Z[tr].T @ Z[tr] + lam * np.eye(Z.shape[1]), Z[tr].T @ (Y[tr] - Y[tr].mean(0)))
            P[te] = Z[te] @ W + Y[tr].mean(0)
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
keys = np.array(m['keys']); retino = np.char.find(keys.astype(str), '|e') >= 0
print('pools', len(keys), 'retinotopic', retino.sum())
for label, mask in [('EYE', cls == 'eye'), ('OPTIC', cls == 'optic'), ('OPT-ret', (cls == 'optic') & retino), ('VPN-ret', (cls == 'vpn') & retino), ('CENTRAL', cls == 'central'), ('ALL', cls != ''), ('DN', cls == 'dn'), ('MBON', cls == 'mbon'), ('KC', cls == 'kc'), ('VPN', cls == 'vpn'),
                    ('DNg02', np.char.startswith(types.astype(str), 'DNg02'))]:
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0
    r2 = fit(X[:, mask], lam)
    print(f"{label:6s} n={mask.sum():5d} " + " ".join(f"{k}={v:.2f}" for k, v in zip(names, r2)))
act = (X > 0.5).mean(0)
print('fraction of recorded neurons ever active', (X.max(0) > 1).mean().round(3))
