"""Ceiling test: logistic readout of the scripted pilot's decision from pooled brain activity."""
import json, sys, numpy as np
m = json.load(open('runs/rec_meta.json'))
X = np.fromfile('runs/rec_X.f32', dtype=np.float32).reshape(m['rows'], m['cols'])
T = np.array(m['tgt']); y = T[:, 4]
keys = np.array(m['keys']).astype(str); cls = np.array(m['cls'])
retino = np.char.find(keys, '|e') >= 0
sel = retino & np.isin(cls, ['optic', 'vpn', 'eye'])
which = sys.argv[1] if len(sys.argv) > 1 else 'retino'
if which == 'dn': sel = cls == 'dn'
if which == 'mbon': sel = cls == 'mbon'
Xs = X[:, sel]; mu = Xs.mean(0); sd = Xs.std(0) + 0.5
Z = (Xs - mu) / sd
n = len(Z); cut = int(n * 0.75)
w = np.zeros(Z.shape[1]); b = 0.0; lam = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-2
for it in range(400):   # full-batch gradient descent, L2
    p = 1 / (1 + np.exp(-(Z[:cut] @ w + b)))
    g = Z[:cut].T @ (p - y[:cut]) / cut + lam * w
    w -= 0.5 * g; b -= 0.5 * (p - y[:cut]).mean()
pt = 1 / (1 + np.exp(-(Z[cut:] @ w + b)))
acc = ((pt > 0.5) == y[cut:]).mean()
print(which, 'features', sel.sum(), 'held-out acc', round(acc, 3), 'base rate', round(1 - y[cut:].mean(), 3))
json.dump({'keys': keys[sel].tolist(), 'mu': mu.tolist(), 'sd': sd.tolist(), 'w': w.tolist(), 'b': b},
          open(f'runs/imitate_{which}.json', 'w'))
