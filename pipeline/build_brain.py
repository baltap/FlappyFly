"""Extract the Flappy Fly brain from the Janelia/Google male CNS connectome.

One fly, one subnetwork: every traced neuron that sits on a short path from
the eye (lamina L1/L2/L3, both eyes) to the descending neurons or the
mushroom body, the full mushroom body learning circuit, and the short paths
from the mushroom body output neurons back to the descending neurons.

Output (web/data/):
  brain.bin   CSR connectivity: indptr u32[n+1], indices u32[m], weight i16[m]
              (signed synapse counts, sign from the predicted transmitter)
  brain.json  neuron metadata, roles, eye geometry and the DAN->MBON
              compartment map used by the plasticity rule

Usage:  uv run python pipeline/build_brain.py [--data DIR] [--min-syn 3] [--hops 3]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("/Users/peter/Data/Amtigravity/FruitFlyBrain/data/cache/male-cns")

# Superclasses that cannot carry the eye -> flight signal in a brain-only model.
DROP_SUPERCLASS = {
    "vnc_intrinsic", "vnc_sensory", "vnc_motor", "ascending_neuron", "sensory_ascending",
    "vnc_efferent", "vnc_endocrine", "vnc_sensory_tbc", "efferent_ascending", "vnc_tbc",
    "sensory_ascending_tbc", "cb_sensory", "ol_sensory", "cb_motor", "cb_endocrine", "ENS",
    "cb_sensory_tbc", "sensory_descending", "efferent_descending", "cb_efferent",
}
EYE_TYPES = ["L1", "L2", "L3"]
MB_RE = re.compile(r"^(KC|MBON|PAM|PPL1|APL|DPM)")
# Fast-transmitter sign. Modulators get 0 here; dopamine acts through plasticity.
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
           "dopamine": 0, "octopamine": 0, "serotonin": 0}
# Wing stroke amplitude / lift (Namiki et al. 2022): the flap readout.
FLAP_DN_RE = re.compile(r"^DNg02")

CLASS_OF_SUPER = {
    "ol_intrinsic": "optic", "visual_projection": "vpn", "visual_centrifugal": "vpn",
    "visual_projection_tbc": "vpn", "cb_intrinsic": "central", "descending_neuron": "dn",
}


def load(data: Path):
    ann = pd.read_feather(data / "body-annotations.feather")
    ann = ann[ann.status == "Traced"]
    ann = ann[~ann.superclass.isin(DROP_SUPERCLASS) & ann.superclass.notna()].reset_index(drop=True)
    nt = pd.read_feather(data / "body-neurotransmitters.feather",
                         columns=["body", "consensus_nt", "celltype_predicted_nt", "predicted_nt"])
    nt = nt[nt.body.isin(ann.bodyId)].drop_duplicates("body").set_index("body")
    edges = feather.read_table(data / "connectome-weights.feather").to_pandas()
    return ann, nt, edges


def reach(start: np.ndarray, M: sp.csr_matrix, hops: int) -> np.ndarray:
    """Hop distance from `start` along M (rows=pre, cols=post); 99 = unreachable."""
    n = M.shape[0]
    dist = np.full(n, 99, dtype=np.int16)
    dist[start] = 0
    seen = start.copy()
    front = start.astype(np.float32)
    for h in range(1, hops + 1):
        nxt = np.asarray(front @ M).ravel() > 0
        nxt &= ~seen
        dist[nxt] = h
        seen |= nxt
        front = nxt.astype(np.float32)
    return dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--min-syn", type=int, default=3)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--strong-frac", type=float, default=0.01,
                    help="an edge counts for path finding when it is >= this fraction of the target's input")
    args = ap.parse_args()

    ann, nt, edges = load(args.data)
    print(f"candidates {len(ann)}; raw edges {len(edges)}")
    idx = pd.Series(np.arange(len(ann)), index=ann.bodyId)
    edges = edges[(edges.weight >= args.min_syn)
                  & edges.body_pre.isin(idx.index) & edges.body_post.isin(idx.index)]
    pre = idx[edges.body_pre].to_numpy()
    post = idx[edges.body_post].to_numpy()
    w = edges.weight.to_numpy().astype(np.float32)
    n = len(ann)
    A = sp.csr_matrix((w, (pre, post)), shape=(n, n))
    insum = np.asarray(A.sum(0)).ravel()
    S = (A.multiply(1 / np.maximum(insum, 1)[None, :]) >= args.strong_frac).astype(np.float32).tocsr()
    ST = S.T.tocsr()

    typ = ann.type.fillna("").astype(str)
    is_eye = typ.isin(EYE_TYPES).to_numpy()
    is_dn = (ann.superclass == "descending_neuron").to_numpy()
    is_mb = typ.str.match(MB_RE).to_numpy()
    is_kc = typ.str.startswith("KC").to_numpy()
    is_mbon = typ.str.startswith("MBON").to_numpy()

    d_eye = reach(is_eye, S, 6)
    d_to_out = reach(is_dn | is_mb, ST, args.hops)
    d_mbon = reach(is_mbon, S, args.hops)
    d_to_dn = reach(is_dn, ST, args.hops)
    k = args.hops
    keep = (d_eye <= k) & (d_to_out <= k)                     # eye -> DN / MB paths
    keep |= (d_mbon <= k) & (d_to_dn <= k)                    # MB output -> DN paths
    keep |= is_mb & ~is_kc                                     # MBONs, DANs, APL, DPM
    keep |= is_kc & (d_eye <= k + 1)                           # visually driven KCs
    keep |= is_dn & ((d_eye <= k + 2) | (d_mbon <= k + 1))     # reachable DNs
    keep |= is_eye

    # Prune dead ends: neurons with no input from the set (except sources) or no output
    # into the set (except sinks), until stable.
    source = is_eye | typ.str.match(r"^(PAM|PPL1)").to_numpy()   # DANs get reward input
    sink = is_dn | is_mbon
    for _ in range(20):
        kk = keep.astype(np.float32)
        has_in = np.asarray(kk @ A).ravel() > 0
        has_out = np.asarray(A @ kk).ravel() > 0
        new = keep & (has_in | source) & (has_out | sink)
        if new.sum() == keep.sum():
            break
        keep = new
    sel = np.flatnonzero(keep)
    print(f"kept {len(sel)} neurons")

    sub = ann.iloc[sel].reset_index(drop=True)
    B = A[sel][:, sel].tocsr()
    styp = sub.type.fillna("").astype(str)

    # Transmitter sign per neuron.
    def nt_of(body):
        if body not in nt.index:
            return "unclear"
        r = nt.loc[body]
        for c in ("consensus_nt", "celltype_predicted_nt", "predicted_nt"):
            if isinstance(r[c], str) and r[c] != "unclear":
                return r[c]
        return "unclear"
    nts = [nt_of(b) for b in sub.bodyId]
    sign = np.array([NT_SIGN.get(x, 1) for x in nts], dtype=np.int8)

    # Drop known axo-axonal contact classes that are not fast synaptic drive
    # (KC-KC, KC->DAN, DAN->KC): they create runaway loops in point-neuron models.
    s_kc = styp.str.startswith("KC").to_numpy()
    s_dan = styp.str.match(r"^(PAM|PPL1)").to_numpy()
    Bc = B.tocoo()
    r, c, v = Bc.row, Bc.col, Bc.data
    kc_mbon_mask = s_kc[r] & styp.str.startswith("MBON").to_numpy()[c]
    # DAN -> MBON compartment map through shared KCs: DAN d modulates the KC inputs
    # of MBON m in proportion to sum_k w(d->k) w(k->m).
    DK = B[np.flatnonzero(s_dan)][:, np.flatnonzero(s_kc)]
    KM = B[np.flatnonzero(s_kc)][:, np.flatnonzero(styp.str.startswith("MBON").to_numpy())]
    DM = (DK @ KM).toarray()
    DM = DM / np.maximum(DM.max(1, keepdims=True), 1e-9)

    bad = (s_kc[r] & s_kc[c]) | (s_kc[r] & s_dan[c]) | (s_dan[r] & s_kc[c])
    signed = v * sign[r]
    ok = ~bad & (signed != 0)
    r, c, signed = r[ok], c[ok], signed[ok]
    order = np.lexsort((c, r))
    r, c, signed = r[order], c[order], signed[order]
    ns = len(sub)
    indptr = np.zeros(ns + 1, dtype=np.uint32)
    np.add.at(indptr, r + 1, 1)
    indptr = np.cumsum(indptr).astype(np.uint32)
    wts = np.clip(signed, -32767, 32767).astype(np.int16)
    print(f"edges {len(c)}  (+{int((wts > 0).sum())} / -{int((wts < 0).sum())}); "
          f"KC->MBON edges {int(kc_mbon_mask.sum())}")

    classes_pre = [("eye" if t in EYE_TYPES else CLASS_OF_SUPER.get(sc, "central")) for t, sc in zip(styp, sub.superclass)]

    # ---- eye geometry: hex column -> viewing direction -------------------------------
    # Lamina somata fit (see docs/DESIGN.md): hex1+hex2 runs ventral->dorsal,
    # hex2-hex1 runs anterior->posterior, on both eyes.
    eye_rows = []
    for side in ("R", "L"):
        m = styp.isin(EYE_TYPES).to_numpy() & (sub.somaSide == side).to_numpy() & sub.assignedOlHex1.notna().to_numpy()
        h1 = sub.assignedOlHex1.to_numpy()[m]
        h2 = sub.assignedOlHex2.to_numpy()[m]
        u = h1 + h2
        vv = h2 - h1
        ulo, uhi = np.percentile(u, [1, 99])
        vlo, vhi = np.percentile(vv, [1, 99])
        el = -70 + 140 * (u - ulo) / (uhi - ulo)          # degrees, + = up
        az = -15 + 175 * (vv - vlo) / (vhi - vlo)         # degrees from straight ahead, + = lateral/back
        for i, e, a_, t in zip(np.flatnonzero(m), el, az, styp.to_numpy()[m]):
            eye_rows.append([int(i), side, t, round(float(np.clip(a_, -20, 170)), 2),
                             round(float(np.clip(e, -80, 80)), 2)])
    print(f"eye inputs {len(eye_rows)}")

    # Viewing direction for every hex-assigned columnar neuron (same per-eye mapping).
    azel = [None] * ns
    for side in ("R", "L"):
        m = styp.isin(EYE_TYPES).to_numpy() & (sub.somaSide == side).to_numpy() & sub.assignedOlHex1.notna().to_numpy()
        u = (sub.assignedOlHex1 + sub.assignedOlHex2).to_numpy()[m]
        vv = (sub.assignedOlHex2 - sub.assignedOlHex1).to_numpy()[m]
        ulo, uhi = np.percentile(u, [1, 99]); vlo, vhi = np.percentile(vv, [1, 99])
        inst_side = sub.instance.fillna("").astype(str).str.extract(r"_([RL])$")[0].fillna(sub.somaSide.fillna("?"))
        mm = sub.assignedOlHex1.notna().to_numpy() & (inst_side == side).to_numpy()
        for i in np.flatnonzero(mm):
            uu = sub.assignedOlHex1[i] + sub.assignedOlHex2[i]
            v2 = sub.assignedOlHex2[i] - sub.assignedOlHex1[i]
            azel[i] = [round(float(np.clip(-15 + 175 * (v2 - vlo) / (vhi - vlo), -20, 170)), 1),
                       round(float(np.clip(-70 + 140 * (uu - ulo) / (uhi - ulo), -80, 80)), 1)]
    print(f"columnar neurons with a viewing direction: {sum(a is not None for a in azel)}")

    # Receptive-field centres for the other visual neurons: synapse-weighted mean of the
    # viewing directions of their excitatory inputs, propagated a few synapses deep.
    # Neurons pooling over a wide field keep a large spread and are treated as wide-field.
    Bpos = B.multiply(sign[:, None] > 0).tocsc() if False else None
    exc = sp.csr_matrix((np.where(sign[B.tocoo().row] > 0, B.tocoo().data, 0), (B.tocoo().row, B.tocoo().col)), shape=B.shape).tocsc()
    AZ = np.array([a[0] if a else np.nan for a in azel]); EL = np.array([a[1] if a else np.nan for a in azel])
    SP = np.where(np.isnan(AZ), np.nan, 3.0)
    visual = np.isin(np.array(classes_pre), ["optic", "vpn", "eye"]) | sub.superclass.isin(["visual_projection", "visual_centrifugal"]).to_numpy()
    for it in range(6):
        known = ~np.isnan(AZ)
        kw = exc.multiply(known[:, None]).tocsc()
        tot = np.asarray(exc.sum(0)).ravel()
        kn = np.asarray(kw.sum(0)).ravel()
        az0 = np.nan_to_num(AZ); el0 = np.nan_to_num(EL)
        maz = np.asarray(kw.T @ az0).ravel() / np.maximum(kn, 1e-9)
        mel = np.asarray(kw.T @ el0).ravel() / np.maximum(kn, 1e-9)
        m2 = np.asarray(kw.T @ (az0 ** 2 + el0 ** 2)).ravel() / np.maximum(kn, 1e-9)
        spread = np.sqrt(np.maximum(m2 - maz ** 2 - mel ** 2, 0))
        new = ~known & visual & (kn >= 0.5 * np.maximum(tot, 1)) & (kn > 0)
        AZ[new] = maz[new]; EL[new] = mel[new]; SP[new] = spread[new]
        print(f"  rf pass {it}: +{int(new.sum())}")
    rf = [None if np.isnan(AZ[i]) else [round(float(AZ[i]), 1), round(float(EL[i]), 1), round(float(SP[i]), 1)] for i in range(ns)]

    # ---- positions for the brain view (frontal projection, x = fly's right on the left) --
    pos = np.full((ns, 3), np.nan)
    has_soma = sub.somaLocation.notna().to_numpy()
    pos[has_soma] = np.vstack(sub.somaLocation[has_soma].values).astype(float)
    # Columnar optic lobe cells without somata: place from their hex column.
    layer_x = {"L": 0.0, "Mi": 0.35, "Tm": 0.5, "T4": 0.62, "T5": 0.7, "C": 0.3, "T1": 0.4, "T2": 0.55, "T3": 0.6,
               "Dm": 0.3, "Pm": 0.4, "Lawf": 0.2}
    hexm = sub.assignedOlHex1.notna().to_numpy() & ~has_soma
    for i in np.flatnonzero(hexm):
        h1, h2 = sub.assignedOlHex1[i], sub.assignedOlHex2[i]
        side = sub.somaSide[i] if isinstance(sub.somaSide[i], str) else ("R" if str(sub.instance[i]).endswith("_R") else "L")
        t = styp[i]
        depth = next((vx for kx, vx in layer_x.items() if t.startswith(kx)), 0.5)
        sx = -1 if side == "R" else 1
        x = 48500 + sx * (38000 - depth * 22000) + sx * (h2 - h1) * 180
        y = 59000 - (h1 + h2) * 560 * 0.9
        pos[i] = [x, y, 26000 + (h2 - h1) * 300]
    # Everything else: mean of the same type, then of the same superclass+side.
    df = pd.DataFrame(pos, columns=list("xyz"))
    df["type"] = styp
    df["grp"] = sub.superclass.astype(str) + "|" + sub.somaSide.fillna("?").astype(str)
    for key in ("type", "grp"):
        means = df.dropna().groupby(key)[["x", "y", "z"]].mean()
        miss = df.x.isna()
        df.loc[miss, ["x", "y", "z"]] = means.reindex(df.loc[miss, key]).to_numpy()
    rng = np.random.default_rng(7)
    miss = df.x.isna()
    df.loc[miss, ["x", "y", "z"]] = [48500, 25000, 26000]
    jitter = rng.normal(0, 700, size=(ns, 3)) * (~has_soma)[:, None]
    P = df[["x", "y", "z"]].to_numpy() + jitter

    # ---- metadata ---------------------------------------------------------------------
    types = sorted(set(styp))
    tix = {t: i for i, t in enumerate(types)}

    def klass(i):
        t = styp[i]
        if t in EYE_TYPES:
            return "eye"
        if t.startswith("KC"):
            return "kc"
        if t.startswith("MBON"):
            return "mbon"
        if re.match(r"^(PAM|PPL1)", t):
            return "dan"
        if t in ("APL", "DPM"):
            return "mbother"
        return CLASS_OF_SUPER.get(sub.superclass[i], "central")

    classes = [klass(i) for i in range(ns)]
    inst_side = sub.instance.fillna("").astype(str).str.extract(r"_([RL])$")[0]
    side = [s if isinstance(s, str) else (x if isinstance(x, str) else "?") for s, x in zip(sub.somaSide, inst_side)]
    roles = {
        "eye": eye_rows,
        "flapDN": [int(i) for i in np.flatnonzero(styp.str.match(FLAP_DN_RE).to_numpy())],
        "dn": [int(i) for i in np.flatnonzero(styp.map(lambda t: t.startswith("DN") or t in ("MDN",)).to_numpy() | (sub.superclass == "descending_neuron").to_numpy())],
        "kc": [int(i) for i in np.flatnonzero(s_kc)],
        "mbon": [int(i) for i in np.flatnonzero(styp.str.startswith("MBON").to_numpy())],
        "pam": [int(i) for i in np.flatnonzero(styp.str.startswith("PAM").to_numpy())],
        "ppl1": [int(i) for i in np.flatnonzero(styp.str.startswith("PPL1").to_numpy())],
        "apl": [int(i) for i in np.flatnonzero((styp == "APL").to_numpy())],
        "vs": [int(i) for i in np.flatnonzero(styp.str.match(r"^VS").to_numpy())],
        "hs": [int(i) for i in np.flatnonzero(styp.str.match(r"^HS").to_numpy())],
        "lplc2": [int(i) for i in np.flatnonzero((styp == "LPLC2").to_numpy())],
    }
    dan_idx = np.flatnonzero(s_dan)
    meta = {
        "source": "Janelia/Google male CNS connectome (male-cns v0.9, Traced bodies)",
        "params": {"min_syn": args.min_syn, "hops": args.hops, "strong_frac": args.strong_frac},
        "n": ns, "m": int(len(c)),
        "types": types,
        "type": [tix[t] for t in styp],
        "class": classes,
        "side": side,
        "nt": nts,
        "bodyId": [int(b) for b in sub.bodyId],
        "pos": np.round(P / 100).astype(int).tolist(),       # 0.8 um units / 100 -> compact
        "roles": roles,
        "rf": rf,
        # dan x mbon compartment weights, rows follow roles.pam + roles.ppl1 order in dan_order
        "danOrder": [int(i) for i in dan_idx],
        "danMbon": np.round(DM, 3).tolist(),
    }
    out = ROOT / "web" / "data"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "brain.bin", "wb") as fh:
        fh.write(np.array([ns, len(c)], dtype=np.uint32).tobytes())
        fh.write(indptr.tobytes())
        fh.write(c.astype(np.uint32).tobytes())
        fh.write(wts.tobytes())
    with open(out / "brain.json", "w") as fh:
        json.dump(meta, fh, separators=(",", ":"))
    cls = pd.Series(classes).value_counts().to_dict()
    print("classes", cls)
    print("flap DNs", len(roles["flapDN"]), "DNs", len(roles["dn"]), "KCs", len(roles["kc"]),
          "MBONs", len(roles["mbon"]), "PAM", len(roles["pam"]), "PPL1", len(roles["ppl1"]))
    print("wrote", out / "brain.bin", (out / "brain.bin").stat().st_size // 1024, "KB;",
          (out / "brain.json").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
