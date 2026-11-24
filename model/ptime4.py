"""S10-01 starter workload in playing time (pre-registered in RESEARCH_LOG).

K1 workload features + K2 recency weights on top of the shipped C2 (opening day) and W1 (winter) pitcher models.
usage: python model/ptime4.py -> work/results/s10_01_ptime4.txt, s10_01_preds.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ptime import HFEAT, PFEAT, RES, SEASON, TARGETS, TRAIN0, build_frames  # noqa: E402
from ptime2 import RFEAT, table  # noqa: E402
from ptime3 import WFEAT, add_winter  # noqa: E402

KFEAT = ["gs1", "pit_app", "pit_gs", "maxbf", "jump", "h2020", "lg_spbf"]
DECAY = 0.8


def add_workload(F):
    pit = pd.read_parquet(SEASON / "pit.parquet")
    pit["pro"] = np.where(pit.season == 2020, 162 / 60, 1.0)
    sp = pit[pit.p_gs / pit.p_g >= 0.8]
    lg = sp.groupby("season").apply(lambda g: g.p_tbf.sum() / g.p_gs.sum(), include_groups=False)
    F = F.copy()
    p1 = F[["player_id", "season"]].assign(s1=F.season - 1).merge(
        pit[["player_id", "season", "p_g", "p_gs", "p_pitch", "pro"]].rename(columns={"season": "s1"}), on=["player_id", "s1"], how="left")
    F["gs1"] = (p1.p_gs * p1.pro).fillna(0).values
    F["pit_app"] = (p1.p_pitch / p1.p_g).fillna(0).values
    F["pit_gs"] = np.where(p1.p_gs / p1.p_g >= 0.5, p1.p_pitch / p1.p_g, 0)
    F["pit_gs"] = F.pit_gs.fillna(0)
    F["maxbf"] = F[["bf1", "bf2", "bf3"]].max(axis=1)
    F["jump"] = F.bf1 - F[["bf2", "bf3"]].max(axis=1)
    F["h2020"] = ((F.season - 3 <= 2020) & (F.season - 1 >= 2020)).astype(float)
    F["lg_spbf"] = (F.season - 1).map(lg)
    return F


def fit_w(F, feats, decay=None):
    pred = pd.Series(np.nan, index=F.index)
    for N in TARGETS:
        tr = F[(F.season >= TRAIN0) & (F.season < N) & (F.season != 2020)]
        w = decay ** (N - 1 - tr.season) if decay else None
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)
        m.fit(tr[feats], tr.y, sample_weight=w)
        te = F.season == N
        pred[te] = np.clip(m.predict(F.loc[te, feats]), 0, None)
    return pred


def band_tab(F, arms, L):
    ex = F[(F.season != 2020) & F.season.isin(TARGETS)]
    for lab, m in [("600+ BF N-1", ex.bf1 >= 600), ("300-599", ex.bf1.between(300, 599.99))]:
        s = ex[m]
        L.append(f"  {lab}: n={len(s)} mean y {s.y.mean():.0f}; " + ", ".join(
            f"{a} {(s[a] - s.y).mean():+.1f}/{np.sqrt(((s[a] - s.y) ** 2).mean()):.1f}" for a in arms))
    s = ex[ex.bf1 >= 600]
    by = s.groupby("season").apply(lambda g: pd.Series({a: (g[a] - g.y).mean() for a in arms}), include_groups=False)
    L.append("  600+ mean error by season:\n" + by.round(1).T.to_string())


if __name__ == "__main__":
    H, P = build_frames()
    P = add_workload(add_winter(P))
    L = ["S10-01 starter workload (targets 2016-24 ex-2020; trained on targets 2013..N-1). K3 = K1 features + K2 recency weights."]
    for base, bf in [("c2", PFEAT + RFEAT), ("w1", PFEAT + WFEAT)]:
        P[base] = fit_w(P, bf)
        P[f"{base}_k1"] = fit_w(P, bf + KFEAT)
        P[f"{base}_k2"] = fit_w(P, bf, DECAY)
        P[f"{base}_k3"] = fit_w(P, bf + KFEAT, DECAY)
        arms = [base, f"{base}_k1", f"{base}_k2", f"{base}_k3"]
        table(P, ["b0"] + arms, [base], L, f"PITCHERS (BF), base {base}")
        ex = P[(P.season != 2020) & P.season.isin(TARGETS) & (P.y > 0)]
        L.append("  among those who pitched: " + ", ".join(f"{a} {np.sqrt(((ex[a] - ex.y) ** 2).mean()):.2f}" for a in ["b0"] + arms))
        band_tab(P, arms, L)
    # hitters: K2 recency only, sensitivity
    H = add_winter(H)
    H["c2"] = fit_w(H, HFEAT + RFEAT)
    H["c2_k2"] = fit_w(H, HFEAT + RFEAT, DECAY)
    table(H, ["c2", "c2_k2"], ["c2"], L, "HITTERS (PA) sensitivity: recency weights on C2")
    txt = "\n".join(L)
    print(txt)
    (RES / "s10_01_ptime4.txt").write_text(txt + "\n")
    P.assign(kind="pit").to_parquet(RES / "s10_01_preds.parquet")
