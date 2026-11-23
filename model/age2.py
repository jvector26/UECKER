"""S5-02: measured (delta-method) hitter age curve with job-retention and regression fixes. See RESEARCH_LOG.
usage: python model/age2.py   (tee to work/results/s5_02_age2.txt)
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import age  # noqa: E402
from age import AGES, chain, smooth  # noqa: E402
from marcel import TUNE, wrmse  # noqa: E402
from park_test import clustered_t  # noqa: E402

R_ONE = 240.0   # Marcel's single-season regression (5 x PA vs 1200)
CAP = 5.0
MIN_PA = 100


def logit_fit(X, y, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        b = b + np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(len(b)), X.T @ (y - p))
    return b


def design(d):
    bins = pd.cut(d.age, [0, 25, 28, 31, 34, 99], labels=False)
    X = np.column_stack([np.ones(len(d))] + [(bins == k).astype(float) for k in range(1, 5)] +
                        [d.woba_rel.values / 0.03, np.log(d.woba_den.clip(lower=1)).values])
    return X


def curves(bat, N):
    b = bat[~(bat.p_g > 0.5 * bat.b_g) & (bat.season < N)][["player_id", "season", "age", "woba_rel", "woba_den", "b_pa"]]
    y = b[(b.b_pa >= MIN_PA) & (b.season < N - 1)]
    nxt = b[["player_id", "season", "woba_rel", "woba_den", "b_pa"]].assign(season=b.season - 1)
    d = y.merge(nxt, on=["player_id", "season"], how="left", suffixes=("", "_n"))
    d["ret"] = (d.b_pa_n.fillna(0) >= MIN_PA).astype(float)
    X = design(d)
    beta = logit_fit(X, d.ret.values)
    d["pret"] = 1 / (1 + np.exp(-X @ beta))
    p = d[d.ret == 1].copy()
    hm = 2 / (1 / p.woba_den + 1 / p.woba_den_n)
    ipw = np.minimum(1 / p.pret, CAP)
    shrunk = p.woba_rel * p.woba_den / (p.woba_den + R_ONE)
    a = np.clip(np.floor(p.age), AGES[0], AGES[-1]).astype(int)
    out = {}
    for k, w, y0 in [("D0", hm, p.woba_rel), ("D1", hm * ipw, p.woba_rel), ("D2", hm, shrunk), ("D3", hm * ipw, shrunk)]:
        out[k] = smooth((w * (p.woba_rel_n - y0)).groupby(a).sum(), w.groupby(a).sum())
    return out, d


def run():
    with contextlib.redirect_stdout(io.StringIO()):
        h = age.run()
    bat = pd.read_parquet(age.SEASON / "bat.parquet")
    hh = h.has_hist
    for N in TUNE:
        cv, d = curves(bat, N)
        i = h.season == N
        for k, c in cv.items():
            h.loc[i, f"p_{k}"] = h.pred0[i] + np.where(hh[i], chain(c, h.age_last[i].fillna(0).values,
                                                                     (h.age_last + h.gap)[i].fillna(0).values), 0.0)
        if N in (2016, 2024):
            print(f"{N} return rate by age band:", d.groupby(pd.cut(d.age, [0, 25, 28, 31, 34, 99]), observed=True).ret.mean().round(2).tolist())
            for k, c in cv.items():
                print(f"   {k} per-year change at 22..38 by 2:", " ".join(f"{a}:{c[a]:+.4f}" for a in range(22, 39, 2)),
                      f"| cum 27->33 {c.loc[27:32].sum():+.4f}")

    arms = ["ship", "A1", "A2", "D0", "D1", "D2", "D3"]
    e = h[h.season != 2020]
    subsets = [("2016-24 ex-2020", e.season > 0), ("2016-19", e.season <= 2019), ("2021-24", e.season >= 2021)]
    print("\n| subset | " + " | ".join(arms) + " |")
    for name, m in subsets:
        x = e[m]
        print(f"| {name} | " + " | ".join(f"{wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.5f}" for k in arms) + " |")
    print("\nseason-clustered t vs shipped:")
    for name, m in subsets:
        x = e[m]
        print(f"  {name}: " + "  ".join(f"{k} {clustered_t(None, None, x.woba_rel - x.p_ship, x.woba_rel - x[f'p_{k}'], x.b_pa, x.season):.2f}"
                                       for k in arms[1:]))
    print("  vs A1: " + "  ".join(f"{k} {clustered_t(None, None, e.woba_rel - e.p_A1, e.woba_rel - e[f'p_{k}'], e.b_pa, e.season):.2f}"
                                  for k in ["D0", "D1", "D2", "D3"]))
    print("\nby season:")
    for N in TUNE:
        x = h[h.season == N]
        print(N, " ".join(f"{k} {wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.4f}" for k in arms))
    e = e.assign(band=pd.cut(e.page, [0, 25, 28, 31, 34, 60], labels=["<25", "25-27", "28-30", "31-33", "34+"]))
    print("\nmean error (actual - proj) by projected age band:")
    for b, x in e.groupby("band", observed=True):
        print(f"  {b:6s} n={len(x):4d}: " + "  ".join(f"{k} {np.average(x.woba_rel - x[f'p_{k}'], weights=x.b_pa):+.4f}" for k in arms))
    print("  overall:          " + "  ".join(f"{k} {np.average(e.woba_rel - e[f'p_{k}'], weights=e.b_pa):+.4f}" for k in arms))
    x = e[e.page >= 32]
    print("  32+:              " + "  ".join(f"{k} {np.average(x.woba_rel - x[f'p_{k}'], weights=x.b_pa):+.4f}" for k in arms))
    return h


if __name__ == "__main__":
    run()
