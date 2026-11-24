"""S7-02: pitcher plate discipline + stuff on top of shipped S6-01 C2. See RESEARCH_LOG (pre-registered).
needs model/stuff_build.py outputs. usage: python model/stuff.py | tee work/results/s7_02_stuff.txt
Stuff score for season s = out-of-fold P(whiff|swing) from the model trained on seasons <= s (stuff_oof N = s+1),
so it never uses seasons after s.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcomp  # noqa: E402
pcomp.EB_SHIP = "binom"  # this test was run against the S6-01 constants
from marcel import wrmse  # noqa: E402
from park_test import clustered_t, multipliers, run as park_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEASON = ROOT / "data" / "season"
A = 0.3
FS = {"B1": ["wh", "cs", "fo", "ba"], "B2": ["wh", "cs", "fo", "ba", "stuff"], "B3": ["stuff", "zn"]}
_eb_orig = pcomp.eb_k
STATE = {}


def cov_k(d, col, statcast_only):
    q = d[(d.p_tbf >= 100) & ((d.season >= 2015) if statcast_only else True)][["player_id", "season", "p_tbf", col]].copy()
    q["r"] = q[col] / q.p_tbf
    q["r"] -= q.groupby("season").r.transform(lambda x: np.average(x, weights=q.loc[x.index, "p_tbf"]))
    p = q.merge(q.assign(season=q.season - 1), on=["player_id", "season"], suffixes=("", "_n"))
    p = p[(p.season != 2019) & (p.season != 2020)] if len(p[(p.season != 2019) & (p.season != 2020)]) > 50 else p
    if len(p) < 50:
        return None
    w = 2 / (1 / p.p_tbf + 1 / p.p_tbf_n)
    tau2 = max(np.average(p.r * p.r_n, weights=w), 1e-9)
    obs = np.average(q.r ** 2, weights=q.p_tbf)
    einv = np.average(1 / q.p_tbf, weights=q.p_tbf)
    vpb = max(obs - tau2, 1e-9) / einv
    return vpb / tau2


def eb_patch(d, col, var):
    mode = STATE.get("mode")
    if mode is None or col not in ("k", "ubb"):
        return _eb_orig(d, col, var)
    if mode == "cov_act":
        return cov_k(d, col, False)
    k = cov_k(d, col, True)
    if k is None:  # N=2016: no Statcast pairs yet; use the actual-rate covariance constant
        k = cov_k(d.assign(**{col: d[col + "_act"]}), col, False)
    STATE.setdefault("ks", []).append((STATE["N"], STATE["arm"], col, round(k)))
    return k


pcomp.eb_k = eb_patch


def expected(d, N, feats):
    """xK/BF and xuBB/BF from a BF-weighted linear fit on Statcast pitcher-seasons < N (BF >= 100)."""
    m = d.season.between(2015, N - 1) & d.n.notna() & d[feats].notna().all(axis=1)
    tr = m & (d.p_tbf >= 100)
    X = np.c_[np.ones(m.sum()), d.loc[m, feats].values.astype(float)]
    Xt = np.c_[np.ones(tr.sum()), d.loc[tr, feats].values.astype(float)]
    w = np.sqrt(d.loc[tr, "p_tbf"].values)
    out = {}
    for c in ("k", "ubb"):
        y = (d.loc[tr, c + "_act"] / d.loc[tr, "p_tbf"]).values.astype(float)
        b = np.linalg.lstsq(Xt * w[:, None], y * w, rcond=None)[0]
        out[c] = pd.Series(np.clip(X @ b, 0, 1), index=d.index[m])
        r2 = 1 - np.average((y - Xt @ b) ** 2, weights=w ** 2) / np.average((y - np.average(y, weights=w ** 2)) ** 2, weights=w ** 2)
        STATE.setdefault("r2", []).append((N, tuple(feats), c, round(r2, 3)))
    return out


def run():
    pit = pd.read_parquet(SEASON / "pit.parquet")
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    mix = pd.read_parquet(SEASON / "park_mix.parquet")
    pf = pd.read_parquet(SEASON / "park_factors.parquet")
    pcon, pmom = pcomp.load_pcon(rv)
    _, P = park_run()
    a, rp = pcomp.ARM_SHIP
    d0 = pcomp.seasons_frame(pit, rv, pcon, a)
    pdc = pd.read_parquet(SEASON / "pdisc.parquet")
    for c in ["wh", "cs", "fo", "ba", "zn"]:
        pdc[c] = pdc[c].astype(float) / pdc.n
    so = pd.read_parquet(SEASON / "stuff_oof.parquet")
    so = so[so.N == so.season + 1][["mlbam", "season", "stuff"]]
    d0 = d0.merge(pdc[["mlbam", "season", "n", "wh", "cs", "fo", "ba", "zn"]], on=["mlbam", "season"], how="left")
    d0 = d0.merge(so, on=["mlbam", "season"], how="left")
    d0["k"], d0["ubb"] = d0.k.astype(float), d0.ubb.astype(float)
    d0["k_act"], d0["ubb_act"] = d0.k, d0.ubb
    rows = []
    for N in pcomp.TARGETS:
        mp = multipliers(mix, pf, N, "pit", range(N - 3, N + 1))
        tg = P[P.season == N][["player_id", "p_tbf", "ra9_rel", "season"]].copy()
        STATE.update(mode=None, N=N, arm="B0")
        pr, _ = pcomp.project(d0, rv, pmom, N, a, mp, role_prior=rp)
        tg["B0"] = tg.player_id.map(pr.proj).fillna(0.0)
        STATE.update(mode="cov_act", arm="B1c")
        pr, _ = pcomp.project(d0, rv, pmom, N, a, mp, role_prior=rp)
        tg["B1c"] = tg.player_id.map(pr.proj).fillna(0.0)
        for arm, feats in FS.items():
            if arm != "B1" and N == 2016:  # stuff for 2015 needs stuff_oof N=2016: present
                pass
            d = d0.copy()
            x = expected(d, N, feats)
            for c in ("k", "ubb"):
                i = x[c].index
                d.loc[i, c] = A * d.loc[i, c + "_act"] + (1 - A) * x[c] * d.loc[i, "p_tbf"]
            STATE.update(mode="blend", arm=arm)
            pr, _ = pcomp.project(d, rv, pmom, N, a, mp, role_prior=rp)
            tg[arm] = tg.player_id.map(pr.proj).fillna(0.0)
        rows.append(tg)
        print("done", N, flush=True)
    R = pd.concat(rows)
    R.to_parquet(ROOT / "work" / "results" / "s7_02_preds.parquet", index=False)
    arms = ["B0", "B1c", "B1", "B2", "B3"]
    x = R[R.season != 2020]
    e1, e2 = x.season <= 2019, x.season >= 2021
    print("\nfit R2 of xK / xuBB (pitcher-seasons BF>=100):")
    r2 = pd.DataFrame(STATE["r2"], columns=["N", "feats", "comp", "r2"]).drop_duplicates()
    print(r2.groupby(["feats", "comp"]).r2.agg(["min", "max"]).to_string())
    print("\nEB constants (raw BF) for blended K/uBB:")
    print(pd.DataFrame(STATE["ks"], columns=["N", "arm", "comp", "k"]).pivot_table(index="N", columns=["arm", "comp"], values="k").to_string())
    print("\n| arm | pooled | t vs B0 | 2016-19 (t) | 2021-24 (t) | slope |")
    for c in arms:
        def t(m):
            return clustered_t(None, None, x[m].ra9_rel - x[m].B0, x[m].ra9_rel - x[m][c], x[m].p_tbf, x[m].season)
        allm = x.season > 0
        sl = np.polyfit(x[c], x.ra9_rel, 1, w=np.sqrt(x.p_tbf))[0]
        print(f"| {c} | {wrmse(x.ra9_rel - x[c], x.p_tbf):.4f} | {t(allm) if c != 'B0' else 0:.2f} | "
              f"{wrmse(x[e1].ra9_rel - x[e1][c], x[e1].p_tbf):.4f} ({t(e1) if c != 'B0' else 0:.2f}) | "
              f"{wrmse(x[e2].ra9_rel - x[e2][c], x[e2].p_tbf):.4f} ({t(e2) if c != 'B0' else 0:.2f}) | {sl:.2f} |")
    print("  t vs B1c: " + "  ".join(f"{c} {clustered_t(None, None, x.ra9_rel - x.B1c, x.ra9_rel - x[c], x.p_tbf, x.season):.2f}" for c in ["B1", "B2", "B3"]))
    print("  t B2 vs B1: " + f"{clustered_t(None, None, x.ra9_rel - x.B1, x.ra9_rel - x.B2, x.p_tbf, x.season):.2f}")
    print("\nby season:")
    print(R.groupby("season").apply(lambda g: pd.Series({c: wrmse(g.ra9_rel - g[c], g.p_tbf) for c in arms}), include_groups=False).round(4).to_string())


if __name__ == "__main__":
    run()
