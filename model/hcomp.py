"""S8-01: hitter component projection (K, uBB, HBP, contact value per PA), each component regressed by its own constant
from same-hitter year-to-year covariance; shipped S4-03 contact blend and S5-01 age handling. See RESEARCH_LOG (pre-registered).
usage: python model/hcomp.py | tee work/results/s8_01_hcomp.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import marcel  # noqa: E402
from age import AGES, smooth  # noqa: E402
from age import build as age_build  # noqa: E402
from age import W_SHIP  # noqa: E402
from marcel import TUNE, wrmse  # noqa: E402
from park_test import clustered_t  # noqa: E402
from xcon3 import A_AIR_SHIP, A_GB_SHIP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEASON = ROOT / "data" / "season"
WTS = {1: 1.0, 2: 0.8, 3: 0.6}  # Marcel hitters 5/4/3, normalized so N-1 = 1 (k in raw PA)
K_MARCEL = 1200.0 / 5  # Marcel's single constant in the same units
PA_COMPS = ["k", "ubb", "hbp", "con"]


def seasons_frame(bat, rv, xc):
    d = bat[bat.season <= 2024].copy()
    d["den"] = d.woba_den.astype(float)
    d["k"], d["ubb"], d["hbp"] = d.b_so.astype(float), d.b_ubb.astype(float), d.b_hp.astype(float)
    w = rv.loc[d.season]
    d["con_act"] = (d.b_1b.values * w.w_1b.values + d.b_2b.values * w.w_2b.values + d.b_3b.values * w.w_3b.values
                    + d.b_hr.values * w.w_hr.values)
    x = xc[["player_id", "season", "act_gb", "act_air", "xm_gb", "x_air"]].groupby(["player_id", "season"]).sum().reset_index()
    d = d.merge(x, on=["player_id", "season"], how="left")
    has = d.act_gb.notna()
    d["con"] = d.con_act
    d.loc[has, "con"] = (d.con_act[has] - (1 - A_GB_SHIP) * (d.act_gb[has] - d.xm_gb[has])
                         - (1 - A_AIR_SHIP) * (d.act_air[has] - d.x_air[has]))
    d["bip"] = (d.den - d.k - d.ubb - d.hbp).clip(lower=0)
    d["pit"] = d.p_g > 0.5 * d.b_g
    return d


def cov_k(d, col, den="den"):
    """k (raw units of den) = sampling variance per unit / true variance; true variance = den-weighted lag-1 covariance of
    season-centered rates for the same hitter (both seasons PA >= 100, pitcher-hitters out, 2019->20 / 20->21 dropped)."""
    q = d[(d.den >= 100) & ~d.pit & (d[den] > 0)][["player_id", "season", den, col]].copy()
    q = q.rename(columns={den: "n"})
    q["r"] = q[col] / q.n
    q["r"] -= q.groupby("season").r.transform(lambda x: np.average(x, weights=q.loc[x.index, "n"]))
    p = q.merge(q.assign(season=q.season - 1), on=["player_id", "season"], suffixes=("", "_n"))
    p = p[~p.season.isin([2019, 2020])]
    w = 2 / (1 / p.n + 1 / p.n_n)
    tau2 = max(np.average(p.r * p.r_n, weights=w), 1e-9)
    obs = np.average(q.r ** 2, weights=q.n)
    vpb = max(obs - tau2, 1e-9) / np.average(1 / q.n, weights=q.n)
    return vpb / tau2


def project(d, rv, N, mode):
    """No-age rel-wOBA projection. mode: 'HC' (one Marcel constant), 'H1' (cov k, contact per PA), 'H2' (cov k, contact per BIP)."""
    past = d[d.season < N]
    lg = past.groupby("season")[["den", "k", "ubb", "hbp", "con", "bip"]].sum()
    lgr = lg[PA_COMPS].div(lg.den, axis=0)
    lgcb = lg.con / lg.bip
    if mode == "HC":
        ks = {c: K_MARCEL for c in PA_COMPS}
    else:
        ks = {c: cov_k(past, c) for c in PA_COMPS}
        ks["con_bip"] = cov_k(past, "con", den="bip")
    hist = d[d.season.between(N - 3, N - 1)].copy()
    hist["w"] = (N - hist.season).map(WTS)
    hist["wden"] = hist.w * hist.den
    g = hist.player_id
    wden = hist.wden.groupby(g).sum()
    wN1 = rv.loc[N - 1]
    val = {"k": 0.0, "ubb": wN1.w_bb, "hbp": wN1.w_hbp, "con": 1.0}
    r, lgw = {}, {}
    for c in PA_COMPS:
        pl = (hist.w * hist[c]).groupby(g).sum()
        lgw[c] = (hist.wden * hist.season.map(lgr[c])).groupby(g).sum() / wden
        r[c] = (pl + ks[c] * lgw[c]) / (wden + ks[c])
    if mode == "H2":
        wbip = (hist.w * hist.bip).groupby(g).sum()
        lgb = ((hist.w * hist.bip * hist.season.map(lgcb)).groupby(g).sum() / wbip).fillna(lgcb.loc[N - 1])
        cb = ((hist.w * hist.con).groupby(g).sum() + ks["con_bip"] * lgb) / (wbip + ks["con_bip"])
        r["con"] = (1 - r["k"] - r["ubb"] - r["hbp"]) * cb
    rel = sum((r[c] - lgw[c]) * val[c] for c in PA_COMPS)
    return rel[wden > 0], ks


def age_ship_loso(h, col):
    """Shipped S5-01 handling with the A3 curve refit LOSO on this arm's own no-age residuals."""
    hh, ex = h.has_hist, h.season != 2020
    ia = np.clip(np.floor(h.page), AGES[0], AGES[-1]).astype(int)
    af = np.where(hh, marcel.age_factor(h.page.values), 1.0)
    marcel_p = af * (h[col] + h.lg) - h.lg
    a3 = pd.Series(np.nan, index=h.index)
    r = h.woba_rel - h[col]
    for N in TUNE:
        tr = ex & (h.season != N) & hh
        i = h.season == N
        cur = smooth((h.b_pa * r)[tr].groupby(ia[tr]).sum(), h.b_pa[tr].groupby(ia[tr]).sum())
        a3[i] = h[col][i] + np.where(hh[i], ia[i].map(cur).values, 0.0)
    return (1 - W_SHIP) * marcel_p + W_SHIP * a3


def run():
    bat = pd.read_parquet(SEASON / "bat.parquet")
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    xc = pd.read_parquet(SEASON / "xcon3.parquet")
    h, _ = age_build()
    d = seasons_frame(bat, rv, xc)
    kl = []
    for mode in ["HC", "H1", "H2"]:
        h[f"pred0_{mode}"] = np.nan
        for N in TUNE:
            pr, ks = project(d, rv, N, mode)
            i = h.season == N
            h.loc[i, f"pred0_{mode}"] = h[i].player_id.map(pr).fillna(0.0).values
            if mode != "HC":
                kl.append(dict(season=N, arm=mode, **{c: round(v) for c, v in ks.items()}))
    h["p_H0"] = age_ship_loso(h, "pred0")
    for mode in ["HC", "H1", "H2"]:
        h[f"p_{mode}"] = age_ship_loso(h, f"pred0_{mode}")
    print("covariance regression constants (raw PA; con_bip in BIP), by target season (fit on seasons < N):")
    print(pd.DataFrame(kl)[lambda x: x.arm == "H1"].drop(columns="arm").to_string(index=False))
    print(f"Marcel single constant in the same units: {K_MARCEL:.0f}")

    arms = ["H0", "HC", "H1", "H2"]
    e = h[h.season != 2020]
    e1, e2 = e.season <= 2019, e.season >= 2021

    def t(base, k, m=None):
        x = e if m is None else e[m]
        return clustered_t(None, None, x.woba_rel - x[f"p_{base}"], x.woba_rel - x[f"p_{k}"], x.b_pa, x.season)

    print("\nrel-wOBA PA-weighted RMSE, targets 2016-24 ex-2020")
    print("| arm | pooled | t vs H0 | 2016-19 (t) | 2021-24 (t) | slope | no-age pred0 |")
    for k in arms:
        sl = np.polyfit(e[f"p_{k}"], e.woba_rel, 1, w=np.sqrt(e.b_pa))[0]
        c0 = "pred0" if k == "H0" else f"pred0_{k}"
        row = [f"{wrmse(e.woba_rel - e[f'p_{k}'], e.b_pa):.5f}"]
        row += ["" if k == "H0" else f"{t('H0', k):.2f}"]
        for m in (e1, e2):
            x = e[m]
            row += [f"{wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.5f}" + ("" if k == "H0" else f" ({t('H0', k, m):.2f})")]
        row += [f"{sl:.3f}", f"{wrmse(e.woba_rel - e[c0], e.b_pa):.5f}"]
        print(f"| {k} | " + " | ".join(row) + " |")
    print(f"\nH1 vs HC (regression alone): t {t('HC', 'H1'):.2f} (16-19 {t('HC', 'H1', e1):.2f}, 21-24 {t('HC', 'H1', e2):.2f})")
    print(f"H2 vs H1: t {t('H1', 'H2'):.2f};  HC vs H0 (split alone): t {t('H0', 'HC'):.2f}")
    print("\nby season:")
    for N in TUNE:
        x = h[h.season == N]
        print(N, " ".join(f"{k} {wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.4f}" for k in arms))
    print("\nmean error (actual - proj) / RMSE by PA in target season, and by history PA:")
    hp = bat[bat.season < 2025].copy()
    e = e.assign(pa_band=pd.cut(e.b_pa, [0, 250, 450, 800], labels=["100-249", "250-449", "450+"]))
    for b, x in e.groupby("pa_band", observed=True):
        print(f"  {b:8s} n={len(x):4d}: " + "  ".join(
            f"{k} {np.average(x.woba_rel - x[f'p_{k}'], weights=x.b_pa):+.4f}/{wrmse(x.woba_rel - x[f'p_{k}'], x.b_pa):.4f}" for k in arms))
    e = e.assign(q=pd.qcut(e.p_H0, 5, labels=["q1 low", "q2", "q3", "q4", "q5 high"]))
    print("mean error by H0 projection quintile:")
    for b, x in e.groupby("q", observed=True):
        print(f"  {b:7s}: " + "  ".join(f"{k} {np.average(x.woba_rel - x[f'p_{k}'], weights=x.b_pa):+.4f}" for k in arms))
    print("2020 as target:", {k: round(wrmse(h[h.season == 2020].woba_rel - h[h.season == 2020][f"p_{k}"],
                                                h[h.season == 2020].b_pa), 4) for k in arms})
    h.to_parquet(ROOT / "work" / "results" / "s8_01_preds.parquet", index=False)
    return h


if __name__ == "__main__":
    run()
