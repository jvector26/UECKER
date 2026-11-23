"""Marcel baseline (Tango) on UECKER's own run values, plus the scoring harness for the player-level research bar.

usage: python model/marcel.py            -> prints S1-02 table, writes work/results/marcel_*.csv
Holdout seasons (2025-2026) are never scored here unless HOLDOUT=1 is set at the pre-season final check.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEASON = ROOT / "data" / "season"
RES = ROOT / "work" / "results"

TUNE = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
HOLDOUT = [2025, 2026]
COMP = ["b_ubb", "b_hp", "b_1b", "b_2b", "b_3b", "b_hr"]
WKEY = {"b_ubb": "w_bb", "b_hp": "w_hbp", "b_1b": "w_1b", "b_2b": "w_2b", "b_3b": "w_3b", "b_hr": "w_hr"}


def age_factor(age):
    return np.where(age < 29, 1 + 0.006 * (29 - age), 1 - 0.003 * (age - 29))


def marcel_hitters(bat, rv, N, reg_pa=1200.0, weights=(5, 4, 3), full=False):
    """Project rel-wOBA for season N from seasons N-1..N-3. Returns DataFrame player_id, proj."""
    hist = bat[bat.season.between(N - 3, N - 1)].copy()
    hist["w"] = hist.season.map({N - 1: weights[0], N - 2: weights[1], N - 3: weights[2]})
    # league component rates per season (per wOBA-denominator PA), all non-pitcher-dominant hitters included
    lgs = bat[bat.season.between(N - 3, N - 1)].groupby("season")[COMP + ["woba_den"]].sum()
    lg_rate = lgs[COMP].div(lgs.woba_den, axis=0)
    hist = hist.join(lg_rate.add_prefix("lg_"), on="season")
    hist["wpa"] = hist.w * hist.woba_den
    g = hist.groupby("player_id")
    wpa = g.wpa.sum()
    out = pd.DataFrame(index=wpa.index)
    wN1 = rv.loc[N - 1]
    proj_woba = 0.0
    lg_woba = 0.0
    for c in COMP:
        player_w = (hist.w * hist[c]).groupby(hist.player_id).sum()
        lg_w = (hist.wpa * hist[f"lg_{c}"]).groupby(hist.player_id).sum() / wpa  # league rate in player's mix of years
        lg_w = lg_w.fillna(lg_rate[c].mean())
        rate = (player_w + reg_pa * lg_w) / (wpa + reg_pa)
        out[c] = rate
        out[f"lg_{c}"] = lg_w
    last = hist.sort_values("season").groupby("player_id").tail(1).set_index("player_id")
    age = last.age + (N - last.season)
    af = age_factor(age.reindex(out.index).values)
    for c in COMP:
        proj_woba = proj_woba + wN1[WKEY[c]] * out[c] * af
        lg_woba = lg_woba + wN1[WKEY[c]] * out[f"lg_{c}"]
    out["proj"] = proj_woba - lg_woba
    # last-season-only baseline (unregressed)
    l1 = hist[hist.season == N - 1].set_index("player_id")
    out["last1"] = l1.woba_rel.reindex(out.index)
    out["af"] = af
    if full:
        return out.reset_index()
    return out.reset_index()[["player_id", "proj", "last1"]]


def marcel_pitchers(pit, rv, N, reg_outs=402.0, weights=(3, 2, 1), full=False):
    hist = pit[pit.season.between(N - 3, N - 1)].copy()
    hist["w"] = hist.season.map({N - 1: weights[0], N - 2: weights[1], N - 3: weights[2]})
    lg = pit[pit.season.between(N - 3, N - 1)].groupby("season")[["p_r", "p_out", "p_tbf", "p_so", "p_bb", "p_ibb", "p_hp", "p_hr"]].sum()
    lg["r_out"] = lg.p_r / lg.p_out
    hist = hist.join(lg[["r_out"]].add_prefix("lg_"), on="season")
    hist["wo"] = hist.w * hist.p_out
    g = hist.groupby("player_id")
    wo = g.wo.sum()
    pr = (hist.w * hist.p_r).groupby(hist.player_id).sum()
    lgr = (hist.wo * hist.lg_r_out).groupby(hist.player_id).sum() / wo
    lgr = lgr.fillna(lg.r_out.mean())
    rate = (pr + reg_outs * lgr) / (wo + reg_outs)
    last = hist.sort_values("season").groupby("player_id").tail(1).set_index("player_id")
    age = (last.age + (N - last.season)).reindex(rate.index).values
    af = age_factor(age)
    # age: good pitchers get better young -> runs allowed scale by 1/af
    proj = 27 * (rate / af - lgr)

    # component (FIP-style) version: K, uBB, HBP, HR per BF, regressed; BIP runs = league
    wN1 = rv.loc[N - 1]
    hist["p_ubb"] = hist.p_bb - hist.p_ibb
    hist["wbf"] = hist.w * hist.p_tbf
    wbf = hist.groupby("player_id").wbf.sum()
    reg_bf = reg_outs * (lg.p_tbf.sum() / lg.p_out.sum())
    lg["p_ubb"] = lg.p_bb - lg.p_ibb
    comp_runs = 0.0
    for c, lwk in [("p_so", "lw_k"), ("p_ubb", "lw_bb"), ("p_hp", "lw_hbp"), ("p_hr", "lw_hr")]:
        lgc = (lg[c] / lg.p_tbf)
        hist[f"lg_{c}"] = hist.season.map(lgc)
        pl = (hist.w * hist[c]).groupby(hist.player_id).sum()
        lgw = (hist.wbf * hist[f"lg_{c}"]).groupby(hist.player_id).sum() / wbf
        r = (pl + reg_bf * lgw) / (wbf + reg_bf)
        # runs relative to a generic out-or-BIP: value relative to league out value
        comp_runs = comp_runs + (r - lgw) * (wN1[lwk] - wN1["lw_out"])
    bf_per_27 = 27 * lg.p_tbf.sum() / lg.p_out.sum()
    comp = bf_per_27 * comp_runs / af
    l1 = hist[hist.season == N - 1].set_index("player_id")
    out = pd.DataFrame({"proj": proj, "proj_comp": comp})
    out["last1"] = l1.ra9_rel.reindex(out.index)
    if full:
        out["rate"], out["af"], out["lgr"] = rate, af, lgr
    return out.rename_axis("player_id").reset_index()


def wrmse(err, w):
    return float(np.sqrt(np.sum(w * err ** 2) / np.sum(w)))


def score_hitters(bat, rv, seasons, **kw):
    rows, allp = [], []
    for N in seasons:
        tgt = bat[(bat.season == N) & (bat.b_pa >= 100) & ~(bat.p_g > 0.5 * bat.b_g)][["player_id", "b_pa", "woba_rel"]]
        p = marcel_hitters(bat, rv, N, **kw)
        m = tgt.merge(p, on="player_id", how="left")
        m["has_hist"] = m.proj.notna()
        m["proj"] = m.proj.fillna(0.0)
        m["last1_f"] = m.last1.fillna(0.0)
        m["season"] = N
        allp.append(m)
        w = m.b_pa
        rows.append({"season": N, "n": len(m), "lgavg": wrmse(m.woba_rel, w), "last1": wrmse(m.woba_rel - m.last1_f, w),
                     "marcel": wrmse(m.woba_rel - m.proj, w)})
    return pd.DataFrame(rows), pd.concat(allp)


def score_pitchers(pit, rv, seasons, **kw):
    rows, allp = [], []
    for N in seasons:
        tgt = pit[(pit.season == N) & (pit.p_tbf >= 100)][["player_id", "p_tbf", "ra9_rel"]]
        p = marcel_pitchers(pit, rv, N, **kw)
        m = tgt.merge(p, on="player_id", how="left")
        m["has_hist"] = m.proj.notna()
        for c in ["proj", "proj_comp"]:
            m[c] = m[c].fillna(0.0)
        m["last1_f"] = m.last1.fillna(0.0)
        m["season"] = N
        allp.append(m)
        w = m.p_tbf
        rows.append({"season": N, "n": len(m), "lgavg": wrmse(m.ra9_rel, w), "last1": wrmse(m.ra9_rel - m.last1_f, w),
                     "marcel": wrmse(m.ra9_rel - m.proj, w), "marcel_comp": wrmse(m.ra9_rel - m.proj_comp, w)})
    return pd.DataFrame(rows), pd.concat(allp)


def pooled(allp, err_cols, tgt, wcol, mask=None):
    a = allp if mask is None else allp[mask]
    return {c: round(wrmse(a[tgt] - a[c], a[wcol]), 4) for c in err_cols}


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    bat = pd.read_parquet(SEASON / "bat.parquet")
    pit = pd.read_parquet(SEASON / "pit.parquet")
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    seasons = TUNE + (HOLDOUT if os.environ.get("HOLDOUT") == "1" else [])
    pd.set_option("display.width", 200)

    hs, hp = score_hitters(bat, rv, seasons)
    hp["zero"] = 0.0
    print("HITTERS rel-wOBA, PA-weighted RMSE by season")
    print(hs.round(4).to_string(index=False))
    ex20 = hp.season != 2020
    print("pooled ex-2020:", pooled(hp, ["zero", "last1_f", "proj"], "woba_rel", "b_pa", ex20))
    print("pooled ex-2020, with MLB history only:", pooled(hp, ["zero", "last1_f", "proj"], "woba_rel", "b_pa", ex20 & hp.has_hist))
    for k in [600, 2400]:
        _, x = score_hitters(bat, rv, [s for s in seasons if s != 2020], reg_pa=k)
        print(f"  sensitivity reg_pa={k}:", round(wrmse(x.woba_rel - x.proj, x.b_pa), 4))

    ps, pp = score_pitchers(pit, rv, seasons)
    pp["zero"] = 0.0
    print("\nPITCHERS rel-RA9, BF-weighted RMSE by season")
    print(ps.round(3).to_string(index=False))
    ex20p = pp.season != 2020
    print("pooled ex-2020:", pooled(pp, ["zero", "last1_f", "proj", "proj_comp"], "ra9_rel", "p_tbf", ex20p))
    print("pooled ex-2020, with MLB history only:", pooled(pp, ["zero", "last1_f", "proj", "proj_comp"], "ra9_rel", "p_tbf", ex20p & pp.has_hist))
    for k in [201, 804]:
        _, x = score_pitchers(pit, rv, [s for s in seasons if s != 2020], reg_outs=k)
        print(f"  sensitivity reg_outs={k}:", round(wrmse(x.ra9_rel - x.proj, x.p_tbf), 3), round(wrmse(x.ra9_rel - x.proj_comp, x.p_tbf), 3))

    hs.to_csv(RES / "marcel_hitters_by_season.csv", index=False)
    ps.to_csv(RES / "marcel_pitchers_by_season.csv", index=False)
