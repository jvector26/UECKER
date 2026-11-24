"""S7-02 build: pitch-level physics + plate discipline per pitcher-season, and out-of-fold stuff scores per target N.
writes data/season/pdisc.parquet, data/season/stuff_oof.parquet
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "data" / "statcast"
SEASON = ROOT / "data" / "season"
WHIFF = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FOUL = {"foul", "foul_tip", "foul_bunt", "bunt_foul_tip"}
BALL = {"ball", "blocked_ball"}
CSTR = {"called_strike"}
INPLAY = {"hit_into_play"}
COLS = ["game_pk", "pitcher", "stand", "p_throws", "pitch_type", "description", "zone", "release_speed",
        "release_spin_rate", "spin_axis", "pfx_x", "pfx_z", "release_pos_x", "release_pos_z", "release_extension"]
FEATS = ["velo", "spin", "axis", "hb", "ivb", "rx", "rz", "ext", "dvelo", "dhb", "divb", "same"]


def load_year(y):
    s = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(SC.glob(f"{y}_*.parquet"))], ignore_index=True)
    s = s[s.description.isin(WHIFF | FOUL | BALL | CSTR | INPLAY | {"hit_by_pitch"})].copy()
    s["season"] = y
    L = s.p_throws == "L"
    s["velo"] = s.release_speed.astype("float32")
    s["spin"] = s.release_spin_rate.astype("float32")
    s["axis"] = np.where(L, 360 - s.spin_axis, s.spin_axis).astype("float32")
    s["hb"] = np.where(L, -s.pfx_x, s.pfx_x).astype("float32")
    s["ivb"] = s.pfx_z.astype("float32")
    s["rx"] = np.where(L, -s.release_pos_x, s.release_pos_x).astype("float32")
    s["rz"] = s.release_pos_z.astype("float32")
    s["ext"] = s.release_extension.astype("float32")
    s["same"] = (s.p_throws == s.stand).astype("int8")
    fb = s[s.pitch_type.isin(["FF", "SI"])].groupby(["pitcher", "pitch_type"]).size().rename("n").reset_index()
    fb = fb.sort_values("n").groupby("pitcher").tail(1)[["pitcher", "pitch_type"]]
    pri = s.merge(fb, on=["pitcher", "pitch_type"]).groupby("pitcher")[["velo", "hb", "ivb"]].mean()
    alt = s.groupby("pitcher")[["velo", "hb", "ivb"]].max()  # no FF/SI: use max velo row-wise proxy
    pri = pri.reindex(alt.index).fillna(alt)
    for c, d in [("velo", "dvelo"), ("hb", "dhb"), ("ivb", "divb")]:
        s[d] = (s[c] - s.pitcher.map(pri[c])).astype("float32")
    s["swing"] = s.description.isin(WHIFF | FOUL | INPLAY)
    s["whiff"] = s.description.isin(WHIFF)
    return s[["season", "pitcher"] + FEATS + ["description", "zone", "swing", "whiff"]]


def main():
    yrs = range(2015, 2025)
    S = {}
    rows = []
    for y in yrs:
        s = load_year(y)
        S[y] = s
        d = s.description
        g = s.assign(n=1, wh=s.whiff, cs=d.isin(CSTR), fo=d.isin(FOUL), ba=d.isin(BALL),
                     zn=s.zone.between(1, 9), sw=s.swing).groupby("pitcher")[["n", "wh", "cs", "fo", "ba", "zn", "sw"]].sum()
        rows.append(g.reset_index().assign(season=y))
        print("loaded", y, len(s), flush=True)
    pd.concat(rows).rename(columns={"pitcher": "mlbam"}).to_parquet(SEASON / "pdisc.parquet", index=False)
    out = []
    params = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2, seed=1)
    for N in range(2016, 2025):
        tr = pd.concat([S[y][S[y].swing] for y in range(2015, N)])
        ap = pd.concat([S[y] for y in range(max(2015, N - 3), N)])
        fold_tr = (tr.pitcher * 2654435761 % 2**32) % 2
        fold_ap = (ap.pitcher * 2654435761 % 2**32) % 2
        ap = ap.assign(p=np.nan)
        for f in (0, 1):
            t = tr[fold_tr != f]
            if len(t) > 1_200_000:
                t = t.sample(1_200_000, random_state=N)
            m = lgb.train(params, lgb.Dataset(t[FEATS], t.whiff.astype(int)), num_boost_round=300)
            i = fold_ap == f
            ap.loc[i, "p"] = m.predict(ap.loc[i, FEATS])
        sc = ap.groupby(["pitcher", "season"]).p.mean().rename("stuff").reset_index()
        out.append(sc.assign(N=N))
        print("stuff N", N, "train swings", len(tr), "mean", round(sc.stuff.mean(), 4), flush=True)
    pd.concat(out).rename(columns={"pitcher": "mlbam"}).to_parquet(SEASON / "stuff_oof.parquet", index=False)


if __name__ == "__main__":
    main()
