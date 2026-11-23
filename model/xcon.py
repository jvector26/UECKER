"""S3-01: hitter contact quality. UECKER's own expected contact value from exit velocity x launch angle, then Marcel on
history where contact value = a*actual + (1-a)*expected. See RESEARCH_LOG S3-01 (pre-registered).

Outputs data/season/xcon.parquet  (season, player_id, mlbam, n_con, n_ev, acon, xcon, delta) — acon/xcon in wOBA-weight units.
usage: python model/xcon.py   -> builds xcon (2015-2026) and prints the S3-01 test
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marcel import TUNE, marcel_hitters, wrmse  # noqa: E402
from park_test import clustered_t  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "data" / "statcast"
SEASON = ROOT / "data" / "season"

HIT = {"single": "w_1b", "double": "w_2b", "triple": "w_3b", "home_run": "w_hr"}
OUTS = {"field_out", "force_out", "grounded_into_double_play", "double_play", "triple_play", "fielders_choice",
        "fielders_choice_out", "sac_fly", "sac_fly_double_play", "field_error"}
EV_EDGES = np.arange(40, 124, 3)
LA_EDGES = np.arange(-80, 84, 4)


def contact(year, rv):
    cols = ["game_pk", "at_bat_number", "pitch_number", "batter", "events", "launch_speed", "launch_angle"]
    s = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(SC.glob(f"{year}_*.parquet"))], ignore_index=True)
    s = s[s.events.isin(OUTS | set(HIT))]
    s = s.sort_values("pitch_number").groupby(["game_pk", "at_bat_number"]).tail(1)
    w = rv.loc[year]
    s["val"] = s.events.map({k: w[v] for k, v in HIT.items()}).fillna(0.0)
    s["has_ev"] = s.launch_speed.notna() & s.launch_angle.notna()
    e = np.clip(np.digitize(s.launch_speed.fillna(0), EV_EDGES), 0, len(EV_EDGES))
    a = np.clip(np.digitize(s.launch_angle.fillna(0), LA_EDGES), 0, len(LA_EDGES))
    s["bin_e"], s["bin_a"] = e, a
    g = s[s.has_ev]
    S = np.zeros((len(EV_EDGES) + 1, len(LA_EDGES) + 1))
    C = np.zeros_like(S)
    np.add.at(S, (g.bin_e.values, g.bin_a.values), g.val.values)
    np.add.at(C, (g.bin_e.values, g.bin_a.values), 1.0)
    # 3x3 smoothing, center weight 4
    k = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 1]], float)

    def conv(M):
        P = np.pad(M, 1)
        return sum(k[i, j] * P[i:i + M.shape[0], j:j + M.shape[1]] for i in range(3) for j in range(3))

    V = conv(S) / np.maximum(conv(C), 1e-9)
    s["xval"] = np.where(s.has_ev, V[s.bin_e.values, s.bin_a.values], s.val)
    return s


def build(years, rv, bat):
    out = []
    for y in years:
        s = contact(y, rv)
        d = s.groupby("batter").agg(n_con=("val", "size"), n_ev=("has_ev", "sum"), acon=("val", "sum"), xcon=("xval", "sum"))
        d = d.reset_index().rename(columns={"batter": "mlbam"})
        d["season"] = y
        print(y, f"contact {len(s)}, with EV/LA {s.has_ev.mean():.3f}, league acon/xcon {s.val.sum() / s.xval.sum():.4f}")
        out.append(d)
    x = pd.concat(out, ignore_index=True)
    ids = bat[["season", "mlbam", "player_id", "woba_den"]].dropna(subset=["mlbam"]).astype({"mlbam": int})
    x = x.merge(ids, on=["season", "mlbam"], how="inner")
    x["delta"] = (x.xcon - x.acon) / x.woba_den.where(x.woba_den > 0)
    return x


def corrections(bat, xc, N, reg_pa=1200.0, weights=(5, 4, 3)):
    """Marcel numerator correction per unit (1-a): sum w*(xcon-acon) / (sum w*den + reg)."""
    hist = bat[bat.season.between(N - 3, N - 1)][["player_id", "season", "woba_den"]].copy()
    hist["w"] = hist.season.map({N - 1: weights[0], N - 2: weights[1], N - 3: weights[2]})
    hist = hist.merge(xc[["player_id", "season", "xcon", "acon"]], on=["player_id", "season"], how="left").fillna({"xcon": 0, "acon": 0})
    num = (hist.w * (hist.xcon - hist.acon)).groupby(hist.player_id).sum()
    den = (hist.w * hist.woba_den).groupby(hist.player_id).sum()
    return num / (den + reg_pa)


def run():
    bat = pd.read_parquet(SEASON / "bat.parquet")
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    xc = build(range(2015, 2027), rv, bat)
    xc.to_parquet(SEASON / "xcon.parquet", index=False)
    print("hitter-season delta (PA>=300) sd:", round(xc[xc.woba_den >= 300].delta.std(), 4))

    # stability check: year-to-year correlation of actual vs expected contact value per BIP, same hitter
    xc["a_rate"], xc["x_rate"] = xc.acon / xc.n_con, xc.xcon / xc.n_con
    q = xc[xc.n_con >= 250][["player_id", "season", "a_rate", "x_rate"]]
    q2 = q.merge(q.assign(season=q.season - 1), on=["player_id", "season"], suffixes=("", "_next"))
    print("y/y corr, BIP>=250: actual->next actual", round(q2.a_rate.corr(q2.a_rate_next), 3),
          "| expected->next actual", round(q2.x_rate.corr(q2.a_rate_next), 3))

    rows = []
    for N in TUNE:
        tgt = bat[(bat.season == N) & (bat.b_pa >= 100) & ~(bat.p_g > 0.5 * bat.b_g)][["player_id", "b_pa", "woba_rel"]]
        p = marcel_hitters(bat, rv, N, full=True).set_index("player_id")
        corr = corrections(bat, xc, N).reindex(p.index).fillna(0.0) * p.af
        m = tgt.merge(p[["proj"]].assign(corr=corr).reset_index(), on="player_id", how="left").fillna({"proj": 0.0, "corr": 0.0})
        m["season"] = N
        rows.append(m)
    h = pd.concat(rows)
    A = np.round(np.arange(0, 1.01, 0.1), 1)
    ex = h[h.season != 2020]
    curve = {a: wrmse(ex.woba_rel - (ex.proj + (1 - a) * ex["corr"]), ex.b_pa) for a in A}
    print("\npooled RMSE by a (a=1 is Marcel), targets 2016-24 ex-2020:")
    print("  " + "  ".join(f"{a}:{v:.5f}" for a, v in curve.items()))
    # leave-one-target-season-out choice of a
    h["proj_x"] = np.nan
    picks = {}
    for N in TUNE:
        tr = ex[ex.season != N]
        best = min(A, key=lambda a: wrmse(tr.woba_rel - (tr.proj + (1 - a) * tr["corr"]), tr.b_pa))
        picks[N] = best
        i = h.season == N
        h.loc[i, "proj_x"] = h.loc[i, "proj"] + (1 - best) * h.loc[i, "corr"]
    print("LOSO a picks:", picks)
    ex = h[h.season != 2020]
    print("\n| subset | Marcel | Marcel+xcon | t |")
    for name, msk in [("2016-24 ex-2020", ex.season > 0), ("2016-19", ex.season <= 2019), ("2021-24", ex.season >= 2021),
                      ("2018-24 ex-2020 (full Statcast hist)", ex.season >= 2018)]:
        e = ex[msk]
        t = clustered_t(None, None, e.woba_rel - e.proj, e.woba_rel - e.proj_x, e.b_pa, e.season)
        print(f"| {name} | {wrmse(e.woba_rel - e.proj, e.b_pa):.4f} | {wrmse(e.woba_rel - e.proj_x, e.b_pa):.4f} | {t:.2f} |")
    e20 = h[h.season == 2020]
    print("2020:", round(wrmse(e20.woba_rel - e20.proj, e20.b_pa), 4), "->", round(wrmse(e20.woba_rel - e20.proj_x, e20.b_pa), 4))
    for N in TUNE:
        e = h[h.season == N]
        print(N, round(wrmse(e.woba_rel - e.proj, e.b_pa), 4), round(wrmse(e.woba_rel - e.proj_x, e.b_pa), 4))
    return h


if __name__ == "__main__":
    run()
