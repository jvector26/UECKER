"""S2-01: does park-neutralising history (and re-applying the target-season park mix) improve Marcel?
usage: python model/park_test.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from marcel import COMP, WKEY, marcel_hitters, marcel_pitchers, wrmse

ROOT = Path(__file__).resolve().parents[1]
SEASON = ROOT / "data" / "season"
TARGETS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
CMAP = {"b_ubb": "ubb", "b_hp": "hbp", "b_1b": "1b", "b_2b": "2b", "b_3b": "3b", "b_hr": "hr"}


def multipliers(mix, pf, N, role, seasons):
    """PA-weighted park multiplier per player-season (factors as of N) -> wide frame player_id, season, comp cols."""
    m = mix[(mix.role == role) & mix.season.isin(seasons)]
    p = pf[pf["asof"] == N]
    out = {}
    if role == "bat":
        for c in CMAP.values():
            f = p[p.comp == c].set_index(["park_id", "hand"]).pf
            x = m.assign(pf=pd.MultiIndex.from_frame(m[["park_id", "hand"]]).map(f)).fillna({"pf": 1.0})
            out[c] = (x.n * x.pf).groupby([x.player_id, x.season]).sum() / x.n.groupby([x.player_id, x.season]).sum()
    else:
        f = p[p.comp == "runs"].set_index("park_id").pf
        x = m.assign(pf=m.park_id.map(f).fillna(1.0))
        out["runs"] = (x.n * x.pf).groupby([x.player_id, x.season]).sum() / x.n.groupby([x.player_id, x.season]).sum()
    return pd.DataFrame(out).reset_index()


def run():
    bat = pd.read_parquet(SEASON / "bat.parquet")
    pit = pd.read_parquet(SEASON / "pit.parquet")
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    mix = pd.read_parquet(SEASON / "park_mix.parquet")
    pf = pd.read_parquet(SEASON / "park_factors.parquet")
    H, P = [], []
    for N in TARGETS:
        # ---- hitters
        mh = multipliers(mix, pf, N, "bat", range(N - 3, N + 1))
        b = bat[bat.season.between(N - 3, N)].merge(mh, on=["player_id", "season"], how="left")
        nb = b.copy()
        for c, k in CMAP.items():
            nb[c] = nb[c] / nb[k].fillna(1.0)
        base = marcel_hitters(bat, rv, N, full=True)
        neut = marcel_hitters(nb[nb.season < N], rv, N, full=True)
        tm = mh[mh.season == N].set_index("player_id")
        wN1 = rv.loc[N - 1]
        neut = neut.set_index("player_id")
        ctx = 0.0
        for c, k in CMAP.items():
            mult = tm[k].reindex(neut.index).fillna(1.0)
            ctx = ctx + wN1[WKEY[c]] * (neut[c] * neut.af * mult - neut[f"lg_{c}"])
        neut["proj_park"] = ctx
        tgt = bat[(bat.season == N) & (bat.b_pa >= 100) & ~(bat.p_g > 0.5 * bat.b_g)][["player_id", "b_pa", "woba_rel", "teams"]]
        m = tgt.merge(base[["player_id", "proj"]], on="player_id", how="left").merge(
            neut[["proj", "proj_park"]].rename(columns={"proj": "proj_neutral"}).reset_index(), on="player_id", how="left")
        m["has_hist"] = m.proj.notna()
        m[["proj", "proj_neutral", "proj_park"]] = m[["proj", "proj_neutral", "proj_park"]].fillna(0.0)
        # flag: changed team vs last season, Rockies (DEN02 tenant) target season
        prev = bat[bat.season == N - 1].set_index("player_id")
        m["tmult"] = m.player_id.map(tm["hr"]).fillna(1.0)
        m["season"] = N
        H.append(m)
        # ---- pitchers
        mp = multipliers(mix, pf, N, "pit", range(N - 3, N + 1))
        q = pit[pit.season.between(N - 3, N - 1)].merge(mp, on=["player_id", "season"], how="left")
        q["p_r"] = q.p_r / q.runs.fillna(1.0)
        tp = mp[mp.season == N].set_index("player_id").runs
        tgtp = pit[(pit.season == N) & (pit.p_tbf >= 100)][["player_id", "p_tbf", "ra9_rel"]]
        for reg in (402.0, 804.0):
            b0 = marcel_pitchers(pit, rv, N, reg_outs=reg).set_index("player_id").proj
            n0 = marcel_pitchers(pd.concat([q, pit[pit.season == N]]), rv, N, reg_outs=reg, full=True).set_index("player_id")
            mult = tp.reindex(n0.index).fillna(1.0)
            pk = 27 * (n0.rate * mult / n0.af - n0.lgr)
            tgtp[f"base_{int(reg)}"] = tgtp.player_id.map(b0).fillna(0.0)
            tgtp[f"park_{int(reg)}"] = tgtp.player_id.map(pk).fillna(0.0)
        tgtp["season"] = N
        P.append(tgtp)
    return pd.concat(H), pd.concat(P)


def clustered_t(a, b, err_a, err_b, w, season):
    """season-clustered t on the per-season weighted MSE difference (positive = b better)."""
    d = pd.DataFrame({"s": season, "d": w * (err_a ** 2 - err_b ** 2), "w": w}).groupby("s").sum()
    x = d.d / d.w
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


if __name__ == "__main__":
    H, P = run()
    ex = H.season != 2020
    h = H[ex]
    print("HITTERS rel-wOBA RMSE (ex-2020)")
    for c in ["proj", "proj_neutral", "proj_park"]:
        print(f"  {c:13s}", round(wrmse(h.woba_rel - h[c], h.b_pa), 4))
    print("  t (park vs base, season-clustered):", round(clustered_t(None, None, h.woba_rel - h.proj, h.woba_rel - h.proj_park, h.b_pa, h.season), 2))
    by = h.groupby("season").apply(lambda g: pd.Series({"base": wrmse(g.woba_rel - g.proj, g.b_pa), "park": wrmse(g.woba_rel - g.proj_park, g.b_pa)}), include_groups=False)
    print(by.round(4).T.to_string())
    for name, mask in [("big target park (HR mult>1.08 or <0.93)", (h.tmult > 1.08) | (h.tmult < 0.93)), ("multi-team target season", h.teams > 1)]:
        g = h[mask]
        print(f"  {name}: n={len(g)} base {wrmse(g.woba_rel - g.proj, g.b_pa):.4f} park {wrmse(g.woba_rel - g.proj_park, g.b_pa):.4f}")
    p = P[P.season != 2020]
    print("PITCHERS rel-RA9 RMSE (ex-2020)")
    for reg in (402, 804):
        print(f"  reg {reg}: base {wrmse(p.ra9_rel - p[f'base_{reg}'], p.p_tbf):.4f}  park {wrmse(p.ra9_rel - p[f'park_{reg}'], p.p_tbf):.4f}  "
              f"t {clustered_t(None, None, p.ra9_rel - p[f'base_{reg}'], p.ra9_rel - p[f'park_{reg}'], p.p_tbf, p.season):.2f}")
    h2 = H[~ex]
    p2 = P[P.season == 2020]
    print(f"2020: hitters base {wrmse(h2.woba_rel - h2.proj, h2.b_pa):.4f} park {wrmse(h2.woba_rel - h2.proj_park, h2.b_pa):.4f}; "
          f"pitchers(804) base {wrmse(p2.ra9_rel - p2.base_804, p2.p_tbf):.3f} park {wrmse(p2.ra9_rel - p2.park_804, p2.p_tbf):.3f}")
