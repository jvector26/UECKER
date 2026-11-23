"""Park factors from Retrosheet events (2010-2025).

Component factors per PA (uBB, HBP, 1B, 2B, 3B, HR, K) by batter hand, plus a runs-per-PA factor (no hand split).
"As of season N" factors use seasons N-3..N-1 only: rate in the park (both teams) / rate in the park's home team's road
games (both teams). Log ratios are shrunk toward 0 by empirical Bayes (true spread = observed variance - sampling
variance, per component and hand), then renormalised so the league PA-weighted mean factor is 1.

Outputs
  data/season/park_factors.parquet   asof, park_id, hand (L/R/A), comp, pf, raw, n_in
  data/season/park_mix.parquet       player-season PA (as batter) / BF (as pitcher) by park and batter hand
usage: python model/parks.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RETRO = ROOT / "data" / "retro"
SEASON = ROOT / "data" / "season"
COMPS = {"ubb": [14], "hbp": [16], "1b": [20], "2b": [21], "3b": [22], "hr": [23], "k": [3]}
Y0, Y1 = 2010, 2025


def events(y):
    e = pd.read_parquet(RETRO / f"pa_{y}.parquet", columns=["game_id", "away_team_id", "bat_home_id", "bat_id", "pit_id",
                                                            "bat_hand_cd", "event_cd", "bat_event_fl", "bat_dest_id",
                                                            "run1_dest_id", "run2_dest_id", "run3_dest_id"])
    d = pd.read_parquet(RETRO / f"daily_{y}.parquet", columns=["game_id", "park_id"]).drop_duplicates("game_id")
    e = e.merge(d, on="game_id", how="left")
    e["home_team"] = e.game_id.str[:3]
    e["runs"] = (e[["bat_dest_id", "run1_dest_id", "run2_dest_id", "run3_dest_id"]].to_numpy() >= 4).sum(axis=1)
    e["season"] = y
    return e


def build():
    agg, mixes = [], []
    for y in range(Y0, Y1 + 1):
        e = events(y)
        runs = e.groupby(["season", "game_id", "park_id", "home_team", "away_team_id"]).runs.sum()
        pa = e[e.bat_event_fl == "T"].copy()
        for c, codes in COMPS.items():
            pa[c] = pa.event_cd.isin(codes).astype(int)
        pa["pa"] = 1
        g = pa.groupby(["season", "game_id", "park_id", "home_team", "away_team_id", "bat_hand_cd"])[list(COMPS) + ["pa"]].sum()
        g = g.reset_index()
        rg = runs.rename("runs").reset_index()
        agg.append((g, rg))
        mb = pa.groupby(["season", "bat_id", "park_id", "bat_hand_cd"]).pa.sum().rename("n").reset_index().rename(columns={"bat_id": "player_id"})
        mb["role"] = "bat"
        mp = pa.groupby(["season", "pit_id", "park_id", "bat_hand_cd"]).pa.sum().rename("n").reset_index().rename(columns={"pit_id": "player_id"})
        mp["role"] = "pit"
        mixes += [mb, mp]
    G = pd.concat([a for a, _ in agg], ignore_index=True)
    R = pd.concat([b for _, b in agg], ignore_index=True)
    # runs per game need PA per game for the rate
    gpa = G.groupby(["season", "game_id"]).pa.sum()
    R["pa"] = R.set_index(["season", "game_id"]).index.map(gpa)
    mix = pd.concat(mixes, ignore_index=True)
    mix.rename(columns={"bat_hand_cd": "hand"}).to_parquet(SEASON / "park_mix.parquet", index=False)

    out = []
    for N in range(Y0 + 1, Y1 + 2):  # as-of seasons 2011..2026
        win = list(range(max(Y0, N - 3), N))
        g = G[G.season.isin(win)]
        r = R[R.season.isin(win)]
        # park's home team = most frequent home team there
        tenant = g.groupby(["park_id", "home_team"]).pa.sum().reset_index().sort_values("pa").groupby("park_id").tail(1)
        tenant = tenant.set_index("park_id").home_team
        rows = []
        for park, team in tenant.items():
            inn = g[g.park_id == park]
            road = g[((g.home_team == team) | (g.away_team_id == team)) & (g.park_id != park)]
            for hand in ["L", "R"]:
                a, b = inn[inn.bat_hand_cd == hand], road[road.bat_hand_cd == hand]
                for c in COMPS:
                    rows.append((park, hand, c, a[c].sum(), a.pa.sum(), b[c].sum(), b.pa.sum()))
            ri = r[r.park_id == park]
            rr = r[((r.home_team == team) | (r.away_team_id == team)) & (r.park_id != park)]
            rows.append((park, "A", "runs", ri.runs.sum(), ri.pa.sum(), rr.runs.sum(), rr.pa.sum()))
        f = pd.DataFrame(rows, columns=["park_id", "hand", "comp", "c_in", "pa_in", "c_rd", "pa_rd"])
        f = f[(f.pa_in > 0) & (f.pa_rd > 0) & (f.c_in > 0) & (f.c_rd > 0)].copy()
        f["raw"] = (f.c_in / f.pa_in) / (f.c_rd / f.pa_rd)
        f["lr"] = np.log(f.raw)
        f["se2"] = 1 / f.c_in + 1 / f.c_rd
        f["pf"] = 1.0
        for (h, c), s in f.groupby(["hand", "comp"]):
            big = s[s.pa_in > 0.5 * s.pa_in.max()]
            tau2 = max(np.average((big.lr - np.average(big.lr, weights=big.pa_in)) ** 2, weights=big.pa_in) - big.se2.mean(), 0.0)
            k = tau2 / (tau2 + s.se2)
            f.loc[s.index, "pf"] = np.exp(k * s.lr)
            f.loc[s.index, "tau"] = np.sqrt(tau2)
        # renormalise: PA-weighted mean over parks = 1 per (hand, comp)
        f["pf"] = f.pf / f.groupby(["hand", "comp"]).apply(lambda s: np.average(s.pf, weights=s.pa_in), include_groups=False).reindex(
            pd.MultiIndex.from_frame(f[["hand", "comp"]])).values
        f["asof"] = N
        out.append(f[["asof", "park_id", "hand", "comp", "pf", "raw", "tau", "pa_in"]])
    pf = pd.concat(out, ignore_index=True)
    pf.to_parquet(SEASON / "park_factors.parquet", index=False)
    return pf


if __name__ == "__main__":
    pf = build()
    pd.set_option("display.width", 200)
    x = pf[pf["asof"] == 2026]
    print("as of 2026, runs factor (top/bottom 4):")
    r = x[x.comp == "runs"].sort_values("pf")
    print(pd.concat([r.head(4), r.tail(4)])[["park_id", "pf", "raw", "pa_in"]].round(3).to_string(index=False))
    print("HR factor by hand, extremes:")
    h = x[x.comp == "hr"].pivot(index="park_id", columns="hand", values="pf").round(2)
    print(h.sort_values("L").iloc[[0, 1, 2, -3, -2, -1]])
    print("true spread (tau) of log factor, as of 2026:")
    print(x.groupby(["comp", "hand"]).tau.first().round(3).unstack())
