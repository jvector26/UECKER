"""S12 diagnostics (NOT a test): where do V3's team-win misses come from? Uses only scored seasons 2016-24 ex-2020.
usage: python work/diag_s12.py | tee work/results/s12_diag.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from pcomp import EV_EDGES, HIT, LA_EDGES, OUTS  # noqa: E402
from team import RES, SCORED, team_results  # noqa: E402

SC, API, SEASON = ROOT / "data" / "statcast", ROOT / "data" / "statsapi", ROOT / "data" / "season"
AL = {108, 110, 111, 114, 116, 117, 118, 133, 136, 139, 140, 141, 142, 145, 147}


def team_defense(year, rv):
    """per fielding team: BIP, actual contact value allowed, expected (EV x LA bins, same smoothing as pcomp)."""
    cols = ["game_pk", "at_bat_number", "pitch_number", "inning_topbot", "events", "launch_speed", "launch_angle"]
    s = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(SC.glob(f"{year}_*.parquet"))], ignore_index=True)
    s = s[s.events.isin(OUTS | set(HIT))]
    s = s.sort_values("pitch_number").groupby(["game_pk", "at_bat_number"]).tail(1)
    w = rv.loc[year]
    s["val"] = s.events.map({k: w[v] - w["lw_out"] for k, v in HIT.items()}).fillna(0.0)
    s = s[s.events != "home_run"]  # defense can't touch HRs
    s = s[s.launch_speed.notna() & s.launch_angle.notna()]
    e = np.clip(np.digitize(s.launch_speed, EV_EDGES), 0, len(EV_EDGES))
    a = np.clip(np.digitize(s.launch_angle, LA_EDGES), 0, len(LA_EDGES))
    S = np.zeros((len(EV_EDGES) + 1, len(LA_EDGES) + 1))
    C = np.zeros_like(S)
    np.add.at(S, (e, a), s.val.values)
    np.add.at(C, (e, a), 1.0)
    k = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 1]], float)

    def conv(M):
        P = np.pad(M, 1)
        return sum(k[i, j] * P[i:i + M.shape[0], j:j + M.shape[1]] for i in range(3) for j in range(3))
    s["xval"] = (conv(S) / np.maximum(conv(C), 1e-9))[e, a]
    g = pd.read_parquet(API / f"games_{year}.parquet").drop_duplicates("game_pk").set_index("game_pk")
    s["fld"] = np.where(s.inning_topbot == "Top", s.game_pk.map(g.home_id), s.game_pk.map(g.away_id))
    o = s.groupby("fld").agg(bip=("val", "size"), act=("val", "sum"), x=("xval", "sum"))
    o["def"] = o.x - o.act  # runs saved by fielders (+ = good)
    o["season"] = year
    return o.reset_index().rename(columns={"fld": "team_id"})


def main():
    rv = pd.read_parquet(SEASON / "run_values.parquet").set_index("season")
    R = pd.read_parquet(RES / "s11_02_preds.parquet")
    res = pd.concat([team_results(y) for y in range(2015, 2025)])
    D = pd.concat([team_defense(y, rv) for y in range(2015, 2025)])
    D = D.merge(res[["team_id", "season", "g"]], on=["team_id", "season"])
    D["def162"] = D["def"] / D.g * 162
    print("team defense runs saved per 162 (contact ex-HR, actual vs EV x LA expected): SD by season")
    print(D.groupby("season").def162.std().round(1).to_dict())
    d1 = D[["team_id", "season", "def162"]].assign(season=D.season + 1).rename(columns={"def162": "def_1"})
    d0 = D[["team_id", "season", "def162"]].rename(columns={"def162": "def_0"})
    x = R[R.season.isin(SCORED)].merge(d1, on=["team_id", "season"], how="left").merge(d0, on=["team_id", "season"], how="left")
    # 2021 lag = 2020 (60 games, per-162 scaled): keep but flag
    c = lambda col: x[col] - x.groupby("season")[col].transform("mean")  # noqa: E731
    x["e_w"] = x.act - x.V3
    x["e_rs"] = c("rs_a") - c("rs_p")
    x["e_ra"] = c("ra_a") - c("ra_p")
    x["def_1c"], x["def_0c"] = c("def_1"), c("def_0")
    x["nl_pre"] = (~x.team_id.isin(AL) & (x.season < 2022)).astype(int)
    v = pd.read_parquet(ROOT / "data" / "vegas" / "win_totals.parquet")[["season", "team_id", "line"]]
    x = x.merge(v, on=["season", "team_id"], how="left")
    x["veg"] = x.line - x.groupby("season").line.transform("mean") + 81

    print(f"\nV3 RMSE {np.sqrt((x.e_w ** 2).mean()):.2f}, Vegas {np.sqrt(((x.act - x.veg) ** 2).mean()):.2f}")
    print(f"SD of run errors per 162 (UECKER raw roll-up, centered): RS {x.e_rs.std():.1f}  RA {x.e_ra.std():.1f}")

    print("\n1) DH: mean RS error (actual - projected, runs/162) by group")
    print(x.groupby([x.season < 2022, x.team_id.isin(AL)]).e_rs.agg(["mean", "std", "size"]).round(1)
          .rename_axis(["pre2022", "AL"]).to_string())
    print(" mean V3 win error, NL pre-2022 vs rest:", x.groupby("nl_pre").e_w.mean().round(2).to_dict())

    print("\n2) team defense")
    for col in ("def_1c", "def_0c"):
        m = x[col].notna()
        b = np.polyfit(x.loc[m, col], x.loc[m, "e_ra"], 1)[0]
        print(f"  RA error vs {col}: r {np.corrcoef(x.loc[m, col], x.loc[m, 'e_ra'])[0, 1]:+.3f}, slope {b:+.2f} (runs RA per run saved); "
              f"win error r {np.corrcoef(x.loc[m, col], x.loc[m, 'e_w'])[0, 1]:+.3f}")
    m = x.def_1c.notna() & x.def_0c.notna() & (x.season != 2021)
    print(f"  defense year-to-year r (ex-2021 lag) {np.corrcoef(x.loc[m, 'def_1c'], x.loc[m, 'def_0c'])[0, 1]:.3f}")

    print("\n3) team-level persistence of the roll-up's own errors (last year's error -> this year's)")
    rr = x[["team_id", "season", "e_rs", "e_ra"]]
    y = x.merge(rr.assign(season=rr.season + 1), on=["team_id", "season"], suffixes=("", "_1"))
    y = y[y.season != 2021]
    for s_ in ("e_rs", "e_ra"):
        print(f"  {s_}: r {np.corrcoef(y[s_ + '_1'], y[s_])[0, 1]:+.3f}  n {len(y)}")

    print("\n4) what Vegas knows that V3 doesn't: regress V3 win error on (Vegas - V3)")
    m = x.veg.notna()
    gap = x.veg - x.V3
    print(f"  r {np.corrcoef(gap[m], x.e_w[m])[0, 1]:+.3f}, slope {np.polyfit(gap[m], x.e_w[m], 1)[0]:.2f}, SD gap {gap[m].std():.2f}")
    for col in ("def_1c", "nl_pre", "e_rs", "e_ra"):
        mm = m & x[col].notna()
        print(f"  (Vegas - V3) vs {col}: r {np.corrcoef(gap[mm], x.loc[mm, col])[0, 1]:+.3f}")
    print("\nbiggest V3 misses where Vegas was much closer:")
    x["gain"] = np.abs(x.e_w) - np.abs(x.act - x.veg)
    print(x.sort_values("gain", ascending=False).head(12)[["season", "team_id", "act", "V3", "veg", "e_rs", "e_ra", "def_1c"]]
          .round(1).to_string(index=False))
    D.to_parquet(SEASON / "team_def.parquet", index=False)


if __name__ == "__main__":
    main()
