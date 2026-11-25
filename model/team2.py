"""S11-02: U1 (S11-01 N2) + runs-allowed spread stretch + blend with last-year Pythag, constants LOSO. See RESEARCH_LOG.
usage: python model/team2.py | tee work/results/s11_02_team2.txt   (needs work/results/s11_01_preds.parquet)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from park_test import clustered_t  # noqa: E402
from team import RES, SCORED, pythag, team_results  # noqa: E402


def stretch(R, s):
    """re-center RA around the season mean, scale by s, Pythagenpat, re-center wins to 81."""
    out = pd.Series(np.nan, index=R.index)
    for N, g in R.groupby("season"):
        ra = g.ra_p.mean() + s[N] * (g.ra_p - g.ra_p.mean())
        w = 162 * pythag(g.rs_p, ra, 162)
        out[g.index] = w - w.mean() + 81
    return out


def loso_slope(x, y, season, N):
    m = season.isin(SCORED) & (season != N)
    return float(np.polyfit(x[m], y[m], 1)[0])


def run():
    R = pd.read_parquet(RES / "s11_01_preds.parquet")
    res = pd.concat([team_results(y) for y in range(2015, 2025)])
    res["p1"] = 162 * pythag(res.rs, res.ra, res.g)
    R = R.merge(res[["team_id", "season", "p1"]].assign(season=res.season + 1), on=["team_id", "season"], how="left")
    c = lambda col: R[col] - R.groupby("season")[col].transform("mean")  # noqa: E731
    rac, rap = c("ra_a"), c("ra_p")
    s = {N: loso_slope(rap, rac, R.season, N) for N in R.season.unique()}
    R["U2"] = stretch(R, s)
    coef = {}
    for arm, u in (("V1", "N2"), ("V3", "U2")):
        R[arm] = np.nan
        for N in R.season.unique():
            m = R.season.isin(SCORED) & (R.season != N)
            X = np.c_[R.loc[m, u] - 81, R.loc[m, "p1"] - 81]
            b = np.linalg.lstsq(X, R.loc[m, "act"] - 81, rcond=None)[0]
            coef[(arm, N)] = b
            i = R.season == N
            R.loc[i, arm] = 81 + b[0] * (R.loc[i, u] - 81) + b[1] * (R.loc[i, "p1"] - 81)
    R["V2"] = R.U2
    print("RA stretch s (LOSO):", {int(k): round(v, 3) for k, v in sorted(s.items())})
    print("blend (b_u, b_p) LOSO:", {f"{a} {n}": tuple(np.round(b, 3)) for (a, n), b in sorted(coef.items()) if a == "V3"})
    print("V1 blend:", {n: tuple(np.round(b, 3)) for (a, n), b in sorted(coef.items()) if a == "V1"})
    R.to_parquet(RES / "s11_02_preds.parquet", index=False)

    x = R[R.season.isin(SCORED)]
    e1, e2 = x.season <= 2019, x.season >= 2021
    rm = lambda g, k: float(np.sqrt(np.mean((g.act - g[k]) ** 2)))  # noqa: E731
    t = lambda g, b, k: clustered_t(None, None, g.act - g[b], g.act - g[k], np.ones(len(g)), g.season)  # noqa: E731
    print("\nwins per 162, 240 team-seasons. t season-clustered, positive = arm better")
    print("| arm | RMSE | MAE | t vs B1 | t vs B2 | t vs U1 | 2016-19 (t vs B2) | 2021-24 (t vs B2) | slope | SD |")
    for k in ["B1", "B2", "N2", "V1", "V2", "V3"]:
        f = lambda g, b: "" if k == b else f"{t(g, b, k):.2f}"  # noqa: E731
        print(f"| {k} | {rm(x, k):.2f} | {np.mean(np.abs(x.act - x[k])):.2f} | {f(x, 'B1')} | {f(x, 'B2')} | {f(x, 'N2')} | "
              f"{rm(x[e1], k):.2f} ({f(x[e1], 'B2')}) | {rm(x[e2], k):.2f} ({f(x[e2], 'B2')}) | {np.polyfit(x[k], x.act, 1)[0]:.2f} | {x[k].std():.2f} |")
    print("\nby season RMSE:")
    print(R.groupby("season").apply(lambda g: pd.Series({k: rm(g, k) for k in ["B2", "N2", "V1", "V2", "V3"]}), include_groups=False).round(2).to_string())
    return R


def fit_final(R):
    """S11-02 V3 constants on all 8 scored seasons (shipped, provisional; user OK 2026-11-24)."""
    x = R[R.season.isin(SCORED)]
    c = lambda col: x[col] - x.groupby("season")[col].transform("mean")  # noqa: E731
    s = float(np.polyfit(c("ra_p"), c("ra_a"), 1)[0])
    u2 = stretch(x, {N: s for N in SCORED})
    b = np.linalg.lstsq(np.c_[u2 - 81, x.p1 - 81], x.act - 81, rcond=None)[0]
    return {"s_ra": s, "b_u": float(b[0]), "b_p": float(b[1])}


def ship(N):
    """Provisional V3 board for season N from the current playing-time file (winter stand-in until opening day)."""
    import json

    import team as T
    R = pd.read_parquet(RES / "s11_02_preds.parquet")
    k = fit_final(R)
    (T.SEASON / "team_v3.json").write_text(json.dumps(k, indent=1))
    od = T.API / f"rosters_{N}.parquet"
    winter = not od.exists()
    ptf = T.SEASON / (f"pt_winter_{N}.parquet" if winter else f"pt_{N}.parquet")
    pt = pd.read_parquet(ptf)
    P = pt[["player_id", "mlbam", "season", "kind"]].assign(**{"w1" if winter else "c2": pt.proj})
    bat = pd.read_parquet(T.SEASON / "bat.parquet")
    rv = pd.read_parquet(T.SEASON / "run_values.parquet").set_index("season")
    xc = pd.read_parquet(T.SEASON / "xcon3.parquet")
    PFAC = pd.read_parquet(T.SEASON / "park_factors.parquet")
    hist = pd.read_parquet(RES / "s9_02_preds.parquet")[["player_id", "mlbam", "season", "kind"]]
    res = pd.concat([team_results(y) for y in range(N - 1, N)])
    teams = np.sort(T.roster_team(N, winter).unique())
    ctx = (bat, rv, xc, None, T.pitcher_frames(upto=N - 1), pd.concat([hist, P]), res, PFAC)
    sched = N if (T.API / f"games_{N}.parquet").exists() else N - 1
    pfa = min(N, int(PFAC["asof"].max()))
    o, f = T.project_season(N, ctx, winter=winter, modes=("N2",), teams=teams, sched=sched, pf_asof=pfa, curve="ship")
    d = o[("N2", True)]
    ra = d.ra.mean() + k["s_ra"] * (d.ra - d.ra.mean())
    w = 162 * pythag(d.rs, ra, 162)
    u2 = w - w.mean() + 81
    last = res.set_index("team_id")
    p1 = (162 * pythag(last.rs, last.ra, last.g)).reindex(teams)
    v3 = 81 + k["b_u"] * (u2 - 81) + k["b_p"] * (p1 - 81)
    g = pd.read_parquet(T.API / f"games_{N - 1}.parquet")
    name = dict(zip(g.home_id, g.home))
    out = pd.DataFrame({"season": N, "team_id": teams, "team": [name.get(t, str(t)) for t in teams], "wins": v3.values,
                        "uecker_wins": u2.values, "last_pythag": p1.values, "rs": d.rs.values, "ra": ra.values,
                        "pa_cov": d.pa_cov.values, "bf_cov": d.bf_cov.values,
                        "inputs": "winter 40-man + S9-03" if winter else "opening day + S9-02"})
    out.to_parquet(T.SEASON / f"team_wins_{N}.parquet", index=False)
    print("constants:", {a: round(b, 3) for a, b in k.items()}, "| inputs:", out.inputs.iloc[0], "| park factors asof", pfa,
          "| schedule", sched, "| filler", np.round(f, 4))
    print(out.sort_values("wins", ascending=False)[["team", "wins", "uecker_wins", "last_pythag", "rs", "ra", "pa_cov", "bf_cov"]]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--ship":
        ship(int(sys.argv[2]))
    else:
        run()
