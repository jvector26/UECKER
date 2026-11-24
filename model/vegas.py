"""Vegas preseason win totals beside UECKER (yardstick only, never an input). usage: python model/vegas.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from park_test import clustered_t  # noqa: E402
from team import API, RES, SCORED  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALIAS = {"Cleveland Indians": "Cleveland Guardians", "Oakland Athletics": "Athletics", "Florida Marlins": "Miami Marlins",
         "Los Angeles Angels of Anaheim": "Los Angeles Angels", "Anaheim Angels": "Los Angeles Angels", "St Louis Cardinals": "St. Louis Cardinals"}


def load():
    v = pd.read_parquet(ROOT / "data" / "vegas" / "win_totals_raw.parquet")
    v = v[v.table == 0][["season", "Team", "Win Total", "Actual Wins"]].rename(columns={"Team": "team", "Win Total": "line", "Actual Wins": "aw"})
    v["line"] = pd.to_numeric(v.line, errors="coerce")
    v["aw"] = pd.to_numeric(v.aw, errors="coerce")
    names = {}
    for y in range(2016, 2027):
        g = pd.read_parquet(API / f"games_{y}.parquet")
        for a, b in ((g.home, g.home_id), (g.away, g.away_id)):
            names.update(dict(zip(a.map(lambda s: ALIAS.get(s, s)), b)))
    v["team_id"] = v.team.map(lambda s: ALIAS.get(s, s)).map(names)
    return v


if __name__ == "__main__":
    v = load()
    print("unmatched:", v[v.team_id.isna()].team.unique().tolist())
    v.to_parquet(ROOT / "data" / "vegas" / "win_totals.parquet", index=False)
    R = pd.read_parquet(RES / "s11_02_preds.parquet").merge(v[["season", "team_id", "line", "aw"]], on=["season", "team_id"], how="left")
    R["games"] = np.where(R.season == 2020, 60, 162)
    R["VEG_raw"] = R.line * 162 / R.games
    R["VEG"] = R.VEG_raw - R.groupby("season").VEG_raw.transform("mean") + 81
    x = R[R.season.isin(SCORED)]
    print("lines found:", x.line.notna().sum(), "/", len(x), " actual-wins check (site vs ours, max |diff| on 162-g seasons):",
          float((x.aw - x.act).abs().max()), " mean line sum per season:", round(x.groupby("season").line.sum().mean(), 1))
    rm = lambda g, k: float(np.sqrt(np.mean((g.act - g[k]) ** 2)))  # noqa: E731
    t = lambda g, b, k: clustered_t(None, None, g.act - g[b], g.act - g[k], np.ones(len(g)), g.season)  # noqa: E731
    e1, e2 = x.season <= 2019, x.season >= 2021
    print("\n| arm | RMSE | 2016-19 | 2021-24 | t VEG vs arm |")
    for k in ["B2", "N2", "V1", "V3", "VEG_raw", "VEG"]:
        print(f"| {k} | {rm(x, k):.2f} | {rm(x[e1], k):.2f} | {rm(x[e2], k):.2f} | {'' if k.startswith('VEG') else f'{t(x, k, chr(86)+chr(69)+chr(71)):.2f}'} |")
    print("corr V3 vs VEG:", round(np.corrcoef(x.V3, x.VEG)[0, 1], 3), " SD VEG", round(x.VEG.std(), 2))
    print("\nby season RMSE:")
    print(R.groupby("season").apply(lambda g: pd.Series({k: rm(g, k) for k in ["B2", "V3", "VEG"]}), include_groups=False).round(2).to_string())
