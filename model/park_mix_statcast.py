"""Player-season park mix for seasons Retrosheet hasn't posted (2026), from Statcast PA rows + Stats API venues.

Venue -> Retrosheet park_id map is learned on 2025 (Stats API game matched to the Retrosheet game log by date, scores,
home starter) and validated by rebuilding the 2025 mix from Statcast and comparing to the Retrosheet-built mix.
PA = last pitch of each at-bat with a PA-ending event (same definition as model/run_values_statcast.py).

usage: python model/park_mix_statcast.py 2026   -> validates on 2025, appends 2026 rows to data/season/park_mix.parquet
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_values_statcast import EV, load_pa  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEASON = ROOT / "data" / "season"
API = ROOT / "data" / "statsapi"


# venues not used in 2025 but with a Retrosheet park id from earlier seasons
MANUAL = {"Tropicana Field": "STP01"}


def id_map():
    p = pd.read_parquet(ROOT / "data" / "ref" / "people.parquet").dropna(subset=["key_mlbam", "key_retro"])
    return pd.Series(p.key_retro.values, index=p.key_mlbam.astype(int).values)


def venue_park_map():
    gl = pd.read_parquet(ROOT / "data" / "retro" / "gamelog_2025.parquet")
    gl["date"] = pd.to_datetime(gl.date.astype(str)).dt.strftime("%Y-%m-%d")
    g = pd.read_parquet(API / "games_2025.parquet")
    g = g[g.status != "Postponed"].copy()
    g["home_sp_r"] = g.home_sp.astype("Int64").map(id_map())
    m = g.merge(gl, left_on=["date", "home_score", "away_score", "home_sp_r"], right_on=["date", "home_r", "vis_r", "home_sp"],
                how="inner")
    vm = m.groupby(["venue_id", "venue", "park"]).size().rename("n").reset_index().sort_values("n").groupby("venue_id").tail(1)
    print(f"2025 games matched to Retrosheet: {len(m)}/{len(g)}; venues mapped: {len(vm)}")
    return vm.set_index("venue_id").park, vm


def mix(year, vmap):
    pa = load_pa(year)
    pa = pa[pa.ev.notna()].copy()
    g = pd.read_parquet(API / f"games_{year}.parquet")
    g = g[g.status != "Postponed"].drop_duplicates("game_pk")[["game_pk", "venue_id", "venue"]]
    pa = pa.merge(g, on="game_pk", how="left")
    pa["park_id"] = pa.venue_id.map(vmap).fillna(pa.venue.map(MANUAL))
    miss = pa[pa.park_id.isna()].groupby("venue").size()
    if len(miss):
        print(f"{year} neutral-site venues (PA, park factor 1.0):", miss.to_dict())
    pa["park_id"] = pa.park_id.fillna("NEU" + pa.venue_id.astype(str))
    im = id_map()
    out = []
    for col, role in [("batter", "bat"), ("pitcher", "pit")]:
        d = pa.groupby([col, "park_id", "stand"]).size().rename("n").reset_index()
        d["player_id"] = d[col].map(im).fillna("m" + d[col].astype(str))
        d = d.rename(columns={"stand": "hand"})[["player_id", "park_id", "hand", "n"]]
        d["role"], d["season"] = role, year
        out.append(d)
    return pd.concat(out, ignore_index=True)[["season", "player_id", "park_id", "hand", "n", "role"]]


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or [2026]
    vmap, vm = venue_park_map()
    # validate on 2025
    r = pd.read_parquet(SEASON / "park_mix.parquet")
    r25 = r[r.season == 2025]
    s25 = mix(2025, vmap)
    k = ["player_id", "park_id", "hand", "role"]
    c = r25.merge(s25, on=k, how="outer", suffixes=("_r", "_s")).fillna({"n_r": 0, "n_s": 0})
    tot = c.n_r.sum()
    print(f"2025 check: retro PA rows {int(c.n_r.sum())}, statcast {int(c.n_s.sum())}, "
          f"abs diff {int((c.n_r - c.n_s).abs().sum())} ({(c.n_r - c.n_s).abs().sum() / tot:.4%})")
    r = r[~r.season.isin(years)]
    new = [mix(y, vmap) for y in years]
    for y, d in zip(years, new):
        print(y, d.groupby("role").n.sum().to_dict(), "parks:", d.park_id.nunique())
    pd.concat([r] + new, ignore_index=True).to_parquet(SEASON / "park_mix.parquet", index=False)
