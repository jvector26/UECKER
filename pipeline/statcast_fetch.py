"""Fetch Statcast pitch-level data (Baseball Savant via pybaseball) for one season, one parquet per month.
Runs in GitHub Actions (the Claude sandbox can't reach Savant).

usage: python pipeline/statcast_fetch.py 2024            -> data/statcast/2024_03.parquet ... 2024_11.parquet
Keeps a column subset so each file stays well under GitHub's 100 MB limit.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pybaseball import cache, statcast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "statcast"

COLS = ["game_pk", "game_date", "game_type", "home_team", "away_team", "inning", "inning_topbot", "at_bat_number",
        "pitch_number", "batter", "pitcher", "stand", "p_throws", "balls", "strikes", "outs_when_up", "on_1b", "on_2b",
        "on_3b", "pitch_type", "release_speed", "release_spin_rate", "release_extension", "release_pos_x",
        "release_pos_z", "spin_axis", "arm_angle", "pfx_x", "pfx_z", "plate_x", "plate_z", "sz_top", "sz_bot", "zone",
        "description", "type", "events", "bb_type", "launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y",
        "estimated_woba_using_speedangle", "estimated_ba_using_speedangle", "woba_value", "woba_denom", "babip_value",
        "iso_value", "launch_speed_angle", "bat_speed", "swing_length", "delta_run_exp", "home_score", "away_score",
        "fielder_2", "fielder_3", "fielder_4", "fielder_5", "fielder_6", "fielder_7", "fielder_8", "fielder_9",
        "if_fielding_alignment", "of_fielding_alignment"]


def month_ranges(year):
    d = date(year, 3, 1)
    while d.year == year and d.month <= 11:
        nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield d, min(nxt - timedelta(days=1), date.today() - timedelta(days=1))
        d = nxt


def main(year):
    cache.enable()
    OUT.mkdir(parents=True, exist_ok=True)
    for a, b in month_ranges(year):
        if a > b:
            break
        df = statcast(start_dt=a.isoformat(), end_dt=b.isoformat(), verbose=False, parallel=True)
        if df is None or df.empty:
            print(year, a.month, "empty")
            continue
        df = df[[c for c in COLS if c in df.columns]]
        df = df[df.game_type == "R"]
        if df.empty:
            continue
        df.to_parquet(OUT / f"{year}_{a.month:02d}.parquet", index=False, compression="zstd")
        print(year, a.month, len(df), "pitches", df.game_pk.nunique(), "games")


if __name__ == "__main__":
    main(int(sys.argv[1]))
