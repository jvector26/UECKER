"""Fetch Statcast sprint speed leaderboards (Baseball Savant) for a range of seasons. Runs in GitHub Actions
(the Claude sandbox can't reach Savant). Measured values: sprint speed (ft/s, fastest 1-s window on competitive runs),
home-to-first time, competitive-run count.

usage: python pipeline/savant_running.py 2015 2026   -> data/savant/sprint_speed.parquet
"""
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "savant"
URL = "https://baseballsavant.mlb.com/leaderboard/sprint_speed?year={y}&position=&team=&min=0&csv=true"


def fetch(y):
    for i in range(4):
        r = requests.get(URL.format(y=y), headers={"User-Agent": "Mozilla/5.0 (uecker research)"}, timeout=60)
        if r.ok and r.text.strip():
            df = pd.read_csv(io.StringIO(r.text.lstrip("﻿")))
            df.columns = [c.strip().strip('"') for c in df.columns]
            df["season"] = y
            return df
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"sprint speed {y}: HTTP {r.status_code}")


def main(y0, y1):
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for y in range(y0, y1 + 1):
        df = fetch(y)
        print(y, len(df), "players; cols:", list(df.columns))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT / "sprint_speed.parquet", index=False)
    print("wrote", len(out), "rows")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
