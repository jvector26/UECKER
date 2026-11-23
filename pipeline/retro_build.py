"""Build per-season parquet from Retrosheet event files (Chadwick mirror).

Inputs : ext/retrosheet/seasons/<year>/  (sparse checkout, see work/setup.sh)
Outputs: data/retro/daily_<year>.parquet   player-game lines (cwdaily, all fields)
         data/retro/pa_<year>.parquet      event-level rows (cwevent, selected fields)
         data/retro/gamelog_<year>.parquet game logs (GLyyyy.TXT)
Regular season only: team files <year><TEAM>.EV[AN]. Postseason/All-Star files are skipped.

usage: python pipeline/retro_build.py 2010 2025
"""
import io
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RETRO = ROOT / "ext" / "retrosheet" / "seasons"
OUT = ROOT / "data" / "retro"
BIN = Path.home() / ".local" / "bin"

EVENT_FIELDS = "0-4,8-17,26-28,30,32-37,40,43,47,48,50,58-61,96"


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return r.stdout


def team_files(d, year):
    return sorted(p.name for p in d.iterdir() if re.fullmatch(rf"{year}[A-Z]{{3}}\.EV[AN]", p.name))


def build(year):
    d = RETRO / str(year)
    files = team_files(d, year)
    assert len(files) == 30, (year, len(files))
    daily = pd.read_csv(io.StringIO(run([str(BIN / "cwdaily"), "-n", "-y", str(year), *files], d)), low_memory=False)
    daily.columns = [c.strip().lower() for c in daily.columns]
    daily.to_parquet(OUT / f"daily_{year}.parquet", index=False)

    ev = pd.read_csv(io.StringIO(run([str(BIN / "cwevent"), "-n", "-y", str(year), "-f", EVENT_FIELDS, *files], d)),
                     low_memory=False)
    ev.columns = [c.strip().lower() for c in ev.columns]
    ev.to_parquet(OUT / f"pa_{year}.parquet", index=False)

    gl = next((p for p in d.iterdir() if p.name.upper() == f"GL{year}.TXT"), None)
    if gl is not None:
        g = pd.read_csv(gl, header=None, low_memory=False)
        cols = {0: "date", 1: "game_num", 3: "vis", 6: "home", 9: "vis_r", 10: "home_r", 16: "park",
                17: "attendance", 18: "minutes", 101: "vis_sp", 103: "home_sp"}
        g = g[list(cols)].rename(columns=cols)
        g.to_parquet(OUT / f"gamelog_{year}.parquet", index=False)
    print(year, "games", daily.game_id.nunique(), "player-games", len(daily), "events", len(ev))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    a, b = int(sys.argv[1]), int(sys.argv[2])
    for y in range(a, b + 1):
        build(y)
