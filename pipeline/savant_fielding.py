"""Fetch Baseball Savant defense / baserunning leaderboards by season. Runs in GitHub Actions (sandbox can't reach Savant).
Measured values (Statcast): fielding run value (FRV, all positions incl. catcher framing/blocking/throwing), outs above
average (OAA), baserunning run value, catcher framing.

usage: python pipeline/savant_fielding.py 2016 2026  -> data/savant/{frv,oaa,brv,framing}.parquet
Savant's CSV URLs change; each kind tries several known forms and keeps the first that returns a real table.
"""
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "savant"
B = "https://baseballsavant.mlb.com"
CANDIDATES = {
    "frv": [
        B + "/leaderboard/fielding-run-value?gameType=Regular&seasonStart={y}&seasonEnd={y}&type=fielder&position=&minInnings=0&minResults=1&csv=true",
        B + "/leaderboard/fielding-run-value?year={y}&gameType=Regular&type=fielder&position=&minInnings=0&csv=true",
    ],
    "oaa": [
        B + "/leaderboard/outs_above_average?type=Fielder&startYear={y}&endYear={y}&split=no&team=&range=year&min=0&pos=&roles=&viz=hide&csv=true",
        B + "/outs_above_average?type=Fielder&year={y}&team=&range=year&min=0&pos=&roles=&viz=hide&csv=true",
    ],
    "brv": [
        B + "/leaderboard/baserunning-run-value?game_type=Regular&season_start={y}&season_end={y}&sortColumn=runner_runs_tot&sortDirection=desc&split=no&n=0&team=&type=Bat&with_team_only=1&csv=true",
        B + "/leaderboard/baserunning?game_type=Regular&n=0&season_end={y}&season_start={y}&split=no&team=&type=Bat&with_team_only=1&csv=true",
        B + "/leaderboard/baserunning-run-value?year={y}&type=Bat&n=0&csv=true",
    ],
    "framing": [
        B + "/catcher_framing?year={y}&team=&min=0&type=catcher&sort=4,1&csv=true",
        B + "/leaderboard/catcher-framing?type=catcher&seasonStart={y}&seasonEnd={y}&team=&min=0&sortColumn=rv_tot&sortDirection=desc&csv=true",
    ],
}
HDR = {"User-Agent": "Mozilla/5.0 (uecker research)"}


def try_url(url):
    for i in range(3):
        try:
            r = requests.get(url, headers=HDR, timeout=90)
        except requests.RequestException as e:
            print("   err", e)
            time.sleep(5 * (i + 1))
            continue
        txt = r.text.lstrip("﻿")
        if r.ok and txt.strip() and not txt.lstrip().startswith("<"):
            try:
                df = pd.read_csv(io.StringIO(txt))
            except Exception as e:  # noqa: BLE001
                print("   parse fail", e, txt[:200].replace("\n", " | "))
                return None
            df.columns = [str(c).strip().strip('"') for c in df.columns]
            return df
        print("   status", r.status_code, "html" if txt.lstrip().startswith("<") else "", txt[:120].replace("\n", " "))
        if r.status_code in (404, 400):
            return None
        time.sleep(5 * (i + 1))
    return None


def fetch(kind, y):
    for j, u in enumerate(CANDIDATES[kind]):
        url = u.format(y=y)
        df = try_url(url)
        if df is not None and len(df) >= 20:
            df["season"] = y
            df["src_url"] = j
            return df
        print(f"  {kind} {y} candidate {j}: {'none' if df is None else len(df)} rows")
    return None


def main(y0, y1):
    OUT.mkdir(parents=True, exist_ok=True)
    for kind in CANDIDATES:
        frames = []
        for y in range(y0, y1 + 1):
            df = fetch(kind, y)
            if df is None:
                print(f"{kind} {y}: NO DATA")
                continue
            print(f"{kind} {y}: {len(df)} rows; cols {list(df.columns)}")
            frames.append(df)
            time.sleep(1)
        if frames:
            out = pd.concat(frames, ignore_index=True)
            for c in out.columns:
                if out[c].dtype == object:
                    out[c] = out[c].astype(str)
            out.to_parquet(OUT / f"{kind}.parquet", index=False)
            print("wrote", kind, len(out))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
