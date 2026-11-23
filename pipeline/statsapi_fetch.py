"""MLB Stats API pulls (runs in GitHub Actions; statsapi.mlb.com is not reachable from the Claude sandbox).

usage: python pipeline/statsapi_fetch.py games 2015 2026   -> data/statsapi/games_<year>.parquet (results + probables)
       python pipeline/statsapi_fetch.py season 2015 2026  -> data/statsapi/{hitting,pitching}_<year>.parquet (player totals)
       python pipeline/statsapi_fetch.py people             -> data/statsapi/people.parquet (MLBAM id, birthdate, bats/throws)
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "statsapi"
API = "https://statsapi.mlb.com/api/v1"
S = requests.Session()


def get(path, **params):
    for i in range(5):
        r = S.get(f"{API}/{path}", params=params, timeout=60)
        if r.ok:
            return r.json()
        time.sleep(2 ** i)
    r.raise_for_status()


def games(year):
    j = get("schedule", sportId=1, season=year, gameType="R", hydrate="probablePitcher,venue,decisions")
    rows = []
    for d in j.get("dates", []):
        for g in d["games"]:
            a, h = g["teams"]["away"], g["teams"]["home"]
            rows.append({"game_pk": g["gamePk"], "date": d["date"], "game_num": g.get("gameNumber"),
                         "double_header": g.get("doubleHeader"), "status": g["status"]["detailedState"],
                         "away_id": a["team"]["id"], "home_id": h["team"]["id"],
                         "away": a["team"]["name"], "home": h["team"]["name"],
                         "away_score": a.get("score"), "home_score": h.get("score"),
                         "away_sp": (a.get("probablePitcher") or {}).get("id"),
                         "home_sp": (h.get("probablePitcher") or {}).get("id"),
                         "venue_id": g["venue"]["id"], "venue": g["venue"].get("name"),
                         "start_utc": g.get("gameDate"), "day_night": g.get("dayNight"),
                         "scheduled_innings": g.get("scheduledInnings")})
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / f"games_{year}.parquet", index=False)
    print("games", year, len(df), "final", (df.status == "Final").sum())


def season(year):
    for group in ["hitting", "pitching"]:
        j = get("stats", stats="season", group=group, season=year, sportId=1, gameType="R", playerPool="ALL", limit=5000)
        rows = []
        for s in j["stats"][0]["splits"]:
            r = {"mlbam": s["player"]["id"], "name": s["player"]["fullName"], "team_id": (s.get("team") or {}).get("id")}
            r.update({k: v for k, v in s["stat"].items() if isinstance(v, (int, float, str))})
            rows.append(r)
        df = pd.DataFrame(rows)
        df.to_parquet(OUT / f"{group}_{year}.parquet", index=False)
        print(group, year, len(df))


def people(y0=2010, y1=2026):
    ids = set()
    for y in range(y0, y1 + 1):
        for group in ["hitting", "pitching"]:
            f = OUT / f"{group}_{y}.parquet"
            if f.exists():
                ids |= set(pd.read_parquet(f).mlbam)
    ids = sorted(ids)
    rows = []
    for i in range(0, len(ids), 150):
        j = get("people", personIds=",".join(map(str, ids[i:i + 150])))
        for p in j["people"]:
            rows.append({"mlbam": p["id"], "name": p["fullName"], "birth": p.get("birthDate"),
                         "bats": (p.get("batSide") or {}).get("code"), "throws": (p.get("pitchHand") or {}).get("code"),
                         "pos": (p.get("primaryPosition") or {}).get("abbreviation"), "debut": p.get("mlbDebutDate")})
    pd.DataFrame(rows).to_parquet(OUT / "people.parquet", index=False)
    print("people", len(rows))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1]
    if what == "people":
        people()
    else:
        for y in range(int(sys.argv[2]), int(sys.argv[3]) + 1):
            {"games": games, "season": season}[what](y)
