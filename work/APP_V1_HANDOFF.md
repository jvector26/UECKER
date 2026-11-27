# UECKER app v1: build handoff (written 2026-11-26, session 12)

Goal of the session: ship v1 of the UECKER app, a separate PWA on GitHub Pages, with a **team board** and **player
cards** for 2027, styled like 1980s baseball cards. The app is engineering, not a model change: no research bar, but
**no model numbers may change** in this work. Show what the shipped model already produces.

## Start
1. add_repo jvector26/UECKER (push), clone, `bash work/setup.sh`, read `work/HANDOFF.md` (state of the model).
2. add_repo jvector26/JIMBO (read only). Find its `site/` shell: page structure, service worker / offline pattern,
   manifest, how the nightly job writes the JSON the page reads, and how Pages is deployed. Copy the pattern, not the NBA UI.
3. Both repos are public, so GitHub Pages works on the free plan.

## User decisions already made (don't re-ask)
- Separate PWA (decision 4, 2026-11-23), copy JIMBO's site shell + service-worker pattern.
- Value baseline = league AVERAGE: runs above average.
- Vegas is a yardstick only. Not shown in v1 (no live lines yet); never an input.
- Look: lean into 1980s baseball cards for fun: '87-style wood-grain border, '86-style team-name bar, '89-style
  photo-frame cards, card-back stat lines and a cartoon trivia panel. **Original design only:** no real card-maker
  logos/wordmarks, no MLB or team logos, no player photos (use an original silhouette or initials art). Team names in text are fine.
- He edits roles in the app (lineup spot, rotation slot, closer) rather than being quizzed: that's v2 (see below), but
  leave room for it in the card/team layout.
- He's on his phone: mobile-first, fast, readable at a glance. Lead with the number.

## v1 scope
1. **Team board** (home screen): 30 teams sorted by projected 2027 wins, with RS, RA, run differential, last year's
   Pythag wins. Tap a team -> team page: its projected hitters and pitchers (cards in a grid), sorted by projected runs.
2. **Player cards**: front = name, team bar, position, bats/throws, age in 2027, projected value, projected PA or BF.
   Back = last three seasons' stat line (PA, wOBA, rel-wOBA for hitters; BF, IP, K, BB, HR, RA9 for pitchers) + the 2027
   projection row + one trivia line (generated from the data, e.g. "Faced 812 batters in 2025"; no made-up facts).
   Flip on tap.
3. **Search** across players by name.
4. **Footer / about**: last data update, model labels (e.g. "Wins: V3 on winter 40-man rosters"), short glossary.
5. Offline: service worker caches shell + last JSON (JIMBO pattern).

Out of v1 (later): depth-chart editing, playoff odds, daily game odds, Vegas log, in-season updating.

## Data: what exists and what to build
- **Team board:** `data/season/team_wins_2027.parquet` (`team_id, team, wins, uecker_wins, last_pythag, rs, ra, pa_cov,
  bf_cov, inputs, model`). Built by `python model/team2.py --ship 2027`. Currently winter inputs -> model V3.
- **Playing time:** `data/season/pt_winter_2027.parquet` (`player_id` retro id, `mlbam`, `kind` hit/pit, `proj` = PA or BF).
- **Team assignment:** `data/statsapi/rosters_winter_2027.parquet` (Dec-1 40-man; `team_id, mlbam, pos, status`).
  Use `model/team.py` `roster_team(2027, winter=True)`.
- **Names / bio:** `data/statsapi/people.parquet` (`mlbam, name, birth, bats, throws, pos, debut`).
- **History for card backs:** `data/season/bat.parquet`, `pit.parquet` (by `player_id`, `season`, includes `mlbam`).
- **2027 player rates: NOT yet exported. Build `pipeline/export_app.py`** that reuses the shipped code exactly as
  `team2.ship` does (do not reimplement the model):
  - hitters: `team.hitter_rates(bat, rv, xc, 2027, None)` -> projected rel-wOBA (vs league, park-unadjusted; curve=None
    loads the shipped age curve). Runs above average per 600 PA = rel_wOBA / `rv.loc[2026].woba_scale` x 600.
  - pitchers: `team.pitcher_rates(team.pitcher_frames(upto=2026), 2027)` -> projected rel-RA9 (+ = worse). Runs saved
    per 200 IP = -rel_RA9 x 200 / 9.
  - Season totals = rate x projected PA/BF (pitcher runs = -rel_RA9 x BF x outs-per-BF / 27; outs-per-BF from 2026,
    as in `team.project_season`).
  - **Validation (required):** rolling the exported players up by team with `team.rollup(..., mode="N2")` plus the same filler and
    centering as `team.project_season`, must reproduce `team_wins_2027.parquet` rs/ra (before the V3 stretch/blend) to rounding. If not, the export is wrong.
  - Write `site/data/teams.json`, `site/data/players.json` (compact; one file per team if large), `site/data/meta.json`
    (build time, model labels, inputs).
- Players with PT but no team (unsigned FAs in winter): put them in a "Free agents" pseudo-team page.

## Nightly / deploy
- Offseason: rebuild only when inputs change (winter roster pull via backfill job `winter`, then `team2.py --ship 2027`
  and `export_app.py`). Add the export as a step after the ship in whatever workflow rebuilds the board.
- Add a Pages deploy workflow (JIMBO's pattern). From the NBA project: scheduled GitHub cron ran 3-7 h late; on-time runs
  came from a Claude scheduled task dispatching via REST (`gh api -X POST .../dispatches`; GraphQL is blocked, so
  `gh workflow run` fails). Not needed until in-season daily updates.
- Template first, never patch the built page: page source in `site/` (or `site/src/`), data JSON generated.

## Checks before calling v1 done
- Export validation above passes; spot-check 3 hitters and 3 pitchers against the model by hand.
- Open the deployed page on a phone-sized viewport (screenshot it); cards flip; search works; offline reload works.
- No logos, photos, or real card-brand marks anywhere.
- Update `work/HANDOFF.md` (new "App" section: URL, how to rebuild, files) and a project note; commit, push.
