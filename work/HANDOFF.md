# UECKER HANDOFF (current state, single source of truth)
Unified Expected Contribution & Key Efficiency Ratings: ground-up MLB model for 2027 (sibling of NBA JIMBO).
Last updated: 2026-11-23, session 1. Original brief: `work/START_HANDOFF_2026-11-23.md`.

## Start of every chat
1. add_repo jvector26/UECKER (push), clone, `bash work/setup.sh` (add `RETRO=1` only to rebuild Retrosheet parquet).
2. Read this file, then the bottom of `work/RESEARCH_LOG.md`.
3. End of session: update both, commit, push.

## User working style
- Knows the sport well. Test his intuitions against data and report honestly, including when he's right.
- Never change the model without his OK. Report a failed test plainly.
- Can't run code; Claude runs everything. Usually on his phone: short answers, lead with the result.
- No public projections (ZiPS / Steamer / THE BAT / PECOTA / FanGraphs WAR) as training targets. Yardstick only, with his OK. Doesn't bet.
- Edits roles in the app (lineup spot, rotation slot, closer) rather than being quizzed about them.

## Lessons carried from NBA JIMBO
Repo from day one; one source of truth (this file + research log); template first, never patch the built page;
nightly = Actions, but on-time runs come from a Claude scheduled task dispatching via REST (GitHub cron was 3-7 h late,
GraphQL blocked so `gh workflow run` fails); playing time beats rating tweaks (SP, lineup, bullpen, IL early);
hold out recent seasons; use pregame info only in game tests (endogeneity); Vegas = yardstick, logged from game 1.

## Network split (verified 2026-11-23)
| Source | Sandbox | Actions | Notes |
|---|---|---|---|
| Chadwick register (raw.githubusercontent) | yes | yes | in `data/ref/people.parquet` (IDs: retro/mlbam/bbref/fangraphs) |
| Retrosheet via chadwickbureau/retrosheet (git) | yes | yes | events 1871-2025 + game logs; **2026 not posted yet** (only schedule) |
| Chadwick tools (build from source) | yes | yes | `cwevent`, `cwdaily` in ~/.local/bin |
| PyPI (pybaseball etc.) | yes | yes | |
| statsapi.mlb.com | **no** | yes | `pipeline/statsapi_fetch.py` |
| baseballsavant (Statcast) | **no** | yes | `pipeline/statcast_fetch.py` |
| retrosheet.org, fangraphs.com | no | yes | not needed so far |
| chadwickbureau/baseballdatabank (Lahman) | not found | ? | not needed: Retrosheet covers season totals |

## Data in the repo
- `data/retro/daily_<y>.parquet` player-game lines 2010-2025 (cwdaily, all fields incl. fielding)
- `data/retro/pa_<y>.parquet` event rows 2010-2025 (cwevent subset: base-out state, batter/pitcher + hands, event, batted-ball type/location, runner dests)
- `data/retro/gamelog_<y>.parquet` game results + starting pitchers
- `data/season/run_values.parquet` UECKER's own RE24 linear weights + wOBA weights per season 2010-2025
- `data/season/bat.parquet`, `pit.parquet` player-seasons with wOBA, RA9, league-relative versions, age
- `data/ref/people.parquet` Chadwick ID crosswalk; `data/ref/retro_bio.csv` birthdates/hands
- Pending (need Actions): `data/statcast/<y>_<mm>.parquet` 2015-2026, `data/statsapi/*` (2026 season, games, people)

## Production chain (so far)
`pipeline/retro_build.py 2010 2025` -> `model/run_values.py` -> `model/seasons.py` -> `model/marcel.py` (scores the bar)

## Research bar (PROPOSED, needs user OK)
1. Baseline to beat: Marcel (public method, not a target). A component that doesn't beat Marcel out of sample doesn't ship.
2. Player test: project N from < N; hitters wOBA - lg wOBA, PA-weighted RMSE (PA>=100); pitchers RA9 - lg RA9,
   BF-weighted RMSE (BF>=100). Targets 2016-2024, 2020 reported separately. League-relative so run-environment shifts don't count.
3. Game test: pregame info only (probable SP, posted lineup); log loss / run-margin RMSE; pass = better pooled,
   season-clustered t > 2.5, same sign in 2016-20 and 2021-24.
4. Team test: preseason projected wins vs actual 2016-2024, Vegas win totals beside it.
5. Holdout 2025-2026: not scored until the pre-season final check (`HOLDOUT=1`).
6. Pre-register expected sign/size in RESEARCH_LOG before each run. Partial weight in proportion to evidence over binary ship/reject.

## Current numbers (S1-02, targets 2016-2024 ex-2020)
- Hitters rel-wOBA RMSE: Marcel **.0341**, league avg .0393, last season only .0539.
- Pitchers rel-RA9 RMSE: league avg **1.309**, Marcel 1.325 (Marcel loses; 2x regression 1.288). RA9 is mostly noise.

## Decisions (user, 2026-11-23)
1. Headline = pure UECKER; Vegas is a yardstick only (carried from the NBA project, not re-asked).
2. Value baseline = league AVERAGE (runs above average), as in the NBA app.
3. Opening-day scope = season projections + win totals AND daily game odds.
4. App = separate PWA (copy JIMBO's site shell + service-worker pattern).
5. Pitcher bar = a new pitcher model must beat ALL THREE: published-reading Marcel (1.325), Marcel 2x regression (1.288), league average (1.309).
Still open: approve the research bar above as a whole; history depth (2010+ in, extend to ~2000 only if a test needs it).

## Backlog (next)
- BLOCKER: Claude GitHub App lacks access to jvector26/UECKER (push 403). User adds the repo in the App config. Then push; dispatch backfill: `statsapi 2015 2026`, then `statcast` 2015-2026 (4 parallel).
- Map Retrosheet <-> MLBAM ids; add 2026 season from Stats API/Statcast to bat/pit tables (Retrosheet 2026 events arrive later).
- Park factors (multi-year, handedness split) from Retrosheet events.
- Statcast components for hitters (K%, BB%, EV/LA/barrel, xwOBAcon) and pitchers (K-BB%, stuff); test vs Marcel.
- Historical win totals / closing lines source (yardstick).
- Lockout watch: CBA expires 2026-12-01; transactions freeze. Roster-dependent work waits.

## Rejected ideas
(none yet)
