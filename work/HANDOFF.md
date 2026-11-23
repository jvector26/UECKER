# UECKER HANDOFF (current state, single source of truth)
Unified Expected Contribution & Key Efficiency Ratings: ground-up MLB model for 2027 (sibling of NBA JIMBO).
Last updated: 2026-11-23, session 5. Original brief: `work/START_HANDOFF_2026-11-23.md`.

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
| Savant sprint speed leaderboard | **no** | yes | `pipeline/savant_running.py`, backfill job `running` |
| retrosheet.org, fangraphs.com | no | yes | not needed so far |
| chadwickbureau/baseballdatabank (Lahman) | not found | ? | not needed: Retrosheet covers season totals |

## Data in the repo
- `data/retro/daily_<y>.parquet` player-game lines 2010-2025 (cwdaily, all fields incl. fielding)
- `data/retro/pa_<y>.parquet` event rows 2010-2025 (cwevent subset: base-out state, batter/pitcher + hands, event, batted-ball type/location, runner dests)
- `data/retro/gamelog_<y>.parquet` game results + starting pitchers
- `data/season/run_values.parquet` UECKER's own RE24 linear weights + wOBA weights per season 2010-2025 (Retrosheet) + 2026 (Statcast, `source` col; matches Retrosheet within .01 on 2025)
- `data/season/bat.parquet`, `pit.parquet` player-seasons 2010-2026 with wOBA, RA9, league-relative versions, age, `mlbam` (2026 from Stats API; ties out exactly to Retrosheet on 2025)
- `data/season/park_factors.parquet` (as-of-N factors from N-3..N-1, by park/hand/component + runs), `park_mix.parquet` (player-season PA by park/hand; 2010-2025 Retrosheet, 2026 Statcast, validated on 2025)
- `data/season/xcon.parquet` hitter-season actual vs expected contact value (EV x LA bins, UECKER weights) 2015-2026
- `data/savant/sprint_speed.parquet` Savant sprint speed / home-to-first / competitive runs 2015-2026 (MLBAM ids)
- `data/season/xcon3.parquet` S4-03 per hitter-season grounder/air components incl. measured-speed expected values
- `data/season/xcon2.parquet` same, split grounder/air, with speed-aware and spray-zone expected values (S4)
- `data/ref/people.parquet` Chadwick ID crosswalk; `data/ref/retro_bio.csv` birthdates/hands
- `data/statcast/<y>_<mm>.parquet` pitch-level 2015-2026, regular season (~710k pitches/season; game counts match Retrosheet; 386 MB)
- `data/statsapi/games_<y>`, `hitting_<y>`, `pitching_<y>` 2015-2026 + `people.parquet` (MLBAM birthdates/hands)

## Production chain (so far)
`pipeline/retro_build.py 2010 2025` -> `model/run_values.py` -> `model/run_values_statcast.py 2025 2026` -> `model/seasons.py` -> `model/parks.py` -> `model/marcel.py` (scores the bar), `model/park_test.py` (S2-01); `model/park_mix_statcast.py 2026`; `model/xcon.py` (S3-01); `model/xcon2.py` (S4-01/02, tests only); `model/xcon3.py` (S4-03 measured speed, test); `model/age.py` (S5-01 age curve, test); `model/age2.py` (S5-02, test)

## Research bar (APPROVED by user 2026-11-23)
1. Baseline to beat: Marcel (public method, not a target). A component that doesn't beat Marcel out of sample doesn't ship.
2. Player test: project N from < N; hitters wOBA - lg wOBA, PA-weighted RMSE (PA>=100); pitchers RA9 - lg RA9,
   BF-weighted RMSE (BF>=100). Targets 2016-2024, 2020 reported separately. League-relative so run-environment shifts don't count.
3. Game test: pregame info only (probable SP, posted lineup); log loss / run-margin RMSE; pass = better pooled,
   season-clustered t > 2.5, same sign in 2016-20 and 2021-24.
4. Team test: preseason projected wins vs actual 2016-2024, Vegas win totals beside it.
5. Holdout 2025-2026: not scored until the pre-season final check (`HOLDOUT=1`).
6. Pre-register expected sign/size in RESEARCH_LOG before each run. Partial weight in proportion to evidence over binary ship/reject.

## Current numbers (targets 2016-2024 ex-2020)
- Hitters rel-wOBA RMSE: Marcel **.0341**, league avg .0393, last season only .0539. Park-adjusted .0340 (t 1.05, not a pass; Coors over-adjusts).
  S3-01 contact quality (EV x LA, a=.3): **.0333** (t 4.70, both eras, every season better). PASSED, SHIPPED with user OK 2026-11-23.
  S4-03 MEASURED sprint speed on grounders (M-lin, a_gb .3 / a_air .1): **.03306** (t 3.46, both eras, all seasons >= S3).
  PASSED, SHIPPED with user OK 2026-11-23. New hitter baseline to beat: .03306. Fast-hitter absolute mean error worsens (-.0005 -> -.0040) but relative
  to the pool all speed groups line up (spread .0103 -> .0026). Supersedes S4-01B proxy (.03316). S4-01A fails; S4-02 spray rejected.
- Pitchers rel-RA9 RMSE: league avg **1.309**, Marcel 1.325, Marcel 2x 1.288. Park-adjusted: 1.315 / **1.279** (t 4.6 / 3.7, PASSED, SHIPPED with user OK 2026-11-23).

## Decisions (user, 2026-11-23)
1. Headline = pure UECKER; Vegas is a yardstick only (carried from the NBA project, not re-asked).
2. Value baseline = league AVERAGE (runs above average), as in the NBA app.
3. Opening-day scope = season projections + win totals AND daily game odds.
4. App = separate PWA (copy JIMBO's site shell + service-worker pattern).
5. Pitcher bar = a new pitcher model must beat ALL THREE: published-reading Marcel (1.325), Marcel 2x regression (1.288), league average (1.309).
6. Research bar approved as written (2026-11-23).
7. S2-01 pitcher park adjustment shipped (2026-11-23).
8. S3-01 hitter contact quality shipped (2026-11-23).
10. S4-03 measured sprint speed on grounders shipped (2026-11-23).
9. Measured values over surrogates or intermediate models whenever available (e.g. Statcast sprint speed, not an SB/3B speed proxy).
Still open: history depth (2010+ in, extend to ~2000 only if a test needs it).

## Shipped model components
- S2-01 (user OK 2026-11-23): pitcher projections use park-neutral history (runs PF, as-of N from N-3..N-1) and re-apply the
  target-season park mix (`model/park_test.py` has the reference implementation). Hitters stay park-unadjusted.
- S3-01 (user OK 2026-11-23): hitter Marcel history uses contact value = .3 x actual + .7 x expected (UECKER EV x LA bins,
  `data/season/xcon.parquet`); `model/xcon.py` has the reference implementation (`A_SHIP`). Pre-2015 seasons use actual.
- S4-03 (user OK 2026-11-23): supersedes S3-01's single weight. Contact split grounders (LA<10) / air. Grounder expected value
  += beta x (measured sprint speed - season mean), speed shrunk by competitive runs (k=10), beta fit per season. Weight on actual:
  grounders .3, air .1 (`ARM_SHIP`, `A_GB_SHIP`, `A_AIR_SHIP` in `model/xcon3.py`, reference implementation there).

## UI direction (user, 2026-11-23)
- When we get to the app: lean into 1980s baseball card aesthetics, for fun (think '87 Topps wood-grain border, '86 Topps team-name
  bar, '89 Upper Deck/Donruss vibes; card-back stat lines and cartoon trivia). Original design only, no real card-maker logos/wordmarks.

## Backlog (next)
- Coors hitter handling (home/road split, or hitter-specific factor). Park factors as of 2027 still need 2026 park-level rates (Statcast) folded in.
- Hitter K%/BB% stabilization.
- S5-01 hitter age curve TESTED (`model/age.py`): re-tuned Marcel shape (peak 27-28, +.004/-.005) .03303 -> .03280, t 2.08 (fails 2.5),
  mechanism clean (32+ error -.0070 -> -.0009). Half weight .03284, t 3.83 (post hoc). User: HOLD. S5-02 measured delta curve w/ job-retention + regression fixes (`model/age2.py`): FAILS (best t 1.55,
  all worse than A1; retention weighting can't see unobserved decline). A3 residual curve (measured from UECKER errors) .03280, t 2.01. AWAITING USER CALL.
- Pitcher Statcast components (K-BB%, contact allowed, stuff); must beat all three pitcher bars.
- Historical win totals / closing lines source (yardstick).
- Lockout watch: CBA expires 2026-12-01; transactions freeze. Roster-dependent work waits.

## Rejected ideas
- S4-02 spray zones (pull/center/oppo x EV x LA) as built, 2026-11-23: worse than shipped S3 (t -0.53).
