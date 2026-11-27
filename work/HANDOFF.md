# UECKER HANDOFF (current state, single source of truth)
Unified Expected Contribution & Key Efficiency Ratings: ground-up MLB model for 2027 (sibling of NBA JIMBO).
Last updated: 2026-11-26, session 12. Original brief: `work/START_HANDOFF_2026-11-23.md`.

**Next session: build app v1. Read `work/APP_V1_HANDOFF.md` after this file.**

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
| statsapi.mlb.com | **no** | yes | `pipeline/statsapi_fetch.py` (games, season, people, rosters, transactions) |
| baseballsavant (Statcast) | **no** | yes | `pipeline/statcast_fetch.py` |
| Savant sprint speed leaderboard | **no** | yes | `pipeline/savant_running.py`, backfill job `running` |
| retrosheet.org, fangraphs.com | no | yes | not needed so far |
| covers.com/sportsoddshistory (Vegas win totals) | no | yes (playwright) | `pipeline/vegas_fetch.py`, job `vegas`; user OK 2026-11-24 |
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
- `data/statsapi/rosters_<y>` (each team's first-game active + 40-man with IL status) and `transactions_<y>` 2010-2026
- `data/statsapi/games_<y>`, `hitting_<y>`, `pitching_<y>` 2010-2026 + `people.parquet` (MLBAM birthdates/hands)

## Production chain (so far)
`pipeline/retro_build.py 2010 2025` -> `model/run_values.py` -> `model/run_values_statcast.py 2025 2026` -> `model/seasons.py` -> `model/parks.py` -> `model/marcel.py` (scores the bar), `model/park_test.py` (S2-01); `model/park_mix_statcast.py 2026`; `model/xcon.py` (S3-01); `model/xcon2.py` (S4-01/02, tests only); `model/xcon3.py` (S4-03 measured speed, test); `model/age.py` (S5-01 age curve, test); `model/age2.py` (S5-02, test); `model/pcomp.py` (S6-01 pitcher components; builds `data/season/pcon.parquet`); `model/page.py` (S7-01 pitcher age, test, failed); `model/stuff_build.py` + `model/stuff.py` (S7-02 stuff, test, failed; builds `pdisc.parquet`, `stuff_oof.parquet`)

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
  PASSED, SHIPPED with user OK 2026-11-23.
  S5-01 age curve (A3 residual curve, half weight vs Marcel age factor): .03303 (fixed-constant baseline) -> **.03283** (t 3.75, both eras).
  SHIPPED with user OK 2026-11-23. **New hitter baseline to beat: .03283.** Fast-hitter absolute mean error worsens (-.0005 -> -.0040) but relative
  to the pool all speed groups line up (spread .0103 -> .0026). Supersedes S4-01B proxy (.03316). S4-01A fails; S4-02 spray rejected.
- S6-01 pitcher component model (K/uBB/HBP/contact, own EB regression each, expected contact a=.3): **1.2424** vs shipped 1.2786,
  t 7.42, both eras, all 8 seasons. PASSED, SHIPPED with user OK 2026-11-23. Role prior (C3) rejected.
- S7-03 covariance regression constants on S6-01: **1.2360** (t 3.03; 2016-19 t 1.71, 2021-24 t 4.48; all 9 seasons better; slope .82 -> .95).
  SHIPPED with user OK 2026-11-24. **New pitcher baseline to beat: 1.2360.**
- S8-01 hitter component model with covariance k (hitter version of S7-03): .03289 vs .03283 (t -0.75). FAILED. Shipped hitter
  calibration slope already .970; covariance k over-regresses contact (slope 1.036). Hitter baseline stays **.03283**.
- S8-02 history-PA prior (regress toward hitters with similar history PA): .03276 (t 2.07; 16-19 t 2.39, 21-24 t 0.73). FAILED the
  bar (t < 2.5), right direction. Fixes part-timers, over-lifts regulars (800+ mean error -.0006 -> -.0027), slope .932.
- S8-03 same prior only below 500 weighted history PA: **.03272** (t 3.90; 16-19 t 3.08, 21-24 t 2.34; slope .948; cutoffs 250/800
  also t 3.5-3.6). PASSES. Cutoff is post-hoc (flagged); re-check on holdout at pre-season final. SHIPPED with user OK 2026-11-24.
  **New hitter baseline to beat: .03272.**
- S9-01 playing time (new target: PA/BF for players with history, 0 if absent; `model/ptime.py`): boosted trees vs
  Marcel PT: hitters 140.7 vs 200.0 PA (t 15.6), pitchers 152.6 vs 183.3 BF (t 5.0), both eras. PASSES. HITTERS SHIPPED with user OK 2026-11-24; pitchers held for S9-02.
  Caveat: gain is mostly attrition; among those who played pitchers are no better (172.4 vs 170.4). OLS did better on pitchers.
- S9-02 roster + IL playing time (`model/ptime2.py`; Stats API rosters/transactions 2010-26 in `data/statsapi`): hitters
  123.5 PA (t 14.5 vs S9-01), pitchers 130.1 BF (t 8.0 vs S9-01, 12.0 vs Marcel; among those who pitched 156 vs Marcel 170).
  SHIPPED with user OK 2026-11-24 (supersedes S9-01 hitters). Needs opening-day rosters. Starters still +32 BF high.
- S10-01 starter workload (pitch counts, career-high BF, innings jump, 2020 flag, league trend + recency weights): FAILED
  (C2 t -1.12, W1 t 0.48). Bias +32 -> +9 but starter RMSE flat (224 BF = injuries). Go to team-level normalization.
- S11-01 team win totals (`model/team.py`): opening-day roster x S9-02 PT x shipped rates, N2 normalization (scale down
  over-full staffs, fill shortfall at a below-average filler rate), runs -> Pythagenpat. 10.55 wins RMSE vs last-year
  Pythag regressed 10.86 (t 0.97) and last-year wins regressed 11.08 (t 1.75). FAILED (needs t > 2.5 vs both). Actual
  win SD 13.3 in 2016-24, so everything is high. Offense side barely beats last year's RS (r .54 vs .51); run prevention
  clearly better (r .66 vs .52) but under-spread (slope 1.33). `model/team.py` is the team roll-up harness.
- S11-02 (`model/team2.py`): U1 + RA spread stretch + blend with last-year Pythag (LOSO constants): V3 10.28 (t 2.57 vs
  B1, 2.28 vs B2). FAILS narrowly (bar 2.5). Blend alone V1 10.24 (t 2.44). Stretch adds nothing. Vegas yardstick 9.66
  (`data/vegas/win_totals.parquet`, 2016-2026, backfill job `vegas`; yardstick only). SHIPPED as provisional (user OK): constants
  `data/season/team_v3.json`; `python model/team2.py --ship N` -> `data/season/team_wins_N.parquet` (2027 built from winter inputs).
- S12 diagnostics (`work/diag_s12.py`): DH fix worthless for 2027; last-year team defense adds nothing after the Pythag
  blend; roster coverage (share of team PA/BF covered by projected rostered players) predicts V3's error (thin = rebuild).
- S12-01 C1 (`model/team3.py`): V3 blend + centered PA/BF coverage terms, LOSO: **10.02** vs V3 10.28 (t 2.34, FAILS bar
  narrowly; both eras +, 7/8 seasons), vs B1/B2 t 3.69/3.89 (clears the original team bar). Vegas 9.66 (gap .62 -> .36).
  UECKER weight .58 -> .32. SHIPPED PROVISIONAL with user OK 2026-11-26, GATED: goes live only if it beats V3 on the 2025-26
  holdout at the pre-season final; otherwise V3 stays. Constants `data/season/team_c1.json` (b_u .319, b_p .347, c_pa 21.6, c_bf 28.3).
- S9-03 winter stand-in (`model/ptime3.py`; 40-man on Dec 1 of N-1 + winter FA/outright moves + IL at N-1 end, retrained):
  hitters 137.7 PA (t 4.56 vs S9-01), pitchers 150.1 BF (t 2.52, marginal; 21-24 t 1.44). Recovers only ~15% of the
  opening-day gain. SHIPPED with user OK 2026-11-24 as the pre-March model; S9-02 takes over once opening-day rosters exist.
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
11. S5-01 hitter age curve shipped: A3 residual curve at half weight (2026-11-23). S5-02 delta-method curve rejected.
9. Measured values over surrogates or intermediate models whenever available (e.g. Statcast sprint speed, not an SB/3B speed proxy).
12. S6-01 pitcher component model shipped (C2, a=.3) (2026-11-23).
13. S7-03 covariance regression constants shipped (2026-11-24). S7-01 pitcher age curve and S7-02 stuff rejected.
14. S8-03 history-PA hitter prior (below 500 PA) shipped (2026-11-24). S8-01 hitter covariance constants rejected.
15. S9-01 hitter playing-time model shipped (2026-11-24); pitcher playing time held for S9-02 (rosters + IL history).
16. S9-02 roster + IL playing time shipped for hitters and pitchers (2026-11-24). OLS variant not preferred.
17. S9-03 winter stand-in (W1) shipped for hitters and pitchers until opening-day rosters exist (2026-11-24).
18. S11-02 V3 team wins shipped as PROVISIONAL (t 2.28 vs Pythag, below bar; user OK 2026-11-24). Re-check on holdout.
19. S12-01 C1 coverage blend shipped provisional, gated on the 2025-26 holdout (user OK 2026-11-26). Opening-day inputs only.
Still open: history depth (2010+ in, extend to ~2000 only if a test needs it).

## Shipped model components
- S2-01 (user OK 2026-11-23): pitcher projections use park-neutral history (runs PF, as-of N from N-3..N-1) and re-apply the
  target-season park mix (`model/park_test.py` has the reference implementation). Hitters stay park-unadjusted.
- S3-01 (user OK 2026-11-23): hitter Marcel history uses contact value = .3 x actual + .7 x expected (UECKER EV x LA bins,
  `data/season/xcon.parquet`); `model/xcon.py` has the reference implementation (`A_SHIP`). Pre-2015 seasons use actual.
- S4-03 (user OK 2026-11-23): supersedes S3-01's single weight. Contact split grounders (LA<10) / air. Grounder expected value
  += beta x (measured sprint speed - season mean), speed shrunk by competitive runs (k=10), beta fit per season. Weight on actual:
  grounders .3, air .1 (`ARM_SHIP`, `A_GB_SHIP`, `A_AIR_SHIP` in `model/xcon3.py`, reference implementation there).

- S5-01 (user OK 2026-11-23): hitter age handling = .5 x Marcel age factor projection + .5 x (no-age projection + A3 curve,
  PA-weighted mean residual by integer projected age, [1,2,1]-smoothed). `model/age.py` `ship_hitters`; curve in
  `data/season/age_curve_hit.parquet` (fit 2016-24 ex-2020; refit at pre-season final). Pitchers still use Marcel's age factor.

- S6-01 (user OK 2026-11-23): supersedes S2-01's runs-based Marcel for pitchers. Components per BF: K, uBB, HBP, contact value
  (runs rel. to an out; history = .3 actual + .7 expected from allowed EV x LA, Statcast 2015+, `data/season/pcon.parquet`).
  Weights 3/2/1; each component regressed by its own empirical-Bayes constant fit on seasons < N (K ~68 BF, uBB ~190,
  HBP ~700, contact ~260-430; superseded by S7-03). Park run factor allocated to contact; Marcel age factor. `model/pcomp.py` `project`, `ARM_SHIP`.

- S7-03 (user OK 2026-11-24): S6-01 regression constants now from same-pitcher year-to-year covariance (`EB_SHIP = "cov"`,
  `cov_k` in `model/pcomp.py`): K ~160 BF, uBB ~340, HBP ~900, contact ~400-490.

- S8-03 (user OK 2026-11-24): hitters with weighted history PA < 500 (5/4/3, N-1 = 1) regress toward the mean of hitters
  with similar history PA (bands rookie/<100/100-249/250-499) instead of league average: no-age projection += 1200/(raw PA + 1200)
  x mu(band) (rookies: + mu in full). Then S5-01 age handling, curve refit on the new residuals. `model/hprior2.py` `ship_prior`,
  `CUT_SHIP`; `data/season/hit_pa_prior.parquet`, `age_curve_hit.parquet` (old: `age_curve_hit_s5.parquet`). In-sample .03264.
  Refit both tables at the pre-season final; re-check on 2025-26 holdout (cutoff was post-hoc).

- S9-01 hitters (user OK 2026-11-24): PA projection = boosted trees (`model/ptime.py` `HFEAT`, fixed params) on PA1-3, starts
  share, late-season PA share, catcher/DH/PH share, teams, age, Marcel rel-wOBA, weighted PA; for everyone with PA in N-1/N-2.
  `python model/ptime.py --ship` -> `data/season/hit_pt_2027.parquet`. Seasons without Retrosheet daily lines use Statcast
  features (`model/ptime_statcast.py`, validated on 2025). Prior only: roster/depth-chart edits in the app override it.

- S9-02 (user OK 2026-11-24): supersedes S9-01. PA/BF = boosted trees on S9-01 features + opening-day roster status
  (active, 40-man, optioned, IL, 60-day IL) + IL share of season N-1/N-2 + IL stints N-1..N-3 (`RFEAT`). `model/ptime2.py`
  `ship_pt`; `python model/ptime2.py --ship N` -> `data/season/pt_<N>.parquet`; refuses to run without `rosters_<N>`.
  Prior only: depth-chart edits in the app override it. Refit with 2025-26 targets at the pre-season final.

- S9-03 (user OK 2026-11-24): before opening day, PA/BF = boosted trees on S9-01 features + `WFEAT` (Dec-1 40-man, winter
  FA / outright-release-DFA not on a 40-man, IL at N-1 season end (+60-day), IL history). `python model/ptime3.py --ship N`
  -> `data/season/pt_winter_<N>.parquet`; needs `rosters_winter_<N>` (backfill job `winter`, uses min(Dec 1, today)).
  2027 built as of 2026-11-24: 794 hitters 172k PA, 1023 pitchers 165k BF. Re-pull the 40-man and rebuild when rosters move.
  Switch to S9-02 (`ptime2.py --ship`) once opening-day rosters are pulled.

## UI direction (user, 2026-11-23)
- When we get to the app: lean into 1980s baseball card aesthetics, for fun (think '87 Topps wood-grain border, '86 Topps team-name
  bar, '89 Upper Deck/Donruss vibes; card-back stat lines and cartoon trivia). Original design only, no real card-maker logos/wordmarks.

## Backlog (next)
- Team wins gap to Vegas ~0.36 wins with C1. Remaining gap is rebuilds (Vegas bottom quintile: C1 -2.5 wins, Vegas -0.6):
  deadline selling / in-season PT loss on bad teams, worse-than-filler fill-ins. DH and team defense ruled out (S12 diag).
  Offense side still barely beats last-year RS.
- `team2.py --ship N` uses C1 when opening-day rosters exist (column `model`), V3 on winter inputs. Holdout gate at pre-season final.
- Spring roster updates: S9-03 shows the playing-time gain is March info (active roster, spring IL). Weekly 40-man + IL
  pulls from mid-Feb, and a test of rosters as of ~Mar 1 / Mar 15, would bridge W1 -> C2.
- Starter bias (+32 BF for 600+ starters) to be handled by team-level normalization of BF in the win-totals step (S10-01 showed
  it's not fixable at the player level). Six-man / team rotation usage untested.
- Coors hitter handling (home/road split, or hitter-specific factor). Park factors as of 2027 still need 2026 park-level rates (Statcast) folded in.
- Hitter K%/BB% stabilization.
- Hitter aging leftovers: 31+ still run about -.004 high after S5-01; pitcher age curve untested.
- Hitter calibration slope now .948 (slightly over-spread). Young-age curve rose (+.006 -> +.011 at 21) to offset the prior on young part-timers.
- Stuff may still help small samples (call-ups, in-season updating); revisit with the in-season model.
- Historical win totals / closing lines source (yardstick).
- Lockout watch: CBA expires 2026-12-01; transactions freeze. Roster-dependent work waits.

## Rejected ideas
- S11-01 U1 team wins from player sums (`model/team.py`), 2026-11-24: 10.55 vs B2 10.86 (t 0.97). Harness kept.
- S10-01 starter workload features + recency weights (`model/ptime4.py`), 2026-11-24: C2 130.9 vs 130.1 (t -1.12), W1 149.4 vs 150.1 (t 0.48).
  Cuts starter bias (+32 -> +9) but starter RMSE unchanged (224): it's injury variance.
- S8-02 history-PA band prior for hitters (`model/hprior.py`), 2026-11-24: .03276, t 2.07 (below bar); one-mu control .03304.
- S8-01 hitter components with covariance regression constants (`model/hcomp.py`), 2026-11-24: .03289 (t -0.75); per-BIP variant .03318.
- S7-02 pitcher plate discipline / stuff (LightGBM whiff) blended into K/uBB history, 2026-11-24: 1.2429-1.2460, all fail.
- S7-01 pitcher age curve from S6-01 residuals, 2026-11-24: 1.2437 (A3) / 1.2428 (half) vs shipped 1.2424; Marcel factor stays.
- S5-02 delta-method age curve with job-retention (IPW) and regression fixes, 2026-11-23: t 1.55 max, worse than A1/A3.
- S4-02 spray zones (pull/center/oppo x EV x LA) as built, 2026-11-23: worse than shipped S3 (t -0.53).
