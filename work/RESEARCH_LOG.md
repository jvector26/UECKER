# UECKER research log
Every test, with numbers. Newest at the bottom. Pre-registration (expected sign/size) is written BEFORE the run.

Conventions (session 1):
- Scoring targets are relative to that season's league, so league-wide environment shifts (2019 ball, 2022 dead ball)
  don't count against player skill: hitters = wOBA - lg wOBA; pitchers = RA9 - lg RA9.
- Projections use only data from seasons < N (including the weights/league rates of those seasons).
- wOBA weights and run values are UECKER's own (RE24 linear weights from Retrosheet, `model/run_values.py`).
- Tuning window: targets 2016-2024. Holdout: 2025-2026, not scored until the pre-season final check.
- 2020 (60 games) as a target: reported, but headline = 2016-2024 excluding 2020.

---

## S1-01 Run values 2010-2025 (sanity check, not a model test)
RE24 linear weights per season from Retrosheet events. League wOBA scaled to league OBP.
Result: weights in the familiar range, e.g. 2019 HR 1.94 / 2B 1.21 / 1B ~0.88, wOBA scale 1.18, lg wOBA .322;
2014 (low offense) HR 2.15, scale 1.32. Stored in data/season/run_values.parquet.

## S1-02 Marcel baseline (PRE-REGISTERED before running)
Implementation (Tango's Marcel, as public method, not a target):
- Hitters: component rates per PA (uBB, HBP, 1B, 2B, 3B, HR), weights 5/4/3 for N-1/N-2/N-3, regress with 1200 PA
  of league average (weighted units), age factor +0.006/yr under 29, -0.003/yr over 29 on positive events.
  Projected wOBA uses season N-1's weights; minus the projected league wOBA built the same way.
- Pitchers: runs per out, weights 3/2/1, regress with 134 IP (402 outs) of league average, same age factor (inverted).
  Plus a component version (K, uBB, HBP, HR per BF regressed, converted to runs with own linear weights).
- Players with no MLB history get league average (Marcel default).
Test: targets 2016-2024 ex-2020, hitters PA>=100 weighted by PA; pitchers BF>=100 weighted by BF.
Comparison baselines: league average for everyone (0), and last season only (unregressed).
Expected: Marcel beats both. Hitter rel-wOBA weighted RMSE around .030-.035; pitcher rel-RA9 RMSE around 1.2-1.6.
Constants are Marcel's published ones; I also report 0.5x/2x regression as a sensitivity check only (no tuning).

### S1-02 result (targets 2016-2024 ex-2020; PA- or BF-weighted RMSE)
Hitters, wOBA - lg wOBA (n ~450/season):

| Season | League avg | Last season only | Marcel |
|---|---|---|---|
| 2016 | .0380 | .0564 | .0327 |
| 2017 | .0393 | .0571 | .0346 |
| 2018 | .0397 | .0563 | .0329 |
| 2019 | .0412 | .0479 | .0354 |
| 2021 | .0401 | .0650 | .0347 |
| 2022 | .0393 | .0492 | .0342 |
| 2023 | .0378 | .0499 | .0342 |
| 2024 | .0387 | .0472 | .0342 |
| **Pooled** | **.0393** | **.0539** | **.0341** |
(2020 as target: .0461 / .0515 / .0442.) Marcel beats league average in every season. AS EXPECTED.
Sensitivity (not tuning): 600 PA regression .0350, 2400 PA .0337.

Pitchers, RA9 - lg RA9 (n ~520/season, BF>=100):
pooled: league avg 1.309, last season only 2.534, Marcel RA 1.325, Marcel component (K/BB/HBP/HR) 1.323.
**NOT AS EXPECTED: Marcel does not beat "everyone is league average" on next-season RA9.** Checked for a bug: data
reproduces known stability (pitcher K% year-to-year r .74, BB% .53, HR% .31, RA9 .26 at 200+ BF). Marcel projections
correlate only .18 with next RA9 and the fitted slope is .44, i.e. it under-regresses for this pool (reliever-heavy).
With 2x regression (804 weighted outs) Marcel RA is 1.288, which does beat league average (1.309).
On a component-runs target (K/BB/HBP/HR only, no BIP/sequencing luck): league avg .926, Marcel comp .882 (402) / .882 (804).
Takeaways: (1) single-season RA9 is mostly noise; differences between models will be small and need the season-clustered
t-test. (2) The regression constant for pitchers matters; my reading of Marcel's 134 IP in weighted units may be
too light. Open decision for the user: which pitcher baseline is the bar (see HANDOFF decisions).

## Session 2 (2026-11-23)

### S2-00 data: 2026 season added
- `model/run_values_statcast.py`: RE24/wOBA weights from Statcast PA rows. Check on 2025 vs Retrosheet: lg wOBA identical
  (.3153), weights within .01 (uBB .725/.717, 1B .884/.884, 2B 1.230/1.225, HR 1.976/1.998), 3B .05 (small n). 2026 row written (source=statcast).
- `model/seasons.py`: 2026 player-seasons from Stats API (Retrosheet 2026 not posted). Check on 2025: every hitter PA, pitcher BF,
  R, outs, HBP match Retrosheet exactly. Every 2010-2026 player maps Retrosheet <-> MLBAM (Chadwick); `mlbam` column added.

### S2-01 park factors (PRE-REGISTERED before running)
Method (`model/parks.py`): component park factors per PA (uBB, HBP, 1B, 2B, 3B, HR, K) by batter hand, plus runs per PA.
"As of season N" factors use seasons N-3..N-1 only: rate in park (both teams) / rate in the home team's road games (both teams).
Shrunk toward 1 by empirical Bayes on the log ratio (true spread = observed variance minus sampling variance); no tuned constants.
Players' seasons get a PA-weighted multiplier over the parks they actually played in.
Test: Marcel on park-neutral history, then re-apply each player's target-season park mix (schedule info; uses actual games, so
mid-season trades leak slightly, noted). Same targets/metrics as S1-02. Pitchers: runs per out neutralized by the run factor.
Expected: small gain. Hitters .0341 -> about .0333-.0338, concentrated in Rockies and team-changers. Pitchers Marcel RA9
improves by ~.005-.015 (1.325 and 1.288 versions). If hitters don't improve at all, park factors still go into game odds
(they're needed there), but don't get used in player projections.

### S2-01 result (targets 2016-2024 ex-2020)
Park factors sane: as of 2026, runs PF Coors 1.16 (top), Seattle 0.90 (bottom); HR high Cincinnati 1.19, low Chase/Kauffman/Oakland.
EB shrink spread (tau of log PF): HR .11, 3B .27-.33, 2B .08-.09, 1B .04-.05, K .05-.07, uBB .01-.03, runs .05.

| | base Marcel | park-adjusted | season-clustered t |
|---|---|---|---|
| Hitters rel-wOBA | .0341 | .0340 | 1.05 |
| Pitchers RA9, Marcel 402 | 1.3251 | 1.3152 | 4.61 |
| Pitchers RA9, Marcel 804 | 1.2884 | 1.2786 | 3.67 |
Same sign both eras (pitchers 804: 2016-19 1.292->1.286, 2021-24 1.284->1.271). 2020: hitters .0442->.0444, pitchers 1.677->1.671.
- **Pitchers: AS EXPECTED, passes** (about -.010, at the top of the expected range).
- **Hitters: NOT AS EXPECTED, no real gain.** Big-park and multi-team subgroups flat. Rockies-heavy hitters (n=111) get WORSE:
  .0332 -> .0343, mean error -.0007 -> -.0067 (park version over-projects them). Consistent with the home/road method
  overstating Coors for hitters (Rockies' own road numbers are depressed, the "Coors hangover"), and with raw history already
  carrying the same park when the player stays put. Per the pre-registration: park factors go into game odds; hitter projections
  stay unadjusted for now. Shipping the pitcher adjustment needs the user's OK.
- User OK 2026-11-23: research bar approved; S2-01 pitcher park adjustment shipped.

## Session 3 (2026-11-23, evening)

### S3-00 data: 2026 park mix
- `model/park_mix_statcast.py`: Stats API venue -> Retrosheet park map learned on 2025 (2425/2434 games matched on date,
  score, home SP). Check: 2025 mix rebuilt from Statcast vs Retrosheet: 365,852 PA rows both, abs diff 4 by player/park/hand.
  2026 rows appended. Tropicana (Rays back home) -> STP01 by hand. Neutral sites (Mexico City 157 PA, Field of Dreams 79,
  Las Vegas Ballpark 514) get factor 1.0.

### S3-01 hitter contact quality (PRE-REGISTERED before running)
Idea: on balls in play, exit velocity x launch angle says more about future production than whether the ball found a hole.
Method (`model/xcon.py`): UECKER's own expected contact value. For each season, bin batted balls (AB-ending contact + SF,
no sac bunts) by EV (3 mph) x LA (4 deg); bin value = mean of UECKER wOBA weights for that season's outcomes (out/ROE = 0),
lightly smoothed toward neighbors; no MLB xwOBA column used. Per hitter-season: delta = (expected - actual contact value)
/ woba_den. Balls with no EV/LA keep their actual value. Pre-2015 seasons: delta = 0.
Projection: Marcel on adjusted history, i.e. contact value = a*actual + (1-a)*expected. a chosen leave-one-target-season-out
from {0, .1, ..., 1}; full curve reported. Known gaps: no sprint speed (fast hitters' grounders), no spray angle.
Expected: a around .3-.6. Hitters rel-wOBA .0341 -> about .0331-.0337, t > 2.5, same sign 2016-19 and 2021-24. Bigger gain
on 2018-24 targets (full 3-year Statcast history). If it doesn't beat Marcel, contact quality waits for a fuller model
(speed + spray) rather than shipping.

### S3-01 result (targets 2016-2024 ex-2020)
EV/LA present on 97-99.7% of contact; league actual/expected contact value 1.001-1.003 every year (calibrated).
Hitter-season delta sd .020 wOBA (PA>=300). Y/y per-BIP: expected value predicts next year's actual slightly better than actual does (.622 vs .610).
Curve (a = weight on actual): 0 .03349, .2 .03330, .3 .03327, .4 .03327, .6 .03340, 1 (Marcel) .03413. LOSO picks a = .3 (7 seasons), .4 (2).

| subset | Marcel | Marcel+xcon | season-clustered t |
|---|---|---|---|
| 2016-24 ex-2020 | .0341 | .0333 | 4.70 |
| 2016-19 | .0339 | .0335 | 3.24 |
| 2021-24 | .0343 | .0331 | 6.64 |
| 2018-24 ex-2020 (full Statcast history) | .0343 | .0334 | 3.79 |
Every target season improves (2018 smallest, .0329 -> .0328). 2020: .0442 -> .0425.
- **AS EXPECTED, passes.** Size at the good end of the pre-registered range; a at the low end (.3). Needs user OK to ship.
- Next refinements (separate pre-registrations): sprint speed on grounders, spray angle, K%/BB% stabilization.
- User OK 2026-11-23: S3-01 shipped with a = .3 (`A_SHIP` in model/xcon.py).

## Session 4 (2026-11-23, late)
Baseline for both tests = the shipped model: Marcel + S3-01 contact quality (a = .3), hitters rel-wOBA **.0333**
(targets 2016-24 ex-2020). Pass = better pooled, season-clustered t > 2.5 vs .0333, same sign 2016-19 and 2021-24.
Tuning parameters chosen leave-one-target-season-out, full grids reported.

### S4-01 hitter speed on grounders (PRE-REGISTERED before running)
Idea: fast hitters beat out grounders that EV x LA calls outs, so S3-01 under-rates them and over-trusts the expected side.
Real sprint speed lives on Savant (Actions-only); first test with in-house proxies, fetch sprint speed only if these show signal.
- A, split trust: separate weight on actual for grounders (LA < 10) vs air balls (a_gb, a_air on 0..1 by .1).
  Expected: a_gb > a_air (about .5-.7 vs .2-.3). Gain .0333 -> .0331-.0333; may not reach t 2.5 on its own.
- B, speed-aware expected value: speed score per hitter-season = mean of within-season z-scores of
  (SB+CS)/(1B+uBB+HBP) and 3B/(2B+3B), shrunk by opportunities. Grounder expected value += beta x speed, beta fit per season
  on that season's grounders (history seasons only). Expected: beta > 0 every season; gain .0001-.0003.
- If A or B passes, its size sets whether pulling real sprint speed through Actions is worth it.

### S4-02 spray angle (PRE-REGISTERED before running)
Idea: at the same EV x LA, pulled air balls are worth more (shorter fences, HR) and pulled grounders less (shifts pre-2023);
a hitter who pulls his fly balls is better than EV x LA says.
Method: spray angle from hc_x/hc_y, sign flipped for LHB so negative = pull; 3 zones (pull < -15 deg, center, oppo > 15).
Expected value bins = EV x LA x zone (same 3x3 smoothing within zone). Balls missing spray keep EV x LA value.
Weight on actual re-tuned (expected to stay near .3). Expected: gain .0333 -> .0329-.0332, both eras.
Combined A/B + spray reported too.

### S4 results (targets 2016-2024 ex-2020; `model/xcon2.py`, output `work/results/s4_xcon2.txt`)
Shipped S3 reproduces .03327. Speed score y/y corr .64. Grounder speed beta > 0 every season (.014-.022).
| subset | Marcel | S3 shipped | A split a | B speed | S spray | SB |
|---|---|---|---|---|---|---|
| 2016-24 ex-2020 | .0341 | .0333 | .0332 | .0332 | .0333 | .0333 |
| t vs S3 | | | 1.27 | **3.11** | -0.53 | -0.24 |
| 2016-19 (t) | .0339 | .0335 | .0335 (0.22) | .0334 (0.92) | .0334 (0.32) | .0334 (0.43) |
| 2021-24 (t) | .0343 | .0331 | .0330 (1.81) | .0329 (7.35) | .0332 (-0.72) | .0331 (-0.49) |
Raw RMSE: S3 .03327, A .03317 (a_gb .6, a_air .2), B .03312 (a_gb .4-.5, a_air .1-.2). B better in 7 of 8 seasons (2017 worse by .0001).
- **A: direction as expected (a_gb .6 > a_air .2), size at the top of the range, fails t (1.27).** Not shipped on its own.
- **B: passes the bar as written** (t 3.11, positive both eras, size in the pre-registered range). But the mechanism only half
  matches: the gain is all slow hitters (mean error -.0122 -> -.0089); fast hitters go from -.0009 to -.0043 (RMSE .0311 -> .0315).
  Exploratory age x speed split (not pre-registered): slow hitters are over-projected in every age band, so it is not just aging;
  the linear speed term over-corrects the fast end. 2016-19 support is weak (t 0.92).
- **S spray: NOT AS EXPECTED, fails** (worse pooled; 2024 .0324 -> .0329). Likely sparse 3-zone bins and the 2023 shift ban
  changing what a pulled grounder is worth. Rejected as built.
- Side finding (exploratory): projections run high overall, mean error S3 -.0045, Marcel -.0059, worst at 32+ (-.0085 mid-speed).
  Suggests the Marcel age curve is too kind to older hitters; candidate for its own pre-registered test.
- Recommendation: hold B pending the user's call. Options: ship B now at partial weight, or first pull real sprint speed (Savant,
  via Actions) to test the mechanism with a non-linear speed term.
