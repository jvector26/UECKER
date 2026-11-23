# UECKER: starting handoff (written 2026-11-23 from the NBA JIMBO project)
UECKER = Unified Expected Contribution & Key Efficiency Ratings: the MLB sibling of NBA JIMBO.
Goal: a ground-up MLB model for 2027. Player projections (hitters and pitchers), playing time, all 30 teams' win totals and
playoff odds, and daily game odds, updated every day in season. Same spirit as NBA JIMBO: first principles, tested against
history, Vegas shown beside it as a yardstick only.

Upload this file to the new project and tell Claude: "start UECKER, read the handoff."

---

## 1. Carry these over from NBA JIMBO (lessons that cost real sessions)
1. **Repo from day one.** The NBA project lost a session's research log when a zip went missing, and the code split into two
   versions that had to be reconciled. Create `jvector26/UECKER` in session 1. Keep everything there: code, small data,
   the handoff, the research log. Claude attaches it with add_repo (push access) at the start of every chat.
2. **One source of truth.** `work/HANDOFF.md` in the repo holds the current state, the production chain, the research bar,
   the rejected-ideas list and the backlog. `work/RESEARCH_LOG.md` holds every test, with numbers. Project docs are only dated notes.
   Update both and push at the end of every session.
3. **Template first.** Change the page template, then build. Never patch only the built page.
4. **The nightly runs on GitHub Actions, but on-time runs come from a Claude scheduled task** that dispatches the workflow
   through the REST API. GitHub's own cron ran 3-7 hours late. Use REST because GraphQL is blocked, so `gh workflow run` fails.
   MLB needs more run slots than the NBA did (see section 3, daily game odds).
5. **Network split.** The Claude sandbox reaches GitHub and the package registries only. GitHub Actions reaches everything.
   So the live data fetch happens in Actions and gets committed as parquet/JSON. Historical bulk data comes from GitHub-hosted
   sources (Retrosheet, Chadwick) or from a one-time Actions job that writes to the repo.
6. **Decide the research bar before any results come in** (section 5). Write down the expected sign before the test runs,
   as PREREG.md did for the session-50 tests.
7. **Playing time beats rating tweaks.** In the NBA, known minutes vs projected minutes moved game error about 50 times more
   than any rating change. Rating tests were chasing 0.003-0.03. For MLB the equivalents are the starting pitcher, the posted
   lineup, bullpen availability and the IL. Build those early, not as late patches.
8. **Don't let the backtest become a training set.** Hold out the most recent seasons from tuning until a final check.
   Prefer partial weight in proportion to evidence over a binary ship/reject. Put the original inputs through the bar too,
   not only new ideas.
9. **Endogeneity trap.** The NBA test with actual full-game minutes was biased: fringe players got more minutes in blowouts.
   The MLB versions: relievers used depend on the score, pinch hitters depend on the matchup, and starters get pulled early
   when they're getting hit. Use pregame information (lineup card, probable starter) in game-level tests.
10. **Vegas = yardstick, set from the start.** Headline forecast = UECKER alone. Show win totals and game lines beside it.
    Log UECKER vs the line every day from opening day so the record is clean from game 1, with a model version tag on every game.

### User working style (copy into the new HANDOFF)
- Knows the sport well. Test his intuitions against data and report honestly, including when he's right.
- Never change the model without his OK. Report a failed test plainly.
- Can't run code, so Claude runs everything. Usually on his phone: short answers, lead with the result.
- Doesn't want public models (for MLB: ZiPS / Steamer / THE BAT / PECOTA / FanGraphs WAR) used as training targets.
  They're fine as a yardstick if he agrees. Doesn't bet.
- Edits roles in the app rather than being quizzed about them (MLB: lineup spot, rotation slot, closer).

---

## 2. Calendar reality (as of 2026-11-23)
- The CBA expires **Dec 1, 2026**. A lockout is widely expected, and a lockout freezes transactions. Rosters may be unsettled
  until late, which is what happened in 2022, when the deal came March 10 after a 99-day lockout. Opening Day is normally late March 2027.
- Use that to your advantage: the winter is for historical data, the player model and the research bar, all of which are
  roster-independent. Put roster-dependent work (depth charts, playing time, team sims) on a pipeline that can rebuild
  in minutes once transactions resume.
- Rule environment: check what's in force for 2027 (the ABS challenge system, pitch clock and so on) before trusting older
  seasons' catcher framing and pace numbers. A new CBA could also change roster rules (26-man, options, playoff format).

| When | Phase |
|---|---|
| Late Nov - Dec | Repo, data pipeline, historical database 2015-2026, Marcel baseline |
| Jan | Hitter + pitcher rating model, aging, regression; research bar running |
| Feb | Playing time model, team run environment, season sim, backtest vs Vegas win totals |
| Mar (spring training) | Depth charts, nightly pipeline dry runs on spring games, app shell |
| Opening Day | Live: daily ratings update, game odds, vs Vegas log |

---

## 3. Model architecture (MLB translation of NBA JIMBO)
**Scale:** runs per plate appearance above average (show as runs per 600 PA for hitters and runs per 200 IP or per batter
faced for pitchers). Baseline = AVERAGE player, as in the NBA app, unless he decides otherwise. Wins via runs-per-win
(~10, recompute each year, like PPW 36.7 in the NBA).

**Hitters (the NBA "box part" equivalent):** components with different stabilization rates, each regressed separately:
- Plate discipline: K%, BB% (stabilize fast)
- Contact quality: EV, launch angle, barrel rate, xwOBA on contact (Statcast, 2015+)
- Batted-ball luck: BABIP, which is mostly regressed. Speed and spray carry part of it.
- Power: ISO / HR per fly ball, park-adjusted
- Baserunning (sprint speed, stolen-base runs) and defense (Outs Above Average / fielding run value, catcher framing,
  which the ABS challenge system may shrink) and positional adjustment
- Platoon split (regressed hard toward the league split)

**Pitchers:** K-BB% (the backbone), contact management (barrels/EV allowed, mostly regressed), GB rate, pitch-quality
("stuff") from pitch velocity/movement/extension. Starter vs reliever role effects, times through the order, platoon.

**The "lineup residual" equivalent:** in the NBA it was ridge RAPM toward the box prior. MLB doesn't need it as much,
because baseball is close to individual events. The natural analogue is a matchup/event model: each PA's outcome
(K, BB, HBP, 1B, 2B, 3B, HR, out types) predicted from batter, pitcher, park, platoon and fielders (log5 / multinomial).

**Aging + projection:** component-specific age curves, weighted multi-year window (Marcel uses 5/4/3), regression
toward the mean by PA. Rookies/prospects: minor-league stats translated with level factors (MLE), with a
prospect-rank or draft prior as the fallback. That's MLB's version of the college/rookie prior.

**Playing time (the NBA "minutes"):** depth chart + IL + platoon roles -> projected PA per hitter, and rotation slot +
IP per start + bullpen role -> projected batters faced per pitcher. The team total is fixed (~6,200 PA, ~1,450 IP), the
same way NBA minutes summed to 240 per game. Injury rates by age/position for the starters' expected missed time.

**Team + season:** runs scored = lineup by batting order x PA rates x park; runs allowed = rotation + bullpen + defense.
Game win probability from expected runs (Pythag / negative-binomial or PA simulation). Season sim with the real 2027
schedule, then playoffs (12-team format, byes, best-of-3 wild card).

**Daily game odds:** the starting pitcher is the biggest single input. Then the posted lineup, bullpen fatigue (pitches
over the last 3 days), park, weather/roof, rest/travel. Lineups post 1-4 h before first pitch, so in-season runs need
to land after lineups. There are ~15 games a day with first pitches from noon to 10 pm ET.

---

## 4. Data sources (check availability in session 1; things change)
| Need | Source | Notes |
|---|---|---|
| Schedule, box scores, probable pitchers, lineups, transactions, IL | MLB Stats API (statsapi.mlb.com) | Free JSON. The backbone of the nightly, like nba_fetch. |
| Pitch-level Statcast 2015+ | Baseball Savant via `pybaseball` | ~700k pitches/season. Fetch in Actions, store per-season parquet. |
| Historical play-by-play 1990s+ | Retrosheet (chadwickbureau GitHub mirrors) | Reachable from the sandbox. Long backtest history. |
| Player ID crosswalk | Chadwick Bureau register (GitHub) | MLBAM / Retrosheet / FanGraphs / BBRef IDs. |
| Season totals, history | Lahman database | Quick sanity checks. |
| Minor-league stats | MLB Stats API (sportId for AAA/AA/...) | For the prospect prior. |
| Park factors | Build your own from Retrosheet/Statcast | Multi-year, handedness-split. |
| Win totals, game lines | Need a source | Historical closing lines were the NBA's best yardstick. Find an MLB equivalent early (public GitHub datasets, as with the NBA lines). |
| Depth charts / roles | MLB.com depth charts, Stats API roster status | FanGraphs Roster Resource is great but scraping it is fragile. |

---

## 5. Research bar (proposed; agree with the user in session 1)
1. **Baseline to beat: Marcel** (simple, public method, not a target). If a component doesn't beat Marcel out of sample,
   it doesn't ship.
2. **Player-level test:** project season N from seasons < N, score next-season wOBA (hitters) and RA9/FIP-type runs
   (pitchers) by PA-weighted RMSE, leave-one-season-out, 2016-2026 (drop 2020 or down-weight it: 60 games).
3. **Game-level test:** predict each game from pregame info only (probable starters, posted lineups), score log loss
   or run-margin RMSE. Pass = better LOSO, season-clustered t > 2.5, same sign in both halves (2016-20 / 2021-26).
4. **Team test:** preseason projected wins vs actual, 2016-2026, compared with Vegas win totals.
5. Hold out 2025-2026 from all tuning until a pre-season final check.
6. Expect less from team forecasts than in the NBA: MLB has more luck per game, so win-total RMSE of ~8-9 wins is normal and
   even Vegas misses by about that much. Game-level edges are small (favorites rarely above 65%). Calibration matters more than accuracy headlines.

---

## 6. Decisions for the user in session 1
1. Headline = pure UECKER with Vegas as yardstick (recommended, same as NBA since 11-09)?
2. Baseline for player value: average (as in the NBA app) or replacement level (WAR-style)?
3. Scope for opening day: season projections + win totals only, or also daily game odds (much more pipeline work)?
4. App: a separate PWA (copy the NBA site/ shell and service-worker pattern) or a sport switch inside the existing app?
5. History depth: Statcast era only (2015+) or Retrosheet back to ~2000 for the long-run tests?

---

## 7. Paste into the new project's instructions
> You and I are building UECKER, a ground-up statistical model for the 2027 MLB season: player projections for
> hitters and pitchers, playing time, win totals and playoff odds for all 30 teams, and daily game odds, updated every
> day in season. Everything lives in the GitHub repo jvector26/UECKER; at the start of each chat attach it,
> run work/setup.sh, and read work/HANDOFF.md. Every model change goes through the research bar and needs my OK.
> Vegas is a yardstick, not an input. I'm usually on my phone: lead with the result, keep it short.

**First session checklist for Claude:** create the repo + skeleton (pipeline/, model/, work/, site/), write HANDOFF.md
from this file, check each data source in section 4 (sandbox vs Actions), pull Chadwick + Retrosheet, run a first
Statcast fetch job in Actions, build Marcel for 2016-2026 and score it. That gives the bar its first number.
