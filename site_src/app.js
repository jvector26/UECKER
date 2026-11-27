// UECKER app: JIMBO-style layout. Standings, Tonight (placeholder), Players, vs Vegas, Settings; team and player views.
// Reads data/teams.json, players.json, meta.json (pipeline/export_app.py).
(() => {
'use strict';
// team colour (chip / stripe only; no logos)
const TC = {ARI:'#A71930',ATL:'#13274F',BAL:'#DF4601',BOS:'#BD3039',CHC:'#0E3386',CWS:'#27251F',CIN:'#C6011F',CLE:'#00385D',
  COL:'#33006F',DET:'#0C2340',HOU:'#EB6E1F',KC:'#004687',LAA:'#BA0021',LAD:'#005A9C',MIA:'#00A3E0',MIL:'#FFC52F',MIN:'#002B5C',
  NYM:'#FF5910',NYY:'#0C2340',ATH:'#003831',PHI:'#E81828',PIT:'#FDB827',SD:'#2F241D',SF:'#FD5A1E',SEA:'#005C5C',STL:'#C41E3A',
  TB:'#8FBCE6',TEX:'#003278',TOR:'#134A8E',WSH:'#AB0003',FA:'#9A9A9A'};
const DIVS = ['AL East','AL Central','AL West','NL East','NL Central','NL West'];
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const MN = '−';
const sgn = (x, d=1) => x == null ? '—' : (x > 0 ? '+' : x < 0 ? MN : '') + Math.abs(x).toFixed(d);
const pm = (x, d=1) => `<span class="${x > 0.05 ? 'pos' : x < -0.05 ? 'neg' : ''}">${sgn(x, d)}</span>`;
const f3 = x => x == null ? '—' : x.toFixed(3).replace(/^0/, '');
const s3 = x => x == null ? '—' : (x >= 0 ? '+' : MN) + Math.abs(x).toFixed(3).replace(/^0/, '');
const norm = s => s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
let T = [], P = [], M = {}, teamById = {}, byTeam = {}, prevHash = '';
const view = $('#view');

const abOf = p => (teamById[p.t] || {}).ab || 'FA';
const role = p => {
  if (p.k === 'hit') return p.pos === 'TWP' ? 'DH' : (p.pos || '—');
  const last = p.h[p.h.length - 1];
  return last && last[8] / Math.max(1, last[7]) >= 0.5 ? 'SP' : 'RP';
};
const unit = p => p.k === 'hit' ? 'PA' : 'BF';

// ---------- standings
const st = {group: 'div', sort: 'w', desc: true};
const SORTS = {w: ['Wins', t => t.w], rd: ['Run diff', t => t.rd], rs: ['Runs scored', t => t.rs],
               ra: ['Runs allowed', t => -t.ra], py: ["'26 Pythag", t => t.py]};
const LO = 60, HI = 105, xp = w => Math.max(0, Math.min(100, (w - LO) / (HI - LO) * 100));
function chart(t) {
  const gl = [70, 81, 90, 100].map(w => `<i class="gl${w === 81 ? ' mid' : ''}" style="left:${xp(w)}%"></i>`).join('');
  return `<div class="chart" aria-hidden="true">${gl}<i class="py" style="left:${xp(t.py)}%"></i><i class="dot" style="left:${xp(t.w)}%"></i></div>`;
}
function standings() {
  const key = SORTS[st.sort][1];
  const order = [...T].sort((a, b) => st.desc ? key(b) - key(a) : key(a) - key(b));
  const groups = st.group === 'div' ? DIVS.map(d => [d, order.filter(t => t.div === d)])
    : st.group === 'lg' ? [['American League', order.filter(t => t.div.startsWith('AL'))], ['National League', order.filter(t => t.div.startsWith('NL'))]]
    : [['All teams', order]];
  const head = `<thead><tr><th class="l">#</th><th class="l">Team</th>
    <th class="l ch"><div class="axis"><span>${LO}</span><span>81</span><span>${HI}</span></div></th>
    <th>Wins</th><th>Diff</th><th class="hide-s">RS</th><th class="hide-s">RA</th><th class="hide-s">'26 Py</th></tr></thead>`;
  const row = (t, i) => `<tr><td class="l rk">${i + 1}</td>
    <td class="l tm"><a href="#/team/${t.id}"><i class="chip" style="--tc:${TC[t.ab]}"></i>${t.ab}</a></td>
    <td class="ch">${chart(t)}</td><td class="w">${t.w.toFixed(1)}</td><td>${pm(t.rd, 0)}</td>
    <td class="hide-s">${t.rs}</td><td class="hide-s">${t.ra}</td><td class="hide-s dim">${t.py.toFixed(1)}</td></tr>`;
  view.innerHTML = `
    <div class="legend"><span><i class="lg-dot"></i>UECKER wins</span><span><i class="lg-tick"></i>'26 Pythag wins</span></div>
    <div class="tools">
      <select id="stg" aria-label="Group teams"><option value="div">Divisions</option><option value="lg">Leagues</option><option value="all">All teams</option></select>
      <select id="sts" aria-label="Sort teams by">${Object.entries(SORTS).map(([k, v]) => `<option value="${k}">${v[0]}</option>`).join('')}</select>
      <button class="btn" id="std" type="button" aria-label="Reverse sort order">${st.desc ? '↓' : '↑'}</button>
    </div>
    ${groups.map(([g, ts]) => `<h2 class="sec">${g}</h2><div class="tbl"><table>${head}<tbody>${ts.map(row).join('')}</tbody></table></div>`).join('')}
    <p class="note">Wins come from projected runs scored and allowed, built from each rostered player's projection and playing time and blended with last season's Pythag record. Ranges and playoff odds come with the season simulator. Tap a team for its roster.</p>`;
  $('#stg').value = st.group; $('#sts').value = st.sort;
  $('#stg').onchange = e => { st.group = e.target.value; standings(); };
  $('#sts').onchange = e => { st.sort = e.target.value; st.desc = true; standings(); };
  $('#std').onclick = () => { st.desc = !st.desc; standings(); };
}

// ---------- player table (Players tab and team view)
const PCOL = {runs: p => p.runs, std: p => p.std, pt: p => p.pt};
function ptable(list, opt = {}) {
  const sort = opt.sort || 'runs';
  const rows = list.map((p, i) => `<tr><td class="l rk">${i + 1}</td>
    <td class="l nm"><a href="#/player/${p.id}">${esc(p.n)}</a><span class="sub">${opt.team ? '' : abOf(p) + ' · '}${role(p)}${p.age ? ' · age ' + Math.floor(p.age) : ''}</span></td>
    <td class="nw">${p.pt}<span class="dim"> ${unit(p)}</span></td><td class="hide-s">${pm(p.std)}</td><td class="big">${pm(p.runs)}</td></tr>`).join('');
  const th = (k, label, cls = '', title = '') => `<th class="${cls}"${k === sort ? ' aria-sort="descending"' : ''}>${opt.sortable ? `<button data-sort="${k}" title="${title}">${label}</button>` : label}</th>`;
  return `<div class="tbl"><table><thead><tr><th class="l">#</th><th class="l">Player</th>
    ${th('pt', 'Proj PT', '', 'Projected plate appearances or batters faced')}
    ${th('std', 'Per season', 'hide-s', 'Runs vs average per 600 PA (hitters) or 200 IP (pitchers)')}
    ${th('runs', 'Runs', '', 'Projected runs vs an average player over his projected playing time')}</tr></thead>
    <tbody>${rows || '<tr><td colspan="5" class="l dim">No players match. Try a wider filter.</td></tr>'}</tbody></table></div>`;
}

const pl = {q: '', team: '', pos: '', min: 100, sort: 'runs', n: 50};
function players() {
  const tOpts = [...T].sort((a, b) => a.ab.localeCompare(b.ab)).map(t => `<option value="${t.id}">${t.ab}</option>`).join('');
  view.innerHTML = `<div class="tools">
      <input id="q" type="search" placeholder="Search player" aria-label="Search player" autocomplete="off">
      <select id="tf" aria-label="Team"><option value="">All teams</option>${tOpts}<option value="fa">Unsigned</option></select>
      <select id="pf" aria-label="Position"><option value="">All positions</option><option value="hit">Hitters</option><option value="pit">Pitchers</option>
        <option value="SP">SP</option><option value="RP">RP</option>${['C','1B','2B','3B','SS','LF','CF','RF','DH'].map(x => `<option>${x}</option>`).join('')}</select>
      <select id="mf" aria-label="Minimum playing time"><option value="0">Any playing time</option><option value="100">100+ PA or BF</option><option value="300">300+ PA or BF</option><option value="500">500+ PA or BF</option></select>
      <span class="count" id="count"></span></div><div id="pt"></div><button class="btn more" id="more">Show 50 more</button>
      <p class="note">Runs = projected runs vs an average player over his projected playing time: hitters count batting, baserunning, fielding and position; pitchers count runs saved. Per season = the same per 600 PA or 200 IP. Tap a header to sort.</p>`;
  $('#q').value = pl.q; $('#tf').value = pl.team; $('#pf').value = pl.pos; $('#mf').value = pl.min;
  const draw = () => {
    const k = norm(pl.q.trim());
    const L = P.filter(p => p.pt >= pl.min
      && (!pl.team || (pl.team === 'fa' ? p.t == null : p.t === +pl.team))
      && (!pl.pos || p.k === pl.pos || role(p) === pl.pos)
      && (!k || p._s.includes(k)));
    L.sort((a, b) => PCOL[pl.sort](b) - PCOL[pl.sort](a));
    $('#count').textContent = `${L.length.toLocaleString()} players`;
    $('#pt').innerHTML = ptable(L.slice(0, pl.n), {sort: pl.sort, sortable: true});
    $('#more').hidden = L.length <= pl.n;
    $('#pt').querySelectorAll('th button').forEach(b => b.onclick = () => { pl.sort = b.dataset.sort; draw(); });
  };
  let qt;
  $('#q').oninput = e => { clearTimeout(qt); qt = setTimeout(() => { pl.q = e.target.value; pl.n = 50; draw(); }, 120); };
  $('#tf').onchange = e => { pl.team = e.target.value; pl.n = 50; draw(); };
  $('#pf').onchange = e => { pl.pos = e.target.value; pl.n = 50; draw(); };
  $('#mf').onchange = e => { pl.min = +e.target.value; pl.n = 50; draw(); };
  $('#more').onclick = () => { pl.n += 50; draw(); };
  draw();
}

// ---------- team view
let tside = 'hit';
function teamView(id) {
  const t = teamById[id];
  if (!t) return notFound();
  const rank = [...T].sort((a, b) => b.w - a.w).indexOf(t) + 1;
  const all = byTeam[id] || [];
  const list = all.filter(p => p.k === tside).sort((a, b) => b.runs - a.runs);
  const nH = all.filter(p => p.k === 'hit').length;
  view.innerHTML = `<div class="pvtop"><a class="back" href="#/standings">‹ Standings</a></div>
    <div class="pvname">${esc(t.name)}</div><p class="pvsub">${t.div}</p><div class="stripe" style="--tc:${TC[t.ab]}"></div>
    <div class="tiles">
      <div class="tile"><div class="k">UECKER wins</div><div class="v">${t.w.toFixed(1)}</div><div class="s">#${rank} of 30</div></div>
      <div class="tile"><div class="k">Run differential</div><div class="v">${pm(t.rd, 0)}</div><div class="s">${t.rs} scored, ${t.ra} allowed</div></div>
      <div class="tile"><div class="k">'26 Pythag wins</div><div class="v">${t.py.toFixed(1)}</div><div class="s">from last season's runs</div></div>
      <div class="tile"><div class="k">Roster alone</div><div class="v">${t.uw.toFixed(1)}</div><div class="s">wins before the blend with '26</div></div>
    </div>
    <div class="seg" role="group" aria-label="Hitters or pitchers"><button aria-pressed="${tside === 'hit'}" data-s="hit">Hitters (${nH})</button><button aria-pressed="${tside === 'pit'}" data-s="pit">Pitchers (${all.length - nH})</button></div>
    ${ptable(list, {team: true})}
    <p class="note">Everyone on the 40-man roster with projected playing time. Depth charts and role editing come in a later version.</p>`;
  view.querySelectorAll('.seg button').forEach(b => b.onclick = () => { tside = b.dataset.s; teamView(id); });
}

// ---------- player view
const BALL = `<svg viewBox="0 0 40 40" aria-hidden="true"><circle cx="20" cy="20" r="17" fill="#FBF6E8" stroke="#17213A" stroke-width="2.5"/>
<path d="M9 9c4 4 4 18 0 22M31 9c-4 4-4 18 0 22" fill="none" stroke="#C8302C" stroke-width="2" stroke-dasharray="2.5 2"/></svg>`;
// run breakdown (S14): components from the export (p.bd = [[key, runs], ...]) and the one-line comment (p.cm)
function breakdown(p) {
  const bd = (p.bd || []).filter(x => x[1] !== null);
  if (!bd.length) return '';
  const lab = M.parts || {}, hit = p.k === 'hit';
  const top = Math.max(10, ...bd.map(x => Math.abs(x[1])));
  const rows = bd.map(([k, v]) => {
    const w = Math.min(50, 50 * Math.abs(v) / top);
    return `<div class="bdr"><span class="bdk">${esc(lab[k] || k)}</span>
      <span class="bdbar" aria-hidden="true"><i class="${v < 0 ? 'n' : 'p'}" style="width:${w.toFixed(1)}%;${v < 0 ? 'right:50%' : 'left:50%'}"></i></span>
      <span class="bdv ${v > 0.05 ? 'pos' : v < -0.05 ? 'neg' : ''}">${pm(v)}</span></div>`;
  }).join('');
  const note = hit
    ? (bd.length === 1 ? 'Batting only for now.'
                       : 'Each piece in runs vs an average player over his projected playing time. Position credits harder spots (catcher, shortstop) and charges easier ones (first base, DH). Team win totals use batting only: fielding and baserunning did not improve them in testing.')
    : 'Each piece in runs vs an average pitcher over his projected batters faced. Strikeouts, walks and hit batters are each counted against an average ball in play, so Contact is quality per ball in play.';
  return `<h2 class="sec">Run breakdown</h2>
    ${p.cm ? `<p class="bdcm">${esc(p.cm)}</p>` : ''}
    <div class="bd">${rows}<div class="bdr tot"><span class="bdk">Total</span><span class="bdbar"></span><span class="bdv">${pm(p.runs)}</span></div></div>
    <p class="note">${note}</p>`;
}
function playerView(id) {
  const p = P.find(x => x.id === id);
  if (!p) return notFound();
  const hit = p.k === 'hit', t = teamById[p.t];
  const peers = P.filter(x => x.k === p.k).sort((a, b) => b.runs - a.runs);
  const bt = p.b && p.th ? `bats ${p.b}, throws ${p.th}` : '';
  const sub = [t ? t.name : 'Unsigned', role(p), p.age ? `age ${p.age.toFixed(1)} in '27` : '', bt].filter(Boolean).join(' · ');
  const rows = p.h.map(h => hit
    ? `<tr><td>${h[0]}</td><td>${h[1]}</td><td>${h[4]}</td><td>${h[5]}</td><td>${h[6]}</td><td>${f3(h[2])}</td><td>${s3(h[3])}</td></tr>`
    : `<tr><td>${h[0]}</td><td>${h[7]}</td><td>${h[8]}</td><td>${h[2].toFixed(1)}</td><td>${h[3]}</td><td>${h[4]}</td><td>${h[5]}</td><td>${h[6].toFixed(2)}</td></tr>`).join('');
  const proj = hit ? `<tr class="proj"><td>2027</td><td>${p.pt}</td><td colspan="4"></td><td>${s3(p.rate)}</td></tr>`
                   : `<tr class="proj"><td>2027</td><td colspan="6">${p.pt} batters faced</td><td>${sgn(p.rate, 2)}</td></tr>`;
  const head = hit ? '<th>Year</th><th>PA</th><th>HR</th><th>BB</th><th>SO</th><th>wOBA</th><th>vs lg</th>'
                   : '<th>Year</th><th>G</th><th>GS</th><th>IP</th><th>K</th><th>BB</th><th>HR</th><th>RA9</th>';
  const back = prevHash && !/^#\/player\//.test(prevHash) ? prevHash : '#/players';
  view.innerHTML = `<div class="pvtop"><a class="back" href="${esc(back)}">‹ Back</a></div>
    <div class="pvname">${esc(p.n)}</div><p class="pvsub">${esc(sub)}</p><div class="stripe" style="--tc:${TC[abOf(p)]}"></div>
    <div class="tiles">
      <div class="tile"><div class="k">Runs vs average</div><div class="v">${pm(p.runs)}</div><div class="s">#${peers.indexOf(p) + 1} of ${peers.length.toLocaleString()} ${hit ? 'hitters' : 'pitchers'}</div></div>
      <div class="tile"><div class="k">Per ${hit ? '600 PA' : '200 IP'}</div><div class="v">${pm(p.std)}</div><div class="s">runs vs average</div></div>
      <div class="tile"><div class="k">${hit ? 'wOBA vs league' : 'RA9 vs league'}</div><div class="v">${hit ? s3(p.rate) : sgn(p.rate, 2)}</div><div class="s">${hit ? 'higher is better' : 'lower is better'}</div></div>
      <div class="tile"><div class="k">Projected ${hit ? 'PA' : 'batters faced'}</div><div class="v">${p.pt}</div><div class="s">${M.inputs.startsWith('winter') ? 'winter roster estimate' : 'opening-day roster'}</div></div>
    </div>
    ${breakdown(p)}
    <h2 class="sec">Last three seasons</h2>
    <div class="cardback"><div class="tbl"><table><thead><tr>${head}</tr></thead><tbody>${rows || `<tr><td colspan="8" class="l">No MLB time 2024-26.</td></tr>`}${proj}</tbody></table></div>
      <div class="tv">${BALL}<span>${esc(p.tv)}</span></div></div>`;
}

// ---------- placeholders
function tonight() {
  const d = new Date();
  view.innerHTML = `<div class="datebar"><button class="btn" disabled aria-label="Previous day">‹</button>
      <div><b>Tonight</b><span>${d.toLocaleDateString(undefined, {weekday: 'long', month: 'long', day: 'numeric'})}</span></div>
      <button class="btn" disabled aria-label="Next day">›</button></div>
    <div class="ph"><div class="ic" aria-hidden="true">⚾</div><h3>No games tonight</h3>
      <p>It's the offseason, and game odds aren't built yet. Once they are, this page lists every game with each team's win chance and the probable starters, with the Vegas line beside it.</p></div>`;
}
function vegas() {
  const b = M.backtest || [];
  view.innerHTML = `<h2 class="sec">UECKER vs Vegas</h2>
    <div class="ph"><div class="ic" aria-hidden="true">🎰</div><h3>2027 lines aren't in yet</h3>
      <p>When sportsbooks post 2027 win totals they'll show here next to UECKER's, team by team. Vegas is a yardstick only: it never enters the model.</p></div>
    <h2 class="sec">Past seasons</h2>
    <p class="note">Typical preseason miss in wins (root mean square error), 2016-2024 seasons without 2020. Lower is better.</p>
    <div class="tiles">${b.map(x => `<div class="tile"><div class="k">${esc(x[0])}</div><div class="v">${x[1].toFixed(2)}</div><div class="s">${esc(x[2])}</div></div>`).join('')}</div>`;
}
function settings() {
  const L = M.labels || {};
  view.innerHTML = `<h2 class="sec">Your changes</h2>
    <div class="ph"><div class="ic" aria-hidden="true">✏️</div><h3>Role editing is coming</h3>
      <p>Later you'll be able to set a player's lineup spot, rotation slot or bullpen role and watch the standings recompute.</p></div>
    <div class="prose"><h2 class="sec">How it works</h2>
    <p>UECKER (Unified Expected Contribution &amp; Key Efficiency Ratings) projects every player's rate and playing time, then adds up each 40-man roster into runs scored and allowed. Vegas lines are never an input.</p>
    <ul><li>${esc(L.wins)}</li><li>${esc(L.hit)}</li><li>${esc(L.pit)}</li><li>${esc(L.pt)}</li></ul>
    <p>Data built ${esc(new Date(M.built).toLocaleString())}.</p>
    <h3>Glossary</h3><dl>
     <dt>Runs (vs average)</dt><dd>Projected runs above an average player over his projected playing time: hitters count batting, baserunning, fielding and position; pitchers count runs saved. 0 is average.</dd>
     <dt>Per season</dt><dd>The same per 600 plate appearances (hitters) or 200 innings (pitchers).</dd>
     <dt>PA, BF</dt><dd>Plate appearances; batters faced.</dd>
     <dt>wOBA vs league</dt><dd>Weighted on-base average minus the league's (+.020 is 20 points better).</dd>
     <dt>RA9 vs league</dt><dd>Runs allowed per nine innings minus the league's (negative is better).</dd>
     <dt>Pythag</dt><dd>Wins that a team's runs scored and allowed would predict.</dd>
     <dt>SP, RP</dt><dd>Starter if he started at least half his games last season, otherwise reliever.</dd></dl></div>`;
}
function notFound() { view.innerHTML = '<div class="ph"><h3>Nothing here</h3><p><a href="#/standings">Go to the standings</a>.</p></div>'; }

// ---------- routing
const NAV = {standings, tonight, players, vegas, settings};
const SEC = {team: 'standings', player: 'players'};
let cur = '';
function route() {
  prevHash = cur; cur = location.hash || '#/standings';
  const [, a = 'standings', b] = decodeURIComponent(cur.slice(1)).split('/');
  const sec = SEC[a] || (NAV[a] ? a : 'standings');
  document.querySelectorAll('.bnav a').forEach(x => x.dataset.sec === sec ? x.setAttribute('aria-current', 'page') : x.removeAttribute('aria-current'));
  if (a === 'team') teamView(+b); else if (a === 'player') playerView(b); else (NAV[a] || standings)();
  window.scrollTo(0, 0);
}

const getJSON = u => fetch(u).then(r => { if (!r.ok) throw new Error(r.status + ' ' + u); return r.json(); });
Promise.all(['data/teams.json', 'data/players.json', 'data/meta.json'].map(getJSON)).then(([t, p, m]) => {
  T = t; P = p; M = m;
  T.forEach(x => teamById[x.id] = x);
  P.forEach(x => { x._s = norm(x.n); (byTeam[x.t ?? 'fa'] ||= []).push(x); });
  const b = new Date(M.built);
  $('#fresh').textContent = `${M.inputs.startsWith('winter') ? 'Winter update' : 'Update'} ${b.toLocaleString(undefined, {weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'})} · 40-man rosters`;
  window.addEventListener('hashchange', route); route();
}).catch(e => { view.innerHTML = `<div class="ph"><h3>The projections didn't load</h3><p>${esc(e.message)}. Reconnect and reload.</p></div>`; });

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
})();
