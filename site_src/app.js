// UECKER app v1: team board, team pages, flip cards, search. Reads data/teams.json, players.json, meta.json.
(() => {
'use strict';
// [bar colour, letter colour, sunburst alt] per team (colours only; no logos)
const TC = {
  ARI:['#A71930','#E3D4AD','#30CED8'], ATL:['#13274F','#EAAA00','#CE1141'], BAL:['#DF4601','#FFFFFF','#1A1A1A'],
  BOS:['#BD3039','#FFFFFF','#0C2340'], CHC:['#0E3386','#FFFFFF','#CC3433'], CWS:['#27251F','#C4CED4','#5B5B5B'],
  CIN:['#C6011F','#FFFFFF','#1A1A1A'], CLE:['#00385D','#FFFFFF','#E50022'], COL:['#33006F','#C4CED4','#1A1A1A'],
  DET:['#0C2340','#FA4616','#5A6B86'], HOU:['#002D62','#EB6E1F','#F4911E'], KC:['#004687','#BD9B60','#7AB2DD'],
  LAA:['#BA0021','#FFFFFF','#003263'], LAD:['#005A9C','#FFFFFF','#EF3E42'], MIA:['#00A3E0','#FFFFFF','#EF3340'],
  MIL:['#12284B','#FFC52F','#2C5A8F'], MIN:['#002B5C','#FFFFFF','#D31145'], NYM:['#002D72','#FF5910','#4A78B8'],
  NYY:['#0C2340','#FFFFFF','#8A96A8'], ATH:['#003831','#EFB21E','#3E7A52'], PHI:['#E81828','#FFFFFF','#002D72'],
  PIT:['#27251F','#FDB827','#5B5B5B'], SD:['#2F241D','#FFC425','#6B5A48'], SF:['#27251F','#FD5A1E','#5B5B5B'],
  SEA:['#0C2C56','#C4CED4','#005C5C'], STL:['#C41E3A','#FFFFFF','#0C2340'], TB:['#092C5C','#F5D130','#8FBCE6'],
  TEX:['#003278','#FFFFFF','#C0111F'], TOR:['#134A8E','#FFFFFF','#E8291C'], WSH:['#AB0003','#FFFFFF','#14225A'],
  FA:['#6B6B6B','#F4C430','#3A3A3A']};
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sgn = (x, d=1) => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(d);
const f3 = x => x == null ? '—' : x.toFixed(3).replace(/^0/, '').replace(/^-0/, '−');
const sgn3 = x => x == null ? '—' : (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(3).replace(/^0/, '');
let T = [], P = [], M = {}, byTeam = {}, teamById = {};
const view = $('#view');

const colours = ab => { const c = TC[ab] || TC.FA; return `--tc:${c[0]};--tc2:${c[1]};--tc3:${c[2]}`; };
const initials = n => n.split(/\s+/).filter(w => !/^(Jr\.?|Sr\.?|II|III|IV)$/.test(w)).map(w => w[0]).slice(0, 2).join('').toUpperCase();
const role = p => {
  if (p.k === 'hit') return p.pos === 'TWP' ? 'DH' : (p.pos || '—');
  const last = p.h[p.h.length - 1];
  return last && last[8] / Math.max(1, last[7]) >= 0.5 ? 'SP' : 'RP';
};

// ---------- cards
const TRIV = `<svg viewBox="0 0 40 40" aria-hidden="true"><circle cx="20" cy="20" r="17" fill="#FBF6E8" stroke="#17213A" stroke-width="2.5"/>
<path d="M9 9c4 4 4 18 0 22M31 9c-4 4-4 18 0 22" fill="none" stroke="#C8302C" stroke-width="2" stroke-dasharray="2.5 2"/>
<circle cx="15" cy="17" r="2" fill="#17213A"/><circle cx="25" cy="17" r="2" fill="#17213A"/><path d="M14 24q6 5 12 0" fill="none" stroke="#17213A" stroke-width="2" stroke-linecap="round"/></svg>`;

function card(p) {
  const t = teamById[p.t], ab = t ? t.ab : 'FA';
  const hit = p.k === 'hit', r = role(p);
  const unit = hit ? 'PA' : 'BF';
  const rows = p.h.map(h => hit
    ? `<tr><td>'${String(h[0]).slice(2)}</td><td>${h[1]}</td><td>${h[4]}</td><td>${h[5]}</td><td>${h[6]}</td><td>${f3(h[2])}</td><td>${sgn3(h[3])}</td></tr>`
    : `<tr><td>'${String(h[0]).slice(2)}</td><td>${h[1]}</td><td>${h[2].toFixed(0)}</td><td>${h[3]}</td><td>${h[4]}</td><td>${h[5]}</td><td>${h[6].toFixed(2)}</td></tr>`).join('');
  const proj = hit
    ? `<tr class="proj"><td>'27</td><td>${p.pt}</td><td colspan="4">${sgn(p.runs)} runs</td><td>${sgn3(p.rate)}</td></tr>`
    : `<tr class="proj"><td>'27</td><td>${p.pt}</td><td colspan="4">${sgn(p.runs)} runs</td><td>${sgn(p.rate, 2)}</td></tr>`;
  const head = hit ? '<th>YR</th><th>PA</th><th>HR</th><th>BB</th><th>SO</th><th>wOBA</th><th>REL</th>'
                   : '<th>YR</th><th>BF</th><th>IP</th><th>K</th><th>BB</th><th>HR</th><th>RA9</th>';
  const bt = p.b && p.th ? `B/T ${p.b}/${p.th}` : '';
  return `<button class="card" style="${colours(ab)}" aria-label="${esc(p.n)}, ${sgn(p.runs)} runs. Tap to flip">
  <div class="flip">
    <div class="face front">
      <div class="art"><div class="ini">${esc(initials(p.n))}</div>
        <div class="val">${sgn(p.runs)}<small>runs vs avg</small></div><div class="badge">${esc(r)}</div></div>
      <div class="plate"><div class="pn">${esc(p.n)}</div>
        <div class="pm"><span>${esc(ab)}${p.age ? ' · age ' + Math.floor(p.age) : ''}</span><span>${p.pt} ${unit}</span></div></div>
    </div>
    <div class="face back">
      <div class="bh"><span>${esc(p.n)}</span><span>${esc(r)}</span></div>
      <div class="bio">${esc(bt)}${p.age ? `${bt ? ', ' : ''}age ${p.age.toFixed(1)} in '27` : ''}</div>
      <table><thead><tr>${head}</tr></thead><tbody>${rows}${proj}</tbody></table>
      <div class="tv">${TRIV}<span>${esc(p.tv)}</span></div>
    </div>
  </div></button>`;
}

function grid(list, step = 40) {
  let n = 0;
  const wrap = document.createElement('div');
  const g = document.createElement('div'); g.className = 'grid';
  const btn = document.createElement('button'); btn.className = 'more';
  const add = () => {
    g.insertAdjacentHTML('beforeend', list.slice(n, n + step).map(card).join(''));
    n += step; btn.hidden = n >= list.length; btn.textContent = `Show ${Math.min(step, list.length - n)} more`;
  };
  btn.onclick = add; add();
  wrap.append(g, btn);
  return wrap;
}

// ---------- views
let league = 'MLB';
function board() {
  const seg = ['MLB', 'AL', 'NL', 'Divisions'].map(x => `<button aria-pressed="${league === x}" data-l="${x}">${x}</button>`).join('');
  const sorted = [...T].sort((a, b) => b.w - a.w);
  const bar = (t, i) => `<li><a class="bar" href="#/team/${t.id}" style="${colours(t.ab)}">
    <span class="rk">${i + 1}</span>
    <span class="nm"><div class="tn">${esc(t.name)}</div>
      <div class="ln">RS ${t.rs} · RA ${t.ra} · <span class="${t.rd >= 0 ? 'pos' : 'neg'}">${sgn(t.rd, 0)}</span></div>
      <div class="ln">'26 Pythag ${t.py.toFixed(1)}</div></span>
    <span class="w">${t.w.toFixed(1)}<span>wins</span></span></a></li>`;
  let body;
  if (league === 'Divisions') {
    const divs = ['AL East', 'AL Central', 'AL West', 'NL East', 'NL Central', 'NL West'];
    body = divs.map(d => `<h2 class="div-h">${d}</h2><ol class="board">${sorted.filter(t => t.div === d).map(bar).join('')}</ol>`).join('');
  } else {
    const l = sorted.filter(t => league === 'MLB' || (t.div || '').startsWith(league));
    body = `<ol class="board">${l.map(bar).join('')}</ol>`;
  }
  view.innerHTML = `<div class="seg" role="group" aria-label="League">${seg}</div>${body}
    <a class="fa-link" href="#/team/fa">Unsigned players (${(byTeam.fa || []).length})</a>`;
  view.querySelectorAll('.seg button').forEach(b => b.onclick = () => { league = b.dataset.l; board(); });
}

let side = 'hit';
function teamPage(id) {
  const fa = id === 'fa', t = fa ? null : teamById[id];
  if (!fa && !t) { view.innerHTML = '<p class="empty">No team with that id. <a href="#/">Back to the board</a>.</p>'; return; }
  const list = (byTeam[fa ? 'fa' : id] || []).filter(p => p.k === side).sort((a, b) => b.runs - a.runs);
  const nH = (byTeam[fa ? 'fa' : id] || []).filter(p => p.k === 'hit').length;
  const head = fa
    ? `<div class="th" style="${colours('FA')}"><div class="band">Unsigned</div>
       <p class="sub" style="padding:8px 12px 0;margin:0 0 10px">Players with projected playing time and no 40-man spot today. Not counted in any team's total.</p></div>`
    : `<div class="th" style="${colours(t.ab)}"><div class="band">${esc(t.name)}</div>
       <div class="nums"><div><b>${t.w.toFixed(1)}</b><span>wins</span></div><div><b>${t.rs}</b><span>runs scored</span></div>
       <div><b>${t.ra}</b><span>runs allowed</span></div><div><b>${t.py.toFixed(1)}</b><span>'26 Pythag</span></div></div></div>`;
  view.innerHTML = `<a class="back-link" href="#/">‹ Board</a>${head}
    <div class="seg" role="group" aria-label="Side"><button aria-pressed="${side === 'hit'}" data-s="hit">Hitters (${nH})</button>
    <button aria-pressed="${side === 'pit'}" data-s="pit">Pitchers (${(byTeam[fa ? 'fa' : id] || []).length - nH})</button></div>
    <p class="sub">Sorted by projected runs vs an average player. Tap a card to flip it.</p>`;
  view.querySelectorAll('.seg button').forEach(b => b.onclick = () => { side = b.dataset.s; teamPage(id); });
  view.append(list.length ? grid(list) : Object.assign(document.createElement('p'), {className: 'empty', textContent: 'No projected players.'}));
}

function searchPage(q) {
  const k = q.trim().toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
  const hits = k.length < 2 ? [] : P.filter(p => p._s.includes(k)).sort((a, b) => b.runs - a.runs);
  view.innerHTML = k.length < 2 ? '<p class="empty">Type at least two letters of a player\'s name.</p>'
    : `<p class="sub">${hits.length} player card${hits.length === 1 ? '' : 's'} for “${esc(q)}”</p>`;
  if (hits.length) view.append(grid(hits, 24));
  else if (k.length >= 2) view.insertAdjacentHTML('beforeend', '<p class="empty">No projected player matches that name. Check the spelling, or try a last name.</p>');
}

function about() {
  const L = M.labels || {};
  view.innerHTML = `<a class="back-link" href="#/">‹ Board</a><div class="about">
  <h2>About UECKER ${M.season}</h2>
  <p>Unified Expected Contribution &amp; Key Efficiency Ratings: a ground-up projection model for the ${M.season} season. Every number here comes from the shipped model; Vegas is never an input.</p>
  <h3>Models in use</h3><ul><li>${esc(L.wins)}</li><li>${esc(L.hit)}</li><li>${esc(L.pit)}</li><li>${esc(L.pt)}</li></ul>
  <p>Data last built ${esc(new Date(M.built).toLocaleString())}.</p>
  <h3>Glossary</h3><dl>
   <dt>Runs vs avg</dt><dd>Projected runs above an average player over the projected playing time. Hitters: runs created; pitchers: runs saved. 0 is league average.</dd>
   <dt>PA / BF</dt><dd>Projected plate appearances (hitters) or batters faced (pitchers).</dd>
   <dt>wOBA, REL</dt><dd>Weighted on-base average, and the same relative to that season's league (+.020 is 20 points better).</dd>
   <dt>RA9</dt><dd>Runs allowed per nine innings. A pitcher's '27 row shows projected RA9 relative to league (negative is better).</dd>
   <dt>RS, RA</dt><dd>Projected team runs scored and allowed.</dd>
   <dt>'26 Pythag</dt><dd>Wins last season's runs scored and allowed would have predicted.</dd>
   <dt>SP / RP</dt><dd>Starter if he started at least half his games last season. Role editing comes in a later version.</dd>
  </dl></div>`;
}

function route() {
  const h = decodeURIComponent(location.hash.slice(1) || '/');
  const [, a, b] = h.split('/');
  if (a !== 's') $('#q').value = '';
  if (a === 'team') teamPage(b === 'fa' ? 'fa' : +b);
  else if (a === 's') searchPage(b || '');
  else if (a === 'about') about();
  else board();
  if (a !== 's') window.scrollTo(0, 0);
}

view.addEventListener('click', e => { const c = e.target.closest('.card'); if (c) c.classList.toggle('on'); });
let qt;
$('#q').addEventListener('input', e => { clearTimeout(qt); qt = setTimeout(() => {
  const v = e.target.value; history.replaceState(null, '', v ? '#/s/' + encodeURIComponent(v) : '#/'); route(); }, 150); });

const getJSON = u => fetch(u).then(r => { if (!r.ok) throw new Error(r.status + ' ' + u); return r.json(); });
Promise.all(['data/teams.json', 'data/players.json', 'data/meta.json'].map(getJSON)).then(([t, p, m]) => {
  T = t; P = p; M = m;
  T.forEach(x => teamById[x.id] = x);
  P.forEach(x => { x._s = x.n.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, ''); (byTeam[x.t ?? 'fa'] ||= []).push(x); });
  $('#built').textContent = `${M.labels.wins}. Built ${new Date(M.built).toLocaleDateString()}.`;
  window.addEventListener('hashchange', route); route();
}).catch(e => { view.innerHTML = `<p class="empty">The projections didn't load (${esc(e.message)}). Reconnect and pull to refresh.</p>`; });

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
})();
