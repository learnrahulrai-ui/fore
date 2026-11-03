// Stock notes: a private per-ticker dump pad, with a per-ticker public switch.
//
// Why it lives here rather than in a repo of files: this repo is public, so
// notes committed as files would be public from the first keystroke. KV is
// server-side, so the CODE stays public while the NOTES stay private, and a
// ticker becomes public only when its switch is flipped.
//
// Auth is one secret, JOURNAL_KEY, set in the Cloudflare dashboard (never
// committed). Every private route demands it in the x-jk header. The only
// keyless route is /n/p/<ticker>, which serves a ticker already marked public.
//
// KV layout, on the existing COMPANY_CACHE binding, prefixed j: so it cannot
// collide with the analysis cache:
//     j:m:<TICKER>  ->  {name, public, created}
//     j:e:<TICKER>  ->  [{t: epochMillis, b: "body"}]   oldest first
// Tickers are listed by scanning the j:m: prefix, so there is no index to keep
// in step with reality.

const CORS = {
  'access-control-allow-origin': '*',
  'access-control-allow-methods': 'GET,POST,DELETE,OPTIONS',
  'access-control-allow-headers': 'content-type,x-jk',
};

const J = (o, status = 200) =>
  new Response(JSON.stringify(o), {
    status,
    headers: { 'content-type': 'application/json', ...CORS },
  });

const H = (html, status = 200) =>
  new Response(html, {
    status,
    headers: { 'content-type': 'text/html; charset=utf-8', ...CORS },
  });

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// A ticker is a key in a public URL, so keep it to characters that cannot
// confuse a path or a KV prefix.
const clean = (t) => String(t || '').trim().toUpperCase().replace(/[^A-Z0-9._-]/g, '').slice(0, 24);

const ist = (ms) =>
  new Intl.DateTimeFormat('en-IN', {
    timeZone: 'Asia/Kolkata', day: 'numeric', month: 'short', year: 'numeric',
    hour: 'numeric', minute: '2-digit', hour12: true,
  }).format(new Date(ms));

const istDay = (ms) =>
  new Intl.DateTimeFormat('en-IN', {
    timeZone: 'Asia/Kolkata', day: 'numeric', month: 'short', year: 'numeric',
  }).format(new Date(ms));

async function readMeta(env, ticker) {
  return (await env.COMPANY_CACHE.get(`j:m:${ticker}`, 'json')) || null;
}
async function readEntries(env, ticker) {
  return (await env.COMPANY_CACHE.get(`j:e:${ticker}`, 'json')) || [];
}

export async function journal(request, env, url) {
  const path = url.pathname;
  if (request.method === 'OPTIONS') return new Response(null, { headers: CORS });

  // ---- public read: the only route that needs no key ----------------------
  const pub = path.match(/^\/n\/p\/([^/]+)\/?$/);
  if (pub && request.method === 'GET') {
    const ticker = clean(decodeURIComponent(pub[1]));
    const meta = await readMeta(env, ticker);
    if (!meta || !meta.public) return H(notFoundPage(), 404);
    const entries = await readEntries(env, ticker);
    return H(publicPage(ticker, meta, entries));
  }

  // ---- the pad itself: plain HTML, asks for the key on first use ----------
  if (path === '/n' || path === '/n/') return H(padPage());

  // ---- everything below demands the key ----------------------------------
  if (!env.JOURNAL_KEY) {
    return J({ error: 'JOURNAL_KEY is not set on this Worker. Add it under Settings > Variables and Secrets in the Cloudflare dashboard, then reload.' }, 503);
  }
  if (request.headers.get('x-jk') !== env.JOURNAL_KEY) {
    return J({ error: 'bad key' }, 401);
  }

  // list every ticker with a count and the time of its newest entry
  if (path === '/n/list' && request.method === 'GET') {
    const out = [];
    let cursor;
    do {
      const page = await env.COMPANY_CACHE.list({ prefix: 'j:m:', cursor });
      for (const k of page.keys) {
        const ticker = k.name.slice(4);
        const meta = (await env.COMPANY_CACHE.get(k.name, 'json')) || {};
        const entries = await readEntries(env, ticker);
        out.push({
          ticker,
          name: meta.name || '',
          public: !!meta.public,
          count: entries.length,
          last: entries.length ? entries[entries.length - 1].t : meta.created || 0,
        });
      }
      cursor = page.list_complete ? null : page.cursor;
    } while (cursor);
    out.sort((a, b) => b.last - a.last);
    return J({ tickers: out });
  }

  if (path === '/n/get' && request.method === 'GET') {
    const ticker = clean(url.searchParams.get('t'));
    if (!ticker) return J({ error: 'need t' }, 400);
    const meta = (await readMeta(env, ticker)) || {};
    const entries = await readEntries(env, ticker);
    return J({ ticker, name: meta.name || '', public: !!meta.public, entries });
  }

  if (path === '/n/add' && request.method === 'POST') {
    let b;
    try { b = await request.json(); } catch { return J({ error: 'bad json' }, 400); }
    const ticker = clean(b.ticker);
    const body = String(b.body || '').trim();
    if (!ticker) return J({ error: 'need a ticker' }, 400);
    if (!body) return J({ error: 'nothing to save' }, 400);

    const meta = (await readMeta(env, ticker)) || { public: false, created: Date.now() };
    if (b.name && !meta.name) meta.name = String(b.name).slice(0, 120);

    const entries = await readEntries(env, ticker);
    // t comes from the server, so a phone with a wrong clock cannot misdate a
    // thought. b.t is honoured only for a queued offline entry being flushed.
    const t = Number.isFinite(b.t) && b.t > 0 && b.t <= Date.now() ? b.t : Date.now();
    entries.push({ t, b: body.slice(0, 20000) });
    entries.sort((x, y) => x.t - y.t);

    await env.COMPANY_CACHE.put(`j:e:${ticker}`, JSON.stringify(entries));
    await env.COMPANY_CACHE.put(`j:m:${ticker}`, JSON.stringify(meta));
    return J({ ok: true, ticker, count: entries.length, t });
  }

  if (path === '/n/pub' && request.method === 'POST') {
    let b;
    try { b = await request.json(); } catch { return J({ error: 'bad json' }, 400); }
    const ticker = clean(b.ticker);
    const meta = await readMeta(env, ticker);
    if (!meta) return J({ error: 'no such ticker' }, 404);
    meta.public = !!b.public;
    await env.COMPANY_CACHE.put(`j:m:${ticker}`, JSON.stringify(meta));
    return J({ ok: true, ticker, public: meta.public });
  }

  // remove one entry by its timestamp — for a typo or a duplicate
  if (path === '/n/del' && request.method === 'DELETE') {
    let b;
    try { b = await request.json(); } catch { return J({ error: 'bad json' }, 400); }
    const ticker = clean(b.ticker);
    const t = Number(b.t);
    const entries = await readEntries(env, ticker);
    const kept = entries.filter((e) => e.t !== t);
    await env.COMPANY_CACHE.put(`j:e:${ticker}`, JSON.stringify(kept));
    return J({ ok: true, removed: entries.length - kept.length });
  }

  // whole journal, for a backup or for rendering into static pages
  if (path === '/n/export' && request.method === 'GET') {
    const out = {};
    let cursor;
    do {
      const page = await env.COMPANY_CACHE.list({ prefix: 'j:m:', cursor });
      for (const k of page.keys) {
        const ticker = k.name.slice(4);
        out[ticker] = {
          meta: await env.COMPANY_CACHE.get(k.name, 'json'),
          entries: await readEntries(env, ticker),
        };
      }
      cursor = page.list_complete ? null : page.cursor;
    } while (cursor);
    return new Response(JSON.stringify(out, null, 2), {
      headers: {
        'content-type': 'application/json',
        'content-disposition': `attachment; filename="stock-notes-${new Date().toISOString().slice(0, 10)}.json"`,
        ...CORS,
      },
    });
  }

  return J({ error: 'no such route' }, 404);
}

// ---------------------------------------------------------------------------

function shellCss() {
  return `
  :root { --ink:#111; --dim:#666; --line:#d8d8d8; --bg:#fcfcfa; --accent:#7a1f1f; }
  * { box-sizing:border-box; -webkit-text-size-adjust:100%; }
  body { font-family:Georgia,"Times New Roman",serif; background:var(--bg); color:var(--ink);
         margin:0; padding:0 1em 4em; line-height:1.6; }
  .wrap { max-width:720px; margin:0 auto; padding-top:1.2em; }
  h1 { font-size:1.15em; font-weight:normal; font-style:italic; margin:0 0 .2em; }
  .sub { color:var(--dim); font-size:.86em; margin:0 0 1.2em; }
  textarea, input, button, select {
    font-family:inherit; font-size:1em; color:var(--ink);
    border:1px solid var(--line); border-radius:3px; background:#fff; }
  textarea { width:100%; min-height:9em; padding:.7em; line-height:1.55; resize:vertical; }
  input { width:100%; padding:.6em .7em; }
  button { padding:.75em 1.1em; background:var(--ink); color:#fff; border:none;
           border-radius:3px; cursor:pointer; font-size:1em; }
  button.ghost { background:none; color:var(--accent); border:1px solid var(--line); }
  button:active { opacity:.75; }
  .row { display:flex; gap:.5em; align-items:center; margin:.6em 0; }
  .row > :first-child { flex:1; }
  .e { border-top:1px solid var(--line); padding:.9em 0; }
  .e .when { color:var(--dim); font-size:.8em; letter-spacing:.02em; }
  .e .body { white-space:pre-wrap; margin-top:.25em; }
  .day { font-size:.82em; color:var(--dim); font-style:italic; margin:1.6em 0 .2em; }
  .msg { font-size:.86em; color:var(--accent); min-height:1.3em; }
  .pill { display:inline-block; font-size:.74em; padding:.15em .5em; border:1px solid var(--line);
          border-radius:10px; color:var(--dim); margin-left:.4em; }
  a { color:var(--accent); }
  .tlist a { display:block; padding:.45em 0; border-top:1px solid var(--line); text-decoration:none; color:var(--ink); }
  .tlist .meta { color:var(--dim); font-size:.8em; }
  @media (prefers-color-scheme:dark) {
    :root { --ink:#e8e6e0; --dim:#9a9a94; --line:#333; --bg:#16181a; --accent:#d99; }
    textarea, input { background:#1e2124; }
    button { background:#e8e6e0; color:#16181a; }
  }`;
}

// The pad. One screen: pick a ticker, type, save. Saves that fail are queued in
// the browser and retried, because the likeliest moment to have a thought is on
// a train with two bars of signal.
function padPage() {
  return `<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>Notes</title><style>${shellCss()}</style></head><body><div class="wrap">

<h1>Notes</h1>
<p class="sub" id="sub">Pick a ticker, type, save. Nothing here is public until you switch it.</p>

<div id="gate" style="display:none">
  <p class="sub">Paste the key you set as <b>JOURNAL_KEY</b> in the Cloudflare dashboard. It is kept in this browser only.</p>
  <div class="row"><input id="key" type="password" placeholder="JOURNAL_KEY" autocomplete="off"><button onclick="saveKey()">Use</button></div>
</div>

<div id="app" style="display:none">
  <div class="row">
    <input id="ticker" list="known" placeholder="TICKER (e.g. ASKAUTOLTD)" autocapitalize="characters" autocomplete="off">
    <button class="ghost" onclick="loadList()" title="all tickers">&#9776;</button>
  </div>
  <datalist id="known"></datalist>
  <textarea id="body" placeholder="Whatever you just heard or worked out. Raw is fine."></textarea>
  <div class="row">
    <button onclick="save()" id="savebtn">Save</button>
    <button class="ghost" onclick="togglePub()" id="pubbtn">&nbsp;</button>
    <button class="ghost" onclick="location.href='/n/export'" title="download everything">&#8681;</button>
  </div>
  <p class="msg" id="msg"></p>
  <div id="entries"></div>
  <div id="all" class="tlist"></div>
</div>

<script>
const $ = id => document.getElementById(id);
let KEY = localStorage.getItem('jk') || '';
let CUR = { ticker:'', public:false };

function hdr(){ return { 'content-type':'application/json', 'x-jk':KEY }; }
function show(){ $('gate').style.display = KEY ? 'none':'block'; $('app').style.display = KEY ? 'block':'none'; }
function saveKey(){ KEY = $('key').value.trim(); if(!KEY) return; localStorage.setItem('jk',KEY); show(); boot(); }
function note(m){ $('msg').textContent = m; if(m) setTimeout(()=>{ if($('msg').textContent===m) $('msg').textContent=''; }, 4000); }

// --- offline queue ---------------------------------------------------------
const Q = {
  get(){ try { return JSON.parse(localStorage.getItem('jq')||'[]'); } catch { return []; } },
  set(v){ localStorage.setItem('jq', JSON.stringify(v)); },
  push(e){ const q=this.get(); q.push(e); this.set(q); },
};
async function flush(){
  let q = Q.get(); if(!q.length) return;
  const left = [];
  for (const e of q) {
    try {
      const r = await fetch('/n/add',{method:'POST',headers:hdr(),body:JSON.stringify(e)});
      if(!r.ok) left.push(e);
    } catch { left.push(e); }
  }
  Q.set(left);
  if (q.length && !left.length) note('Saved ' + q.length + ' queued note' + (q.length>1?'s':'') + '.');
}

async function save(){
  const ticker = $('ticker').value.trim().toUpperCase();
  const body = $('body').value.trim();
  if(!ticker){ note('Which stock?'); $('ticker').focus(); return; }
  if(!body){ note('Nothing typed.'); return; }
  const payload = { ticker, body, t: Date.now() };
  $('savebtn').disabled = true;
  try {
    const r = await fetch('/n/add',{method:'POST',headers:hdr(),body:JSON.stringify(payload)});
    const j = await r.json();
    if(!r.ok) throw new Error(j.error||'failed');
    $('body').value = '';
    localStorage.setItem('lastticker', ticker);
    note('Saved. ' + j.count + ' note' + (j.count>1?'s':'') + ' on ' + ticker + '.');
    await openTicker(ticker);
  } catch (err) {
    Q.push(payload); $('body').value = '';
    note('No connection — queued. It will go up by itself.');
  } finally { $('savebtn').disabled = false; }
}

async function openTicker(t){
  if(!t) return;
  CUR.ticker = t;
  const r = await fetch('/n/get?t='+encodeURIComponent(t),{headers:hdr()});
  const j = await r.json();
  CUR.public = !!j.public;
  $('pubbtn').textContent = j.public ? 'Public \\u2713' : 'Make public';
  $('all').innerHTML = '';
  const es = (j.entries||[]).slice().reverse();
  $('entries').innerHTML = es.length
    ? es.map(e => '<div class="e"><div class="when">'+fmt(e.t)+'</div><div class="body">'+escp(e.b)+'</div></div>').join('')
    : '<p class="sub">Nothing on '+escp(t)+' yet.</p>';
}

async function togglePub(){
  if(!CUR.ticker){ note('Open a ticker first.'); return; }
  const want = !CUR.public;
  if (want && !confirm('Put every note on '+CUR.ticker+' on a public page?')) return;
  const r = await fetch('/n/pub',{method:'POST',headers:hdr(),body:JSON.stringify({ticker:CUR.ticker,public:want})});
  const j = await r.json();
  if(!r.ok){ note(j.error||'failed'); return; }
  CUR.public = j.public;
  $('pubbtn').textContent = j.public ? 'Public \\u2713' : 'Make public';
  note(j.public ? 'Live at /n/p/'+CUR.ticker : 'Private again.');
}

async function loadList(){
  const r = await fetch('/n/list',{headers:hdr()});
  const j = await r.json();
  $('known').innerHTML = (j.tickers||[]).map(t=>'<option value="'+escp(t.ticker)+'">').join('');
  $('entries').innerHTML = '';
  $('all').innerHTML = (j.tickers||[]).map(t =>
    '<a href="#" onclick="pick(\\''+escp(t.ticker)+'\\');return false">'+escp(t.ticker)+
    (t.public?' <span class="pill">public</span>':'')+
    '<div class="meta">'+t.count+' note'+(t.count===1?'':'s')+' \\u00b7 '+fmt(t.last)+'</div></a>').join('')
    || '<p class="sub">No tickers yet. Type one above.</p>';
}
function pick(t){ $('ticker').value = t; openTicker(t); }

function fmt(ms){ try { return new Intl.DateTimeFormat('en-IN',{timeZone:'Asia/Kolkata',day:'numeric',month:'short',year:'numeric',hour:'numeric',minute:'2-digit',hour12:true}).format(new Date(ms)); } catch { return ''; } }
function escp(s){ const d=document.createElement('div'); d.textContent=s==null?'':s; return d.innerHTML; }

$('body').addEventListener('keydown', e => {
  if ((e.metaKey||e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); save(); }
});
$('ticker').addEventListener('change', () => openTicker($('ticker').value.trim().toUpperCase()));
window.addEventListener('online', flush);

async function boot(){
  await flush();
  await loadList();
  const last = localStorage.getItem('lastticker');
  if(last){ $('ticker').value = last; await openTicker(last); }
  $('body').focus();
}
show(); if(KEY) boot();
</script>
</div></body></html>`;
}

function publicPage(ticker, meta, entries) {
  const es = entries.slice().reverse();
  let lastDay = '';
  const items = es.map((e) => {
    const d = istDay(e.t);
    const head = d === lastDay ? '' : `<p class="day">${esc(d)}</p>`;
    lastDay = d;
    return `${head}<div class="e"><div class="when">${esc(ist(e.t))}</div><div class="body">${esc(e.b)}</div></div>`;
  }).join('');

  return `<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${esc(ticker)}${meta.name ? ' — ' + esc(meta.name) : ''} — notes</title>
<style>${shellCss()}</style></head><body><div class="wrap">
<h1>${esc(ticker)}${meta.name ? ' &mdash; ' + esc(meta.name) : ''}</h1>
<p class="sub">Working notes, newest first. ${entries.length} entr${entries.length === 1 ? 'y' : 'ies'}.
Written as they occurred and left standing; nothing here is edited after the fact.</p>
<hr style="border:none;border-top:1px solid var(--line)">
${items || '<p class="sub">Nothing yet.</p>'}
<hr style="border:none;border-top:1px solid var(--line);margin-top:2em">
<p class="sub">Notes, not advice on where to put money. No target prices and no recommendations.
&nbsp;&middot;&nbsp; <a href="https://learnrahulrai-ui.github.io/fore/promoters.html">Full reports</a></p>
</div></body></html>`;
}

function notFoundPage() {
  return `<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Not here</title><style>${shellCss()}</style></head><body><div class="wrap">
<h1>Nothing public under that name</h1>
<p class="sub">Either no notes exist for it, or they have not been made public.
&nbsp;&middot;&nbsp; <a href="https://learnrahulrai-ui.github.io/fore/promoters.html">Full reports</a></p>
</div></body></html>`;
}
