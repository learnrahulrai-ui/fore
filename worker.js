// ---------------------------------------------------------------------------
// Grounded analysis worker
//
// Flow:
//   1. Extract company name from the PDF.
//   2. Fan out ~6 targeted web searches (Brave API) about the company.
//   3. Run every snippet through the GATE: the model proposes facts WITH a
//      verbatim quote + source id; code verifies the quote is really in that
//      snippet. Anything that fails is dropped. The model is never the source.
//   4. Cache the verified research + raw search in KV (never re-search a company).
//   5. Step 2: analyze the PDF with the verified facts as context.
//   6. Step 3: condense into the final report.
//
// Search is server-side (worker), so there is no CORS wall and the key stays
// secret. The raw search is returned to the browser so the user can see and
// verify what was looked up. If BRAVE_API_KEY is absent it falls back to
// Wikipedia so the app still runs.
// ---------------------------------------------------------------------------

const ALLOWED_ORIGIN = 'https://learnrahulrai-ui.github.io';

const FAST_MODEL = '@cf/meta/llama-3.1-8b-instruct';
const MAIN_MODEL = '@cf/meta/llama-3.3-70b-instruct-fp8-fast';

const BRAVE_ENDPOINT = 'https://api.search.brave.com/res/v1/web/search';

// Results from these domains are tagged "primary" so they stand out.
const PRIMARY_DOMAINS = new Set([
  'sebi.gov.in', 'nseindia.com', 'bseindia.com', 'mca.gov.in',
  'indiankanoon.org', 'nclt.gov.in', 'rbi.org.in',
]);

const CORS = {
  'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status, headers: { 'content-type': 'application/json', ...CORS },
  });
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

export default {
  async fetch(request, env) {
    if (request.method === 'OPTIONS') return new Response(null, { headers: CORS });
    if (request.method !== 'POST') return new Response('Method not allowed', { status: 405 });

    let body;
    try { body = await request.json(); }
    catch { return json({ error: 'Invalid JSON' }, 400); }
    if (!body.text) return json({ error: 'Missing text' }, 400);

    try {
      const pdfText = body.text;

      const companyName = await extractCompanyName(env, pdfText);
      const cacheKey = 'co_' + companyName.toLowerCase()
        .replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '').slice(0, 80);

      // --- Step 1: grounded research (cache-first) ---
      let research, sources, droppedCount, searchHits, fromCache = false;

      if (cacheKey !== 'co_' && env.COMPANY_CACHE) {
        try {
          const cached = await env.COMPANY_CACHE.get(cacheKey, 'json');
          if (cached && cached.research !== undefined) {
            ({ research, sources, droppedCount, searchHits } = cached);
            fromCache = true;
          }
        } catch { /* old/stale entry -> miss */ }
      }

      if (research === undefined) {
        const r = await groundedResearch(env, companyName);
        research = r.research; sources = r.sources;
        droppedCount = r.dropped; searchHits = r.searchHits;
        if (cacheKey !== 'co_' && env.COMPANY_CACHE) {
          await env.COMPANY_CACHE.put(cacheKey,
            JSON.stringify({ research, sources, droppedCount, searchHits }),
            { expirationTtl: 2592000 });
        }
      }

      // --- Step 2: analyze the PDF with verified facts as context ---
      const analysis = await runModel(env, MAIN_MODEL, env.SYSTEM_PROMPT_2,
        `VERIFIED COMPANY FACTS:\n${research}\n\nDOCUMENT:\n${pdfText}`, 2048);

      // --- Step 3: condense ---
      const report = await runModel(env, MAIN_MODEL, env.SYSTEM_PROMPT_3, analysis, 1024);

      return json({
        company: companyName, cached: fromCache, research, sources,
        dropped_count: droppedCount, search: searchHits || [], analysis, report,
      });
    } catch (e) {
      return json({ error: e.message || String(e) }, 500);
    }
  },
};

// ---------------------------------------------------------------------------
// Model helpers
// ---------------------------------------------------------------------------
async function runModel(env, model, system, user, maxTokens) {
  const out = await env.AI.run(model, {
    messages: [
      { role: 'system', content: system || '' },
      { role: 'user', content: user },
    ],
    max_tokens: maxTokens,
  });
  return (out.response ?? '').trim();
}

async function extractCompanyName(env, pdfText) {
  const out = await env.AI.run(FAST_MODEL, {
    messages: [
      { role: 'system', content: 'Extract only the primary company name from this financial document. Reply with just the name, nothing else.' },
      { role: 'user', content: pdfText.slice(0, 4000) },
    ],
    max_tokens: 30,
  });
  return (out.response ?? '').trim().replace(/^["']+|["']+$/g, '');
}

// ---------------------------------------------------------------------------
// 1. RETRIEVAL
// ---------------------------------------------------------------------------
function trustOf(url) {
  const dom = url.replace(/^https?:\/\/(www\.)?/, '').split('/')[0];
  if (PRIMARY_DOMAINS.has(dom)) return 'primary';
  if (dom.endsWith('wikipedia.org')) return 'reference';
  return 'news';
}

// The targeted query fan-out — one per "trick" you care about.
function makeQueries(name) {
  const q = `"${name}"`;
  return [
    `${q} promoters directors`,
    `${q} SEBI penalty order action`,
    `${q} QIP FPO preferential allotment equity dilution`,
    `${q} subsidiaries related party transactions`,
    `${q} debt default fraud investigation`,
    `${q} promoter pledge warrants`,
  ];
}

async function braveSearch(query, key) {
  const url = BRAVE_ENDPOINT + '?count=3&q=' + encodeURIComponent(query);
  const res = await fetch(url, {
    headers: { 'Accept': 'application/json', 'X-Subscription-Token': key },
  });
  if (!res.ok) return [];
  const data = await res.json();
  const results = data?.web?.results ?? [];
  return results.map(r => {
    const extra = Array.isArray(r.extra_snippets) ? r.extra_snippets.join(' ') : '';
    const text = [r.description, extra].filter(Boolean).join(' ').replace(/<\/?[^>]+>/g, '');
    return { url: r.url, title: (r.title || '').replace(/<\/?[^>]+>/g, ''), text };
  }).filter(r => r.text && r.text.length > 40);
}

async function fetchSourcesBrave(companyName, key) {
  const sources = {};
  const searchHits = [];
  const queries = makeQueries(companyName);
  for (let i = 0; i < queries.length; i++) {
    try {
      const hits = await braveSearch(queries[i], key);
      for (const h of hits) {
        const id = 's' + Object.keys(sources).length;
        sources[id] = { id, url: h.url, trust: trustOf(h.url), text: h.text };
        searchHits.push({ query: queries[i], title: h.title, url: h.url, snippet: h.text });
      }
    } catch { /* skip a failed query, keep going */ }
    if (i < queries.length - 1) await sleep(1100); // free tier = 1 req/sec
  }
  return { sources, searchHits };
}

// Fallback when no Brave key is set.
async function fetchSourcesWikipedia(companyName) {
  const sources = {};
  const UA = { 'User-Agent': 'fore-analyzer/1.0 (educational project)' };
  try {
    const searchUrl = 'https://en.wikipedia.org/w/api.php?action=query&list=search'
      + '&srsearch=' + encodeURIComponent(companyName) + '&srlimit=2&format=json&origin=*';
    const sj = await (await fetch(searchUrl, { headers: UA })).json();
    for (const h of (sj?.query?.search ?? [])) {
      const exUrl = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts'
        + '&explaintext=1&redirects=1&titles=' + encodeURIComponent(h.title) + '&format=json&origin=*';
      const ej = await (await fetch(exUrl, { headers: UA })).json();
      const pages = ej?.query?.pages ?? {};
      for (const pid in pages) {
        const text = pages[pid].extract;
        if (text && text.length > 200) {
          const url = 'https://en.wikipedia.org/wiki/' + encodeURIComponent(h.title.replace(/ /g, '_'));
          const id = 's' + Object.keys(sources).length;
          sources[id] = { id, url, trust: trustOf(url), text };
        }
      }
    }
  } catch {}
  return sources;
}

async function fetchSources(env, companyName) {
  if (env.BRAVE_API_KEY) {
    const { sources, searchHits } = await fetchSourcesBrave(companyName, env.BRAVE_API_KEY);
    if (Object.keys(sources).length > 0) return { sources, searchHits };
  }
  const sources = await fetchSourcesWikipedia(companyName);
  return { sources, searchHits: [] };
}

// ---------------------------------------------------------------------------
// 2. EXTRACTION — model returns facts, each WITH a verbatim quote + source id.
// ---------------------------------------------------------------------------
const EXTRACT_PROMPT = `You are given SOURCES: real search-result text we fetched. Extract the FIELDS.
For EVERY value, copy an exact verbatim quote (at least 8 words) from the SOURCE it came
from, and give that source's id. Copy the quote character-for-character; do not paraphrase.
If a field is not present in any SOURCE, do not include it. Invent nothing - no name,
company, date, or action that is not in the text.

Return ONLY a JSON array, no prose:
[{"field":"...","value":"...","source_id":"...","quote":"..."}]

FIELDS:
- promoter_names (one row per person)
- related_companies (subsidiaries, parents, associates; one row each)
- regulatory_actions (value format: "<what happened> | legal_status: alleged|order|conviction")
- capital_events (QIP, FPO, preferential allotment, warrants, dilution; one row each)
- distress_signals (debt default, pledge, fraud/investigation; one row each)

SOURCES:
`;

async function extractFacts(env, sources) {
  const ids = Object.keys(sources);
  if (ids.length === 0) return [];
  const blob = ids.map(id => `[${id}] ${sources[id].url}\n${sources[id].text.slice(0, 2000)}`).join('\n\n');
  const out = await env.AI.run(MAIN_MODEL, {
    messages: [
      { role: 'system', content: 'You extract only facts present verbatim in the provided sources. You never invent.' },
      { role: 'user', content: EXTRACT_PROMPT + blob },
    ],
    max_tokens: 2000,
  });
  return parseFacts(out.response ?? '');
}

function parseFacts(raw) {
  let s = String(raw).trim().replace(/^```(json)?/i, '').replace(/```$/, '').trim();
  const a = s.indexOf('['), b = s.lastIndexOf(']');
  if (a === -1 || b === -1 || b < a) return [];
  try {
    const arr = JSON.parse(s.slice(a, b + 1));
    if (!Array.isArray(arr)) return [];
    return arr.filter(x => x && x.field && x.value && x.source_id && x.quote);
  } catch { return []; }
}

// ---------------------------------------------------------------------------
// 3. THE GATE — a fact survives ONLY if its quote is really in its source.
// ---------------------------------------------------------------------------
const norm = t => String(t || '').replace(/\s+/g, ' ').trim().toLowerCase();

function verifyFact(f, sources) {
  if (norm(f.value) === 'not found') return false;
  const src = sources[f.source_id];
  if (!src) return false;
  if (String(f.quote).trim().split(/\s+/).length < 8) return false;
  if (!norm(src.text).includes(norm(f.quote))) return false;
  return true;
}

function gate(facts, sources) {
  const kept = [], dropped = [];
  for (const f of facts) (verifyFact(f, sources) ? kept : dropped).push(f);
  return { kept, dropped };
}

// ---------------------------------------------------------------------------
// Compose verified research from surviving facts.
// ---------------------------------------------------------------------------
async function groundedResearch(env, companyName) {
  const { sources, searchHits } = await fetchSources(env, companyName);
  const sourceList = Object.values(sources).map(s => ({ url: s.url, trust: s.trust }));

  if (Object.keys(sources).length === 0)
    return { research: 'No public source could be retrieved for this company.', sources: [], dropped: 0, searchHits };

  const { kept, dropped } = gate(await extractFacts(env, sources), sources);

  if (kept.length === 0)
    return { research: 'No verifiable facts survived source-checking.', sources: sourceList, dropped: dropped.length, searchHits };

  const byField = {};
  for (const f of kept) (byField[f.field] = byField[f.field] || []).push(f);

  let research = '';
  for (const field in byField) {
    research += `\n## ${field}\n`;
    for (const f of byField[field]) research += `- ${f.value}  [${sources[f.source_id].url}]\n`;
  }
  return { research: research.trim(), sources: sourceList, dropped: dropped.length, searchHits };
}
