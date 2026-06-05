// ---------------------------------------------------------------------------
// Grounded analysis worker
//
// Flow:
//   1. Extract company name from the PDF.
//   2. Fan out ~6 targeted web searches (Serper API) about the company.
//   3. Run every snippet through the GATE: the model proposes facts WITH a
//      verbatim quote + source id; code verifies the quote is really in that
//      snippet. Anything that fails is dropped. The model is never the source.
//   4. Cache the verified research + raw search in KV (never re-search a company).
//   5. Step 2: analyze the PDF with the verified facts as context.
//   6. Step 3: condense into the final report.
//
// Search is server-side (worker), so there is no CORS wall and the key stays
// secret. The raw search is returned to the browser so the user can see and
// verify what was looked up. If SERPER_API_KEY is absent it falls back to
// Wikipedia so the app still runs.
// ---------------------------------------------------------------------------

const ALLOWED_ORIGIN = 'https://learnrahulrai-ui.github.io';

const FAST_MODEL = '@cf/meta/llama-3.1-8b-instruct';
const MAIN_MODEL = '@cf/meta/llama-3.3-70b-instruct-fp8-fast';

const SERPER_ENDPOINT = 'https://google.serper.dev/search';

// Results from these domains are tagged "primary" so they stand out.
const PRIMARY_DOMAINS = new Set([
  'sebi.gov.in', 'nseindia.com', 'bseindia.com', 'mca.gov.in',
  'indiankanoon.org', 'nclt.gov.in', 'rbi.org.in',
]);

// --- Search site controls (edit these to tune what gets searched) ---------
// Always blocked: social/profile/aggregator noise.
const EXCLUDE_SITES = [
  'facebook.com', 'linkedin.com', 'zoominfo.com', 'twitter.com', 'x.com',
  'instagram.com', 'youtube.com', 'pinterest.com', 'tracxn.com',
];
// If non-empty, every query is restricted to ONLY these sites.
// Leave empty to search the open web (minus EXCLUDE_SITES).
// Official + primary sources only — where promoter disclosures actually live.
const INCLUDE_SITES = [
  'bseindia.com', 'nseindia.com', 'nsearchives.nseindia.com',
  'sebi.gov.in', 'nclt.gov.in', 'mca.gov.in', 'indiankanoon.org',
];

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
      // Name comes from the document head (cover page), sent separately by the
      // browser — the analysis chunk skips the first 30% and would miss it.
      const headText = body.head || pdfText.slice(0, 2000);
      const companyName = await extractCompanyName(env, headText);
      const cacheKey = 'co_' + companyName.toLowerCase()
        .replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '').slice(0, 80);

      // --- Step 1: grounded research (cache-first; "fresh" bypasses the read) ---
      const skipCache = body.fresh === true;
      let research, sources, droppedCount, searchHits, via, promoters = [], fromCache = false;

      if (!skipCache && cacheKey !== 'co_' && env.COMPANY_CACHE) {
        try {
          const cached = await env.COMPANY_CACHE.get(cacheKey, 'json');
          if (cached && cached.research !== undefined) {
            ({ research, sources, droppedCount, searchHits, via } = cached);
            promoters = cached.promoters || [];
            fromCache = true;
            via = 'cache';
          }
        } catch { /* old/stale entry -> miss */ }
      }

      if (research === undefined) {
        const r = await groundedResearch(env, companyName);
        research = r.research; sources = r.sources;
        droppedCount = r.dropped; searchHits = r.searchHits; via = r.via;
        promoters = r.promoters || [];
        if (cacheKey !== 'co_' && env.COMPANY_CACHE) {
          await env.COMPANY_CACHE.put(cacheKey,
            JSON.stringify({ research, sources, droppedCount, searchHits, via, promoters }),
            { expirationTtl: 2592000 });
        }
      }

      // Diagnostics — booleans only, never the secret values themselves.
      const diag = {
        company: companyName,
        promoters,
        search_via: via,
        has_serper_key: !!env.SERPER_API_KEY,
        has_prompt_2: !!env.SYSTEM_PROMPT_2,
        has_prompt_3: !!env.SYSTEM_PROMPT_3,
      };
      console.log('DIAG', JSON.stringify(diag));

      // --- Step 2: analyze the PDF with verified facts as context ---
      const analysis = await runModel(env, MAIN_MODEL, env.SYSTEM_PROMPT_2,
        `VERIFIED COMPANY FACTS:\n${research}\n\nDOCUMENT:\n${pdfText}`, 2048);

      // --- Step 3: condense ---
      const report = await runModel(env, MAIN_MODEL, env.SYSTEM_PROMPT_3, analysis, 1024);

      return json({
        company: companyName, cached: fromCache, research, sources,
        dropped_count: droppedCount, search: searchHits || [], analysis, report,
        diag,
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

async function extractCompanyName(env, headText) {
  const out = await env.AI.run(MAIN_MODEL, {
    messages: [
      { role: 'system', content: 'You are given the opening text of a financial document (cover/first page). Return the issuing company\'s name exactly as printed, including "Limited"/"Ltd" if shown (e.g. "Asian Paints Limited"). Reply with ONLY the name — no quotes, no extra words.' },
      { role: 'user', content: headText.slice(0, 4000) },
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

// Site filters appended to every query. A whitelist already limits to trusted
// sites, so excludes are only needed when searching the open web.
function siteClause() {
  if (INCLUDE_SITES.length)
    return '(' + INCLUDE_SITES.map(d => `site:${d}`).join(' OR ') + ')';
  return EXCLUDE_SITES.map(d => `-site:${d}`).join(' ');
}
const applyFilter = base => `${base} ${siteClause()}`.trim();

// PHASE 1 — company-level: establish who the promoters are + base disclosures.
function companyQueries(name) {
  const q = `"${name}"`;
  return [
    `${q} promoters directors annual report`,
    `${q} SAST shareholding pattern disclosure`,
  ].map(applyFilter);
}

// PHASE 2a — promoter-level: the tricks are filed under PEOPLE's names.
function promoterQueries(company, promoters) {
  const c = `"${company}"`;
  const qs = [];
  for (const p of promoters) {
    qs.push(`"${p}" ${c} insider trading off market sale acquisition disposal`);
    qs.push(`"${p}" ${c} pledge warrants preferential allotment`);
  }
  return qs.map(applyFilter);
}

// PHASE 2b — company-level, PDF-ONLY disclosure hunts. These are the documents
// that record the actual moves: bulk/block/inter-se/off-market transfers,
// repeated OFS offloading, and the pre-IPO PE/VC entry + promoter offer-for-sale
// buried in the DRHP/RHP. (Snippets find the PDF; deep facts may sit inside it.)
function disclosureQueries(company) {
  const c = `"${company}"`;
  return [
    `${c} (bulk deal OR block deal OR "inter-se transfer" OR "off market" OR SAST) filetype:pdf`,
    `${c} (OFS OR "offer for sale") (promoter OR "selling shareholder") filetype:pdf`,
    `${c} (DRHP OR RHP OR prospectus) ("private equity" OR "venture capital" OR "selling shareholders" OR "pre-IPO" OR "offer for sale") filetype:pdf`,
    `${c} (QIP OR FPO OR warrants OR "preferential allotment") filetype:pdf`,
  ].map(applyFilter);
}

async function serperSearch(query, key) {
  const res = await fetch(SERPER_ENDPOINT, {
    method: 'POST',
    headers: { 'X-API-KEY': key, 'Content-Type': 'application/json' },
    body: JSON.stringify({ q: query, num: 5 }),
  });
  if (!res.ok) return [];
  const data = await res.json();
  const results = data?.organic ?? [];
  return results.map(r => ({
    url: r.link, title: r.title || '', text: r.snippet || '',
  })).filter(r => r.url && r.text && r.text.length > 30);
}

// Pull up to 3 promoter/director person-names out of phase-1 snippets, so
// phase 2 can search by person (where off-market sales / insider trades live).
async function extractPromoterNames(env, sources) {
  const ids = Object.keys(sources);
  if (!ids.length) return [];
  const blob = ids.map(id => sources[id].text).join('\n').slice(0, 4000);
  const out = await env.AI.run(FAST_MODEL, {
    messages: [
      { role: 'system', content: 'From the text, list up to 3 individual people who are promoters, directors, or managing directors of the company. Return ONLY a JSON array of full names, e.g. ["Anil Aggarwal","Atul Aggarwal"]. If none are named, return [].' },
      { role: 'user', content: blob },
    ],
    max_tokens: 100,
  });
  try {
    const s = out.response ?? '';
    const a = s.indexOf('['), b = s.lastIndexOf(']');
    const arr = JSON.parse(s.slice(a, b + 1));
    return Array.isArray(arr) ? arr.filter(x => typeof x === 'string' && x.trim()).slice(0, 3) : [];
  } catch { return []; }
}

async function fetchSourcesSerper(env, companyName, key) {
  const sources = {};
  const searchHits = [];
  const runAll = async (queries) => {
    const perQuery = await Promise.all(queries.map(async q => {
      try { return { q, hits: await serperSearch(q, key) }; }
      catch { return { q, hits: [] }; }
    }));
    for (const { q, hits } of perQuery) {
      for (const h of hits) {
        const id = 's' + Object.keys(sources).length;
        sources[id] = { id, url: h.url, trust: trustOf(h.url), text: h.text };
        searchHits.push({ query: q, title: h.title, url: h.url, snippet: h.text });
      }
    }
  };

  // Phase 1: company-level -> find promoter names.
  await runAll(companyQueries(companyName));
  const promoters = await extractPromoterNames(env, sources);
  console.log('PROMOTERS:', JSON.stringify(promoters));

  // Phase 2: promoter-level searches + company-level PDF-only disclosure hunts.
  await runAll([
    ...promoterQueries(companyName, promoters),
    ...disclosureQueries(companyName),
  ]);

  return { sources, searchHits, promoters };
}

// Fallback when no Serper key is set.
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
  if (env.SERPER_API_KEY) {
    const { sources, searchHits, promoters } = await fetchSourcesSerper(env, companyName, env.SERPER_API_KEY);
    // Key is set: trust Serper. If it returned nothing, the key is likely
    // invalid or out of quota — say so rather than masking it with Wikipedia.
    const via = Object.keys(sources).length > 0 ? 'serper' : 'serper-empty';
    console.log('SERPER hits:', searchHits.length, 'via:', via);
    return { sources, searchHits, via, promoters };
  }
  // No key configured -> Wikipedia keeps the app usable.
  const sources = await fetchSourcesWikipedia(companyName);
  return { sources, searchHits: [], via: 'no-key', promoters: [] };
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
- ownership_changes (bulk deal, block deal, inter-se transfer, off-market sale,
  OFS / offer-for-sale, promoter offload; one row each, include who and how much if stated)
- pre_ipo_moves (private equity / venture capital entry or exit around the IPO,
  selling shareholders in the IPO offer-for-sale, pre-IPO stake changes; one row each)

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

// Match-normalization: drop punctuation/ellipses so a snippet quote can match
// even if the model didn't copy "..." and commas character-for-character.
const matchNorm = t => String(t || '')
  .toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim();

function verifyFact(f, sources) {
  if (norm(f.value) === 'not found') return false;
  const src = sources[f.source_id];
  if (!src) return false;
  const q = matchNorm(f.quote);
  if (q.split(' ').filter(Boolean).length < 6) return false;   // real anchor
  if (!matchNorm(src.text).includes(q)) return false;          // invented/altered
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
  const { sources, searchHits, via, promoters } = await fetchSources(env, companyName);
  const sourceList = Object.values(sources).map(s => ({ url: s.url, trust: s.trust }));

  if (Object.keys(sources).length === 0)
    return { research: 'No public source could be retrieved for this company.', sources: [], dropped: 0, searchHits, via, promoters };

  const { kept, dropped } = gate(await extractFacts(env, sources), sources);
  console.log('GATE kept:', kept.length, 'dropped:', dropped.length);

  if (kept.length === 0)
    return { research: 'No verifiable facts survived source-checking.', sources: sourceList, dropped: dropped.length, searchHits, via, promoters };

  const byField = {};
  for (const f of kept) (byField[f.field] = byField[f.field] || []).push(f);

  let research = '';
  for (const field in byField) {
    research += `\n## ${field}\n`;
    for (const f of byField[field]) research += `- ${f.value}  [${sources[f.source_id].url}]\n`;
  }
  return { research: research.trim(), sources: sourceList, dropped: dropped.length, searchHits, via, promoters };
}
