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

// Cloudflare Workers AI models.
// FAST_MODEL (8B) is used for all extraction tasks to stay within the free
// 10,000 neuron/day limit.  MAIN_MODEL (70B) would burn ~8,000 neurons on a
// single analysis call — the whole daily budget for one request.
const FAST_MODEL = '@cf/meta/llama-3.1-8b-instruct';
const MAIN_MODEL = '@cf/meta/llama-3.3-70b-instruct-fp8-fast';  // only used when CF AI is the sole option

// Groq free tier: llama-3.3-70b-versatile, 100k tokens/day, no credit card.
// Only used as a fallback if the user did NOT bring a Gemini key.
const GROQ_ENDPOINT = 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_MODEL = 'llama-3.3-70b-versatile';

// PRIMARY path: the USER brings their own Google Gemini key. Every model call
// then runs on their key and their quota — so this worker never burns its free
// Cloudflare neuron budget and never has to pay for anyone's analysis. The key
// arrives per-request, is used once, and is NEVER logged, cached, or stored.
const GEMINI_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models';
// Each model has its OWN free-tier quota bucket. We use Gemini 3.5 Flash — the
// latest GA Flash model (May 2026), free tier, best reasoning of the free Flash
// options. If a key ever shows "limit: 0"/no access, the doc-confirmed free
// fallback is 'gemini-3-flash-preview'. (gemini-3.1-flash-lite = highest RPD.)
const GEMINI_MODEL = 'gemini-3.5-flash';

const SERPER_ENDPOINT = 'https://google.serper.dev/search';

// Results from these domains are tagged "primary" so they stand out.
const PRIMARY_DOMAINS = new Set([
  'sebi.gov.in', 'nseindia.com', 'bseindia.com', 'mca.gov.in',
  'indiankanoon.org', 'nclt.gov.in', 'rbi.org.in',
]);

// --- Search site whitelist (edit to tune what gets searched) --------------
// General searches (company + promoter names): exchanges, regulator, the
// established Indian market sites, court records, and old-but-gold Rediff.
const GENERAL_SITES = [
  'bseindia.com', 'nseindia.com', 'nsearchives.nseindia.com', 'sebi.gov.in',
  'screener.in', 'trendlyne.com', 'chittorgarh.com', 'rediff.com',
  'indiankanoon.org',
];
// PDF-only disclosure hunts go where the filings actually live.
const PDF_SITES = [
  'bseindia.com', 'nseindia.com', 'nsearchives.nseindia.com',
  'sebi.gov.in', 'chittorgarh.com',
];
// Credit / debt sources (rating agencies) — for default / pledge / downgrade.
// (My read of "banks sites" — tell me if you meant literal bank websites.)
const CREDIT_SITES = ['icra.in', 'crisil.com', 'careratings.com'];
// Cloud storage / CDN where firms park annual reports, transcripts and decks
// for years — forgotten but still public and indexed by Google.
const BUCKET_SITES = [
  'amazonaws.com', 'blob.core.windows.net', 'storage.googleapis.com', 'cloudfront.net',
];
// Junk excluded from the one OPEN-web query (the company's own IR page, domain
// unknown). Social + "fancy" aggregator/AI sites you don't want. (Kept to ~14
// so the query stays under Google's term limit.)
const EXCLUDE_SITES = [
  'facebook.com', 'linkedin.com', 'instagram.com', 'twitter.com', 'x.com', 'youtube.com',
  'marketscreener.com', 'stockinsights.ai', 'simplywall.st', 'investing.com',
  'tradingview.com', 'marketbeat.com', 'stockanalysis.com', 'zoominfo.com',
];
// LAYERING: map a promoter's NAME -> every entity they run. zaubacorp/tofler
// list a director's companies; indiankanoon catches the name in litigation.
const LAYERING_SITES = ['zaubacorp.com', 'tofler.in', 'indiankanoon.org'];
// Broker/securities domains that also host PDFs. NOTE: mostly research/ratings
// (which you don't want), so only queried with disclosure/AR terms.
const BROKER_SITES = ['hdfcsec.com', 'icicidirect.com', 'motilaloswal.com'];

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

    // The user brings their own Gemini key. We use it once, in-memory, for this
    // request only. It is NEVER written to a log line, the KV cache, or the
    // response. (Search the file: the key never appears in any console.log.)
    const geminiKey = (body.geminiKey || '').trim();
    if (!geminiKey) {
      return json({ error: 'Please paste your Google Gemini API key. Get a free one (no card) at https://aistudio.google.com/apikey' }, 400);
    }

    try {
      const pdfText = body.text;
      // Name comes from the document head (cover page), sent separately by the
      // browser — the analysis chunk skips the first 30% and would miss it.
      const headText = body.head || pdfText.slice(0, 2000);

      // Gate 1: is this even a financial document? Cheap one-word classify on
      // the user's key. Saves their quota on the heavy steps if it's the wrong
      // file, and tells them plainly.
      const looksFinancial = await isFinancialPdf(env, geminiKey, headText + '\n' + pdfText.slice(0, 3000));
      if (!looksFinancial) {
        return json({
          financial: false,
          message: 'This does not look like a financial document. Please upload a company financial PDF — an annual report, quarterly results, prospectus/DRHP, shareholding disclosure, investor presentation, or an exchange filing.',
        });
      }

      const companyName = await extractCompanyName(env, geminiKey, headText);
      const cacheKey = 'co_' + companyName.toLowerCase()
        .replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '').slice(0, 80);

      // --- Step 1: grounded research (cache-first; "fresh" bypasses the read) ---
      const skipCache = body.fresh === true;
      let research, sources, droppedCount, dropReasons, enriched, searchHits, via, promoters = [], independents = [], fromCache = false;

      if (!skipCache && cacheKey !== 'co_' && env.COMPANY_CACHE) {
        try {
          const cached = await env.COMPANY_CACHE.get(cacheKey, 'json');
          if (cached && cached.research !== undefined) {
            ({ research, sources, droppedCount, dropReasons, enriched, searchHits, via } = cached);
            promoters = cached.promoters || [];
            independents = cached.independents || [];
            fromCache = true;
            via = 'cache';
          }
        } catch { /* old/stale entry -> miss */ }
      }

      if (research === undefined) {
        const r = await groundedResearch(env, geminiKey, companyName, pdfText);
        research = r.research; sources = r.sources;
        droppedCount = r.dropped; dropReasons = r.dropReasons; enriched = r.enriched; searchHits = r.searchHits; via = r.via;
        promoters = r.promoters || [];
        independents = r.independents || [];
        if (cacheKey !== 'co_' && env.COMPANY_CACHE) {
          await env.COMPANY_CACHE.put(cacheKey,
            JSON.stringify({ research, sources, droppedCount, dropReasons, enriched, searchHits, via, promoters, independents }),
            { expirationTtl: 2592000 });
        }
      }

      // Diagnostics — booleans only, never the secret values themselves.
      const diag = {
        company: companyName,
        promoters,
        independents,
        search_via: via,
        full_text_fetched: enriched || 0,   // how many sources Jina read in full
        drop_reasons: dropReasons || {},
        has_user_key: !!geminiKey,      // boolean only — never the key value
        has_serper_key: !!env.SERPER_API_KEY,
        has_jina_key: !!env.JINA_API_KEY,
        has_prompt_2: !!env.SYSTEM_PROMPT_2,
        has_prompt_3: !!env.SYSTEM_PROMPT_3,
      };
      console.log('DIAG', JSON.stringify(diag));

      // --- Step 2: the ONE grounded call. The model reasons over the PDF +
      // verified facts AND searches the live web (Google Search grounding),
      // all directed by the secret SYSTEM_PROMPT_2. Falls back to a plain call
      // if grounding isn't available on the user's key/quota. ---
      const analysisInput = `VERIFIED COMPANY FACTS:\n${research}\n\nDOCUMENT:\n${pdfText}`;
      let analysis;
      try {
        analysis = await runGemini(geminiKey, env.SYSTEM_PROMPT_2, analysisInput, 2048, { grounded: true, think: 'medium' });
      } catch {
        analysis = await aiText(env, geminiKey, env.SYSTEM_PROMPT_2, analysisInput, 2048);
      }

      // --- Step 3: condense into the final report ---
      const report = await aiText(env, geminiKey, env.SYSTEM_PROMPT_3, analysis, 1024);

      // Strip the raw query strings — never expose the trick queries to the
      // browser. Only the found documents (title/url/snippet) go to the user.
      const publicSearch = (searchHits || []).map(h => ({
        title: h.title, url: h.url, snippet: h.snippet,
      }));

      return json({
        company: companyName, cached: fromCache, research, sources,
        dropped_count: droppedCount, search: publicSearch, analysis, report,
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

// Groq path: 70B quality, generous free tier (100k tokens/day).
// Caps user content at 18,000 chars to stay under the 6,000-token/min rate
// limit (≈4,500 tokens input + system + output ≈ 6,000 total).
async function runGroq(key, system, user, maxTokens) {
  const res = await fetch(GROQ_ENDPOINT, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages: [
        { role: 'system', content: system || '' },
        { role: 'user', content: user.slice(0, 18000) },
      ],
      max_tokens: maxTokens,
    }),
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    throw new Error('Groq: ' + (e.error?.message || res.statusText));
  }
  const d = await res.json();
  return (d.choices?.[0]?.message?.content ?? '').trim();
}

// Cloudflare Workers AI path (cheap 8B).  Used for extraction tasks always,
// and as fallback for analysis/report when no Groq key is set.
async function runCF(env, model, system, user, maxTokens) {
  const out = await env.AI.run(model, {
    messages: [
      { role: 'system', content: system || '' },
      { role: 'user', content: user },
    ],
    max_tokens: maxTokens,
  });
  return (out.response ?? '').trim();
}

// Gemini path: the user's own key. Sent via the x-goog-api-key header (not the
// URL) so the key never lands in any request line that might be logged upstream.
async function runGemini(key, system, user, maxTokens, opts = {}) {
  const reqBody = {
    system_instruction: { parts: [{ text: system || '' }] },
    contents: [{ role: 'user', parts: [{ text: user.slice(0, 24000) }] }],
    generationConfig: {
      maxOutputTokens: maxTokens,
      temperature: 0.2,
      // Gemini 3 replaced 2.5's numeric thinkingBudget with thinking_level
      // (minimal | low | medium | high). On 2.5 the hidden reasoning tokens were
      // drawn from maxOutputTokens, so a tiny budget (the YES/NO gate) came back
      // EMPTY and wrongly rejected real filings. Default 'low' = fast + cheap,
      // right for the mechanical calls; the forensic analysis passes 'medium'.
      thinking_level: opts.think || 'low',
    },
  };
  // Google Search grounding: the model searches the live web (filings, news,
  // exchange disclosures) while it answers and returns citations. WHAT it hunts
  // for is driven entirely by the secret system prompt.
  if (opts.grounded) reqBody.tools = [{ google_search: {} }];
  const res = await fetch(`${GEMINI_ENDPOINT}/${GEMINI_MODEL}:generateContent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
    body: JSON.stringify(reqBody),
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    // Surface key/quota errors to the user — but never echo the key.
    throw new Error('Gemini API: ' + (e.error?.message || res.statusText));
  }
  const d = await res.json();
  const parts = d.candidates?.[0]?.content?.parts || [];
  return parts.map(p => p.text || '').join('').trim();
}

// Unified text-generation call. With a user Gemini key (the normal path) every
// model step runs on the user's key/quota. Groq/CF remain only as a fallback so
// the owner could still run it key-less for testing.
async function aiText(env, geminiKey, system, user, maxTokens, opts = {}) {
  if (geminiKey) return runGemini(geminiKey, system, user, maxTokens, opts);
  if (env.GROQ_API_KEY) return runGroq(env.GROQ_API_KEY, system, user, maxTokens);
  return runCF(env, FAST_MODEL, system, user, maxTokens);
}

// Gate: is this a financial / corporate-disclosure document at all?
// Fails OPEN — if the classifier returns nothing usable we let the upload
// through rather than wrongly blocking a real filing. Only an explicit NO stops it.
async function isFinancialPdf(env, geminiKey, sample) {
  const out = await aiText(env, geminiKey,
    'You are a strict classifier. Decide if the text is from a FINANCIAL or CORPORATE-DISCLOSURE document of a company — e.g. annual report, quarterly/financial results, balance sheet or profit-and-loss, prospectus or DRHP/RHP, shareholding pattern, investor presentation, earnings-call transcript, credit-rating rationale, or a stock-exchange filing. Reply with ONLY one word: YES or NO.',
    sample.slice(0, 4000), 8, { think: 'minimal' });
  const v = (out || '').trim();
  if (/\bno\b/i.test(v) && !/\byes\b/i.test(v)) return false; // explicit reject
  return true; // YES, or empty/ambiguous -> let it through
}

async function extractCompanyName(env, geminiKey, headText) {
  const out = await aiText(env, geminiKey,
    'You are given the opening text of a financial document (cover/first page). Return the issuing company\'s name exactly as printed, including "Limited"/"Ltd" if shown (e.g. "Asian Paints Limited"). Reply with ONLY the name — no quotes, no extra words.',
    headText.slice(0, 1500), 30, { think: 'minimal' });
  return out.trim().replace(/^["']+|["']+$/g, '');
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

// Restrict a query to a given set of sites.
const clause = sites => '(' + sites.map(d => `site:${d}`).join(' OR ') + ')';
const applyFilter = base => `${base} ${clause(GENERAL_SITES)}`.trim();
const applyPdf    = base => `${base} ${clause(PDF_SITES)}`.trim();
const applyCredit = base => `${base} ${clause(CREDIT_SITES)}`.trim();
const applyBucket = base => `${base} ${clause(BUCKET_SITES)}`.trim();
const applyLayer  = base => `${base} ${clause(LAYERING_SITES)}`.trim();
const applyBroker = base => `${base} ${clause(BROKER_SITES)}`.trim();
const applyOpen   = base => `${base} ${EXCLUDE_SITES.map(d => `-site:${d}`).join(' ')}`.trim();

// PHASE 1 — company-level: establish who the promoters + independent directors
// are, plus base disclosures.
function companyQueries(name) {
  const q = `"${name}"`;
  return [
    `${q} promoters directors annual report`,
    `${q} SAST shareholding pattern disclosure`,
    `${q} "independent director" board composition corporate governance`,
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
  const pdf = [
    `${c} (bulk deal OR block deal OR "inter-se transfer" OR "off market" OR SAST) filetype:pdf`,
    `${c} (OFS OR "offer for sale") (promoter OR "selling shareholder") filetype:pdf`,
    `${c} (DRHP OR RHP OR prospectus) ("private equity" OR "venture capital" OR "selling shareholders" OR "pre-IPO") filetype:pdf`,
    `${c} (QIP OR FPO OR "preferential allotment" OR "rights issue" OR "rights entitlement") filetype:pdf`,
    `${c} (warrants OR "convertible warrants" OR "warrant conversion" OR "subscription to warrants" OR forfeiture) (promoter OR allottee) filetype:pdf`,
    `${c} ("step down subsidiary" OR "wholly owned subsidiary" OR "subscription to equity" OR "investment in subsidiary" OR "infused") filetype:pdf`,
    `${c} ("unsecured loan" OR "inter-corporate deposit" OR "loan to subsidiary" OR "written off" OR "Section 186") (subsidiary OR "related party") filetype:pdf`,
    `${c} subsidiary (equity OR "rights issue" OR preferential OR "fund raising" OR "capital raise") filetype:pdf`,
    `${c} ("overseas subsidiary" OR "foreign subsidiary" OR "incorporated outside India" OR "overseas acquisition") filetype:pdf`,
    `${c} (intimation OR announcement OR notice OR "regulation 30" OR disclosure) filetype:pdf`,
    `${c} (pledge OR pledged OR encumbrance OR "Regulation 31" OR "invocation of pledge" OR "release of pledge") filetype:pdf`,
  ].map(applyPdf);
  // Debt / pledge / default via the rating agencies ("banks/credit" sources).
  const credit = [
    `${c} (default OR pledge OR "wilful defaulter" OR downgrade OR insolvency)`,
  ].map(applyCredit);
  return [...pdf, ...credit];
}

// PHASE 2c — forgotten cloud buckets (S3/Azure/GCP/CloudFront): annual reports,
// transcripts and decks in PDF, plus shareholding/deal data in Excel.
function bucketQueries(company) {
  const c = `"${company}"`;
  return [
    `${c} (annual report OR transcript OR concall OR "investor presentation") filetype:pdf`,
    `${c} (shareholding OR "bulk deal" OR financials) (filetype:xlsx OR filetype:xls)`,
  ].map(applyBucket);
}

// PHASE 2d — the company's OWN investor-relations docs, wherever hosted
// (open web; domain is unknown up front, so only junk is excluded).
function irQueries(company) {
  const c = `"${company}"`;
  return [
    `${c} (investor relations OR "earnings call transcript" OR concall OR "investor presentation") filetype:pdf`,
  ].map(applyOpen);
}

// PHASE 2e — LAYERING: track each promoter NAME across the other entities they
// run (zaubacorp/tofler director listings), plus the company's own shell/
// related-party disclosures. First hop only — see the deep-layering note.
function layeringQueries(company, promoters) {
  const qs = promoters.map(p => applyLayer(`"${p}" director`));
  qs.push(applyPdf(`"${company}" (shell OR "related party" OR associate OR "inter-se") filetype:pdf`));
  return qs;
}

// PHASE 2g — INDEPENDENT DIRECTORS: every other board they sit on (capture risk)
// and the fees/commission they collect — across the company's group and beyond.
function independentQueries(company, independents) {
  const qs = [];
  for (const d of independents) {
    qs.push(applyLayer(`"${d}" director`));                                         // all their boards
    qs.push(applyPdf(`"${d}" ("sitting fees" OR commission OR remuneration) filetype:pdf`)); // fees across cos
  }
  qs.push(applyPdf(`"${company}" ("independent director" OR "sitting fees" OR "commission to directors" OR "remuneration to directors") filetype:pdf`));
  return qs;
}

// PHASE 2f — broker/securities domains that host PDFs (disclosure/AR only,
// to avoid their research-report noise).
function brokerQueries(company) {
  return [
    applyBroker(`"${company}" (annual report OR transcript OR shareholding OR disclosure) filetype:pdf`),
  ];
}

// ---------------------------------------------------------------------------
// Edge cache (Cloudflare Cache API, caches.default) — free, colo-local, no
// quota and no write limit. We cache idempotent upstream reads: Serper results
// and Jina full-text. Best-effort: if the platform ever no-ops it the code
// still works, just slower. The synthetic key URL is never fetched and never
// leaves the worker, so the secret trick-queries used as keys stay secret.
// ---------------------------------------------------------------------------
const EDGE = 'https://fore.cache/';
async function edgeGet(key) {
  try {
    const r = await caches.default.match(new Request(EDGE + encodeURIComponent(key)));
    return r ? await r.text() : null;
  } catch { return null; }
}
async function edgePut(key, text, ttl = 86400) {
  try {
    await caches.default.put(
      new Request(EDGE + encodeURIComponent(key)),
      new Response(text, { headers: { 'Cache-Control': 'max-age=' + ttl } }),
    );
  } catch { /* best-effort */ }
}

async function serperSearch(query, key) {
  const ck = 'serper:' + query;
  const hit = await edgeGet(ck);
  if (hit) { try { return JSON.parse(hit); } catch { /* stale -> refetch */ } }
  const res = await fetch(SERPER_ENDPOINT, {
    method: 'POST',
    headers: { 'X-API-KEY': key, 'Content-Type': 'application/json' },
    body: JSON.stringify({ q: query, num: 5 }),
  });
  if (!res.ok) return [];
  const data = await res.json();
  const results = data?.organic ?? [];
  const mapped = results.map(r => ({
    url: r.link, title: r.title || '', text: r.snippet || '',
  })).filter(r => r.url && r.text && r.text.length > 30);
  if (mapped.length) await edgePut(ck, JSON.stringify(mapped)); // real hits only, ~1 day
  return mapped;
}

// Authoritative board roster comes from the UPLOADED filing, not noisy web
// snippets — the document literally lists its own directors + shareholdings.
// Excludes the company secretary, auditors and ceremonial/ministerial mentions.
async function extractBoardFromPdf(env, geminiKey, pdfText) {
  if (!pdfText) return { promoters: [], independents: [] };
  const s = await aiText(env, geminiKey,
    'From this company filing, list the people on its BOARD OF DIRECTORS. Return ONLY JSON: {"promoters":["..."],"independents":["..."]}. promoters = promoters / executive / managing directors (max 4). independents = independent or non-executive directors (max 4). Full personal names only. Do NOT include the Company Secretary, Compliance Officer, auditors, or any government minister mentioned ceremonially. Unknown list = [].',
    pdfText.slice(0, 6000), 200);
  const clean = a => Array.isArray(a) ? a.filter(x => typeof x === 'string' && x.trim()).slice(0, 4) : [];
  try {
    const obj = JSON.parse(s.slice(s.indexOf('{'), s.lastIndexOf('}') + 1));
    return { promoters: clean(obj.promoters), independents: clean(obj.independents) };
  } catch { return { promoters: [], independents: [] }; }
}

// Pull key people out of phase-1 snippets so phase 2 can search by person:
// promoters (where insider/off-market trades live) and independent directors
// (whose other boards + fees reveal capture).
async function extractKeyPeople(env, geminiKey, sources) {
  const ids = Object.keys(sources);
  if (!ids.length) return { promoters: [], independents: [] };
  const blob = ids.map(id => sources[id].text).join('\n').slice(0, 3000);
  const s = await aiText(env, geminiKey,
    'From the text, identify this company\'s board members. Return ONLY JSON: {"promoters":["..."],"independents":["..."]}. promoters = promoters / managing / executive directors (max 3). independents = independent or non-executive directors (max 3). Full names only. Unknown list = [].',
    blob, 200);
  const clean = a => Array.isArray(a) ? a.filter(x => typeof x === 'string' && x.trim()).slice(0, 3) : [];
  try {
    const obj = JSON.parse(s.slice(s.indexOf('{'), s.lastIndexOf('}') + 1));
    return { promoters: clean(obj.promoters), independents: clean(obj.independents) };
  } catch { return { promoters: [], independents: [] }; }
}

async function fetchSourcesSerper(env, geminiKey, companyName, key, board) {
  const sources = {};
  const searchHits = [];
  const seen = new Set();   // dedupe identical URLs across the ~26 queries
  const runAll = async (queries) => {
    const perQuery = await Promise.all(queries.map(async q => {
      try { return { q, hits: await serperSearch(q, key) }; }
      catch { return { q, hits: [] }; }
    }));
    for (const { q, hits } of perQuery) {
      for (const h of hits) {
        if (seen.has(h.url)) continue;       // first query to find a URL wins
        seen.add(h.url);
        const id = 's' + seen.size;
        sources[id] = { id, url: h.url, trust: trustOf(h.url), text: h.text };
        searchHits.push({ query: q, title: h.title, url: h.url, snippet: h.text });
      }
    }
  };

  // Phase 1: company-level disclosures.
  await runAll(companyQueries(companyName));

  // Names: prefer the board parsed from the uploaded filing; only fall back to
  // noisy web-snippet extraction if the document gave us nothing.
  let promoters = (board && board.promoters) || [];
  let independents = (board && board.independents) || [];
  if (!promoters.length && !independents.length) {
    ({ promoters, independents } = await extractKeyPeople(env, geminiKey, sources));
  }
  console.log('PEOPLE:', JSON.stringify({ promoters, independents }));

  // Phase 2: promoters + independents + disclosures + buckets + IR + layering.
  await runAll([
    ...promoterQueries(companyName, promoters),
    ...independentQueries(companyName, independents),
    ...disclosureQueries(companyName),
    ...bucketQueries(companyName),
    ...irQueries(companyName),
    ...layeringQueries(companyName, promoters),
    ...brokerQueries(companyName),
  ]);

  return { sources, searchHits, promoters, independents };
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

async function fetchSources(env, geminiKey, companyName, board) {
  if (env.SERPER_API_KEY) {
    const { sources, searchHits, promoters, independents } = await fetchSourcesSerper(env, geminiKey, companyName, env.SERPER_API_KEY, board);
    // Key is set: trust Serper. If it returned nothing, the key is likely
    // invalid or out of quota — say so rather than masking it with Wikipedia.
    const via = Object.keys(sources).length > 0 ? 'serper' : 'serper-empty';
    console.log('SERPER hits:', searchHits.length, 'via:', via);
    return { sources, searchHits, via, promoters, independents };
  }
  // No key configured -> Wikipedia keeps the app usable.
  const sources = await fetchSourcesWikipedia(companyName);
  return { sources, searchHits: [], via: 'no-key', promoters: [], independents: [] };
}

// ---------------------------------------------------------------------------
// 1b. FULL-TEXT ENRICHMENT — Serper returns one-line snippets, but the real
// numbers (pledge %, off-market size, related-party loan amount, who got the
// warrants) live INSIDE the linked filing. Jina Reader (r.jina.ai) turns any
// page OR pdf into plain text — free, keyless — so the extractor has real
// documents to quote from instead of a single sentence. Does NOT touch the
// user's Gemini quota.
// ---------------------------------------------------------------------------
async function jinaFetch(url, env) {
  const ck = 'jina:' + url;
  const hit = await edgeGet(ck);
  if (hit) return hit;                 // filing text is static — instant on repeat
  try {
    const headers = { 'Accept': 'text/plain', 'X-Return-Format': 'text' };
    if (env.JINA_API_KEY) headers['Authorization'] = `Bearer ${env.JINA_API_KEY}`; // higher limits if set
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 15000);
    const res = await fetch('https://r.jina.ai/' + url, { headers, signal: ctrl.signal });
    clearTimeout(timer);
    if (!res.ok) return '';
    const txt = await res.text();
    const clean = txt.replace(/\s+/g, ' ').trim();
    if (clean) await edgePut(ck, clean, 604800);   // filings don't change — 7 days
    return clean;
  } catch { return ''; }
}

// Rank by how likely a URL is to be a real filing, then read the top few.
function enrichScore(s) {
  let n = 0;
  if (/\.pdf(\?|$)/i.test(s.url)) n += 3;   // a filing PDF — the gold
  if (s.trust === 'primary')     n += 2;    // sebi/nse/bse/mca/...
  if (s.trust === 'reference')   n += 1;
  return n;
}

async function enrichWithFullText(env, sources, limit = 6) {
  const picks = Object.values(sources)
    .sort((a, b) => enrichScore(b) - enrichScore(a))
    .slice(0, limit);
  let enriched = 0;
  await Promise.all(picks.map(async s => {
    const full = await jinaFetch(s.url, env);
    if (full && full.length > s.text.length) {
      s.text = full.slice(0, 4000);   // cap so one PDF can't dominate context
      s.full = true;
      enriched++;
    }
  }));
  return enriched;
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
- capital_events (QIP, FPO, preferential allotment, rights issue, dilution; one row each)
- subsidiary_funding (parent subscribing to / investing in / infusing money into a
  subsidiary or step-down subsidiary — especially one tied to promoter family
  (sons/daughters/relatives); one row each, include the entity and amount)
- loans_to_related (unsecured loans / inter-corporate deposits / advances from the
  company to a subsidiary or related party, and any later default / write-off /
  provision for doubtful; one row each, include amount, entity, and whether written off)
- warrant_events (preferential warrants to promoters/allottees, conversion price
  vs market price, warrant conversion or forfeiture, % upfront; one row each)
- distress_signals (debt default, fraud/investigation, downgrade, insolvency; one row each)
- pledge_events (promoter shares pledged/encumbered, pledge creation/release/
  invocation, % of holding pledged; one row each, include who and how much if stated)
- ownership_changes (bulk deal, block deal, inter-se transfer, off-market sale,
  OFS / offer-for-sale, promoter offload; one row each, include who and how much if stated)
- pre_ipo_moves (private equity / venture capital entry or exit around the IPO,
  selling shareholders in the IPO offer-for-sale, pre-IPO stake changes; one row each)
- layering_links (a named promoter/director linked to ANOTHER company or entity:
  other directorships, associate/holding/shell companies, related parties;
  one row each, format "<person> -> <other entity> (<relationship>)")
- board_overlaps (an independent or non-executive director and the OTHER boards
  they sit on — including related companies / subsidiaries — and any sitting fees /
  commission / remuneration they collect; one row each)
- overseas_subsidiaries (foreign/overseas subsidiary incorporated or acquired, an
  overseas acquisition — especially of a loss-making/near-bankrupt target — and who
  leads it (promoter family placed abroad); one row each, include entity and country)

SOURCES:
`;

async function extractFacts(env, geminiKey, sources) {
  const all = Object.values(sources);
  if (all.length === 0) return [];
  // Full-text (Jina-enriched) sources first — they hold the quotable detail, so
  // give them a big slice; snippets get a small one. Cap total for context.
  all.sort((a, b) => (b.full ? 1 : 0) - (a.full ? 1 : 0));
  const blob = all.slice(0, 30)
    .map(s => `[${s.id}] ${s.url}\n${s.text.slice(0, s.full ? 2000 : 500)}`)
    .join('\n\n');
  const out = await aiText(env, geminiKey,
    'You extract only facts present verbatim in the provided sources. You never invent.',
    EXTRACT_PROMPT + blob, 2000);
  return parseFacts(out);
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

// Sliding-window match: any 5-consecutive-word run from the quote must appear
// in the source.  This catches cases where the model copied words correctly but
// inserted/dropped a punctuation mark or one connecting word.  Pure
// hallucinations still fail: they have no 5-word run shared with the snippet.
const SLIDE_WIN = 5;
function slideMatch(normQuote, normSrc) {
  const words = normQuote.split(' ').filter(Boolean);
  if (words.length < SLIDE_WIN) return normSrc.includes(normQuote);
  for (let i = 0; i <= words.length - SLIDE_WIN; i++) {
    if (normSrc.includes(words.slice(i, i + SLIDE_WIN).join(' '))) return true;
  }
  return false;
}

// Returns { ok: bool, reason: string } so callers can count why facts die.
function verifyFact(f, sources) {
  if (norm(f.value) === 'not found') return { ok: false, reason: 'not_found_value' };
  const src = sources[f.source_id];
  if (!src) return { ok: false, reason: 'no_source' };
  const q = matchNorm(f.quote);
  if (q.split(' ').filter(Boolean).length < 4) return { ok: false, reason: 'too_short' };
  if (!slideMatch(q, matchNorm(src.text))) return { ok: false, reason: 'quote_not_found' };
  return { ok: true, reason: 'ok' };
}

function gate(facts, sources) {
  const kept = [], dropped = [];
  const reasons = {};
  for (const f of facts) {
    const { ok, reason } = verifyFact(f, sources);
    if (ok) {
      kept.push(f);
    } else {
      dropped.push(f);
      reasons[reason] = (reasons[reason] || 0) + 1;
    }
  }
  return { kept, dropped, reasons };
}

// ---------------------------------------------------------------------------
// Compose verified research from surviving facts.
// ---------------------------------------------------------------------------
async function groundedResearch(env, geminiKey, companyName, pdfText) {
  // Authoritative board comes from the uploaded filing, not noisy web snippets.
  const board = await extractBoardFromPdf(env, geminiKey, pdfText);
  const { sources, searchHits, via, promoters, independents } = await fetchSources(env, geminiKey, companyName, board);
  const sourceList = Object.values(sources).map(s => ({ url: s.url, trust: s.trust }));

  if (Object.keys(sources).length === 0)
    return { research: 'No public source could be retrieved for this company.', sources: [], dropped: 0, enriched: 0, searchHits, via, promoters, independents };

  // Read the top filings in full (Jina) so the extractor can quote real numbers.
  const enriched = await enrichWithFullText(env, sources);
  console.log('JINA enriched:', enriched, 'of', Object.keys(sources).length);

  const { kept, dropped, reasons } = gate(await extractFacts(env, geminiKey, sources), sources);
  console.log('GATE kept:', kept.length, 'dropped:', dropped.length, 'reasons:', JSON.stringify(reasons));

  if (kept.length === 0)
    return { research: 'No verifiable facts survived source-checking.', sources: sourceList, dropped: dropped.length, enriched, dropReasons: reasons, searchHits, via, promoters, independents };

  const byField = {};
  for (const f of kept) (byField[f.field] = byField[f.field] || []).push(f);

  let research = '';
  for (const field in byField) {
    research += `\n## ${field}\n`;
    for (const f of byField[field]) research += `- ${f.value}  [${sources[f.source_id].url}]\n`;
  }
  return { research: research.trim(), sources: sourceList, dropped: dropped.length, enriched, dropReasons: reasons, searchHits, via, promoters, independents };
}
