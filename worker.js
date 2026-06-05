// ---------------------------------------------------------------------------
// Grounded analysis worker
//
// The rule: the model is NEVER the source of a fact. For company research it
// only proposes facts it claims come from text we actually fetched, and for
// each it must hand back the exact quote. Code then checks that quote is really
// in that source (the GATE). Anything that fails is dropped before a human sees
// it. Retrieval that returns nothing is an honest empty result, not an error.
//
// Retrieval here = Wikipedia (free, no key, does not block Workers). Primary
// sources (SEBI/MCA/NSE) block datacenter fetches and need paid access — wire
// those into fetchSources() later via a real search/data provider.
// ---------------------------------------------------------------------------

const ALLOWED_ORIGIN = 'https://learnrahulrai-ui.github.io';

const FAST_MODEL = '@cf/meta/llama-3.1-8b-instruct';
const MAIN_MODEL = '@cf/meta/llama-3.3-70b-instruct-fp8-fast';

// If you later add real primary sources, list their domains here.
const PRIMARY_DOMAINS = new Set([
  'sebi.gov.in', 'nseindia.com', 'bseindia.com', 'mca.gov.in',
  'indiankanoon.org', 'nclt.gov.in',
]);

const CORS = {
  'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'content-type': 'application/json', ...CORS },
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

      // --- Extract the company name first (we need it to drive retrieval) ---
      const companyName = await extractCompanyName(env, pdfText);
      const cacheKey = 'co_' + companyName.toLowerCase()
        .replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '').slice(0, 80);

      // --- Step 1: grounded company research (cache-first) ---
      let research, sources, droppedCount, fromCache = false;

      if (cacheKey !== 'co_' && env.COMPANY_CACHE) {
        try {
          const cached = await env.COMPANY_CACHE.get(cacheKey, 'json');
          if (cached && cached.research !== undefined) {
            ({ research, sources, droppedCount } = cached);
            fromCache = true;
          }
        } catch { /* stale/old-format entry -> treat as miss */ }
      }

      if (research === undefined) {
        const r = await groundedResearch(env, companyName);
        research = r.research;
        sources = r.sources;
        droppedCount = r.dropped;
        if (cacheKey !== 'co_' && env.COMPANY_CACHE) {
          await env.COMPANY_CACHE.put(
            cacheKey,
            JSON.stringify({ research, sources, droppedCount }),
            { expirationTtl: 2592000 } // 30 days
          );
        }
      }

      // --- Step 2: analyze the actual PDF, with verified facts as context ---
      const analysis = await runModel(
        env, MAIN_MODEL, env.SYSTEM_PROMPT_2,
        `VERIFIED COMPANY FACTS:\n${research}\n\nDOCUMENT:\n${pdfText}`,
        2048
      );

      // --- Step 3: condense into the final report ---
      const report = await runModel(env, MAIN_MODEL, env.SYSTEM_PROMPT_3, analysis, 1024);

      return json({
        company: companyName,
        cached: fromCache,
        research,
        sources,
        dropped_count: droppedCount,
        analysis,
        report,
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
// 1. RETRIEVAL — Wikipedia only (free, reliable). Empty result is honest.
// ---------------------------------------------------------------------------
function trustOf(url) {
  const dom = url.replace(/^https?:\/\/(www\.)?/, '').split('/')[0];
  if (PRIMARY_DOMAINS.has(dom)) return 'primary';
  if (dom.endsWith('wikipedia.org')) return 'reference';
  return 'news';
}

async function fetchSources(companyName) {
  const sources = {};
  const UA = { 'User-Agent': 'fore-analyzer/1.0 (educational project)' };
  try {
    const searchUrl = 'https://en.wikipedia.org/w/api.php?action=query&list=search'
      + '&srsearch=' + encodeURIComponent(companyName)
      + '&srlimit=2&format=json&origin=*';
    const sj = await (await fetch(searchUrl, { headers: UA })).json();
    const hits = sj?.query?.search ?? [];

    for (const h of hits) {
      const title = h.title;
      const exUrl = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts'
        + '&explaintext=1&redirects=1&titles=' + encodeURIComponent(title)
        + '&format=json&origin=*';
      const ej = await (await fetch(exUrl, { headers: UA })).json();
      const pages = ej?.query?.pages ?? {};
      for (const pid in pages) {
        const text = pages[pid].extract;
        if (text && text.length > 200) {
          const url = 'https://en.wikipedia.org/wiki/' + encodeURIComponent(title.replace(/ /g, '_'));
          const id = 's' + Object.keys(sources).length;
          sources[id] = { id, url, trust: trustOf(url), text };
        }
      }
    }
  } catch {
    // retrieval failure -> empty, handled honestly upstream
  }
  return sources;
}

// ---------------------------------------------------------------------------
// 2. EXTRACTION — model returns facts, each WITH a verbatim quote + source id.
// ---------------------------------------------------------------------------
const EXTRACT_PROMPT = `You are given SOURCES: real text we fetched. Extract the FIELDS.
For EVERY value, copy an exact verbatim quote (at least 8 words) from the SOURCE it
came from, and give that source's id. Copy the quote character-for-character; do not
paraphrase. If a field is not present in any SOURCE, do not include it. Invent nothing
- no name, company, date, or action that is not in the text.

Return ONLY a JSON array, no prose:
[{"field":"...","value":"...","source_id":"...","quote":"..."}]

FIELDS:
- company_legal_name
- promoter_names (one row per person)
- related_companies (one row per company: subsidiaries, parents, associates)
- regulatory_actions (value format: "<what happened> | legal_status: alleged|order|conviction")
- key_events (mergers, IPO/FPO/QIP, major controversies)

SOURCES:
`;

async function extractFacts(env, sources) {
  const ids = Object.keys(sources);
  if (ids.length === 0) return [];
  const blob = ids
    .map(id => `[${id}] ${sources[id].url}\n${sources[id].text.slice(0, 8000)}`)
    .join('\n\n');
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
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// 3. THE GATE — a fact survives ONLY if its quote is really in its source.
//    Deterministic code. This is the line that stops invented people.
// ---------------------------------------------------------------------------
const norm = t => String(t || '').replace(/\s+/g, ' ').trim().toLowerCase();

function verifyFact(f, sources) {
  if (norm(f.value) === 'not found') return false;
  const src = sources[f.source_id];
  if (!src) return false;                                 // source doesn't exist
  if (String(f.quote).trim().split(/\s+/).length < 8) return false; // too short
  if (!norm(src.text).includes(norm(f.quote))) return false;        // invented/altered
  return true;
}

function gate(facts, sources) {
  const kept = [], dropped = [];
  for (const f of facts) (verifyFact(f, sources) ? kept : dropped).push(f);
  return { kept, dropped };
}

// ---------------------------------------------------------------------------
// Compose verified research text from surviving facts.
// ---------------------------------------------------------------------------
async function groundedResearch(env, companyName) {
  const sources = await fetchSources(companyName);
  const sourceList = Object.values(sources).map(s => ({ url: s.url, trust: s.trust }));

  if (Object.keys(sources).length === 0) {
    return { research: 'No public source could be retrieved for this company.', sources: [], dropped: 0 };
  }

  const facts = await extractFacts(env, sources);
  const { kept, dropped } = gate(facts, sources);

  if (kept.length === 0) {
    return {
      research: 'No verifiable facts survived source-checking.',
      sources: sourceList,
      dropped: dropped.length,
    };
  }

  const byField = {};
  for (const f of kept) (byField[f.field] = byField[f.field] || []).push(f);

  let research = '';
  for (const field in byField) {
    research += `\n## ${field}\n`;
    for (const f of byField[field]) {
      research += `- ${f.value}  [${sources[f.source_id].url}]\n`;
    }
  }

  return { research: research.trim(), sources: sourceList, dropped: dropped.length };
}
