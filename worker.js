const ALLOWED_ORIGIN = 'https://learnrahulrai-ui.github.io';

export default {
  async fetch(request, env) {
    const corsHeaders = {
      'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405 });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return new Response('Invalid JSON', { status: 400, headers: corsHeaders });
    }

    if (!body.text) {
      return new Response('Missing text', { status: 400, headers: corsHeaders });
    }

    const pdfText = body.text;

    // Step 1a: Extract company name from beginning of PDF (cheap, small call)
    const nameResult = await env.AI.run('@cf/meta/llama-3.1-8b-instruct', {
      messages: [
        { role: 'system', content: 'Extract only the primary company name from this financial document. Return just the company name, nothing else. No explanation.' },
        { role: 'user', content: pdfText.slice(0, 4000) },
      ],
      max_tokens: 30,
    });
    const companyName = (nameResult.response ?? '').trim();
    const cacheKey = 'co_' + companyName.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '').slice(0, 80);

    // Step 1b: Company research — check KV cache first
    let companyResearch = '';
    let fromCache = false;

    if (cacheKey !== 'co_' && env.COMPANY_CACHE) {
      const cached = await env.COMPANY_CACHE.get(cacheKey);
      if (cached) {
        companyResearch = cached;
        fromCache = true;
      }
    }

    if (!companyResearch) {
      const researchResult = await env.AI.run('@cf/meta/llama-3.3-70b-instruct-fp8-fast', {
        messages: [
          { role: 'system', content: env.SYSTEM_PROMPT_1 },
          { role: 'user', content: `Company name: ${companyName}` },
        ],
        max_tokens: 2048,
      });
      companyResearch = researchResult.response ?? '';
      if (cacheKey !== 'co_' && companyResearch && env.COMPANY_CACHE) {
        await env.COMPANY_CACHE.put(cacheKey, companyResearch, { expirationTtl: 2592000 }); // 30 days
      }
    }

    // Step 2: Analyze the actual PDF content with company context
    const step2Result = await env.AI.run('@cf/meta/llama-3.3-70b-instruct-fp8-fast', {
      messages: [
        { role: 'system', content: env.SYSTEM_PROMPT_2 },
        { role: 'user', content: `COMPANY BACKGROUND:\n${companyResearch}\n\nDOCUMENT:\n${pdfText}` },
      ],
      max_tokens: 2048,
    });
    const analysis = step2Result.response ?? '';

    // Step 3: Condense into 3-paragraph report
    const step3Result = await env.AI.run('@cf/meta/llama-3.3-70b-instruct-fp8-fast', {
      messages: [
        { role: 'system', content: env.SYSTEM_PROMPT_3 },
        { role: 'user', content: analysis },
      ],
      max_tokens: 1024,
    });
    const finalReport = step3Result.response ?? '';

    return new Response(JSON.stringify({
      company: companyName,
      cached: fromCache,
      research: companyResearch,
      analysis: analysis,
      report: finalReport,
    }), {
      headers: { 'content-type': 'application/json', ...corsHeaders },
    });
  },
};
