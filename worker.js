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

    try {
      const response = await env.AI.run('@cf/meta/llama-3.3-70b-instruct-fp8-fast', {
        messages: [
          { role: 'system', content: env.SYSTEM_PROMPT },
          { role: 'user', content: body.text },
        ],
        max_tokens: 4096,
      });

      const text = response.response ?? '';
      return new Response(JSON.stringify({ result: text }), {
        headers: { 'content-type': 'application/json', ...corsHeaders },
      });
    } catch (e) {
      return new Response(JSON.stringify({ error: e.message }), {
        status: 500,
        headers: { 'content-type': 'application/json', ...corsHeaders },
      });
    }
  },
};
