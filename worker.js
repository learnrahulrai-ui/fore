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
      const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${env.GEMINI_API_KEY}`;

      const res = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          system_instruction: { parts: [{ text: env.SYSTEM_PROMPT }] },
          contents: [{ parts: [{ text: body.text }] }],
          generationConfig: { maxOutputTokens: 4096 },
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        const msg = data.error?.message || 'Gemini API error';
        return new Response(JSON.stringify({ error: msg }), {
          status: res.status,
          headers: { 'content-type': 'application/json', ...corsHeaders },
        });
      }

      const text = data.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
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
