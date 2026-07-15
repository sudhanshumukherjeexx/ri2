// Stateless CORS-forwarding proxy for the AI Insight feature.
//
// OpenAI's API does not send Access-Control-Allow-Origin, so a static site
// cannot call it directly from browser JS. This worker exists solely to add
// that header: it forwards the visitor's own Authorization header and
// request body to OpenAI, streams the response straight back, and never
// stores, logs, or inspects the key. Swap ALLOWED_ORIGIN below to your
// GitHub Pages origin before going live.

const ALLOWED_ORIGIN = '*'

function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  }
}

export default {
  async fetch(request) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders() })
    }

    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405, headers: corsHeaders() })
    }

    const authorization = request.headers.get('Authorization')
    if (!authorization) {
      return new Response(JSON.stringify({ error: { message: 'Missing Authorization header' } }), {
        status: 401,
        headers: { ...corsHeaders(), 'Content-Type': 'application/json' },
      })
    }

    const body = await request.text()

    const upstream = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: authorization,
      },
      body,
    })

    const responseBody = await upstream.text()
    return new Response(responseBody, {
      status: upstream.status,
      headers: { ...corsHeaders(), 'Content-Type': 'application/json' },
    })
  },
}
