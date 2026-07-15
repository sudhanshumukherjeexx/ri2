# AI Insight proxy

OpenAI's API doesn't send `Access-Control-Allow-Origin`, so a static site can't
call `api.openai.com` directly from browser JS — every request is blocked by
CORS. This is a stateless Cloudflare Worker that does nothing but forward the
visitor's request to OpenAI and add that header back. It never stores, logs,
or reads the API key it forwards.

This is the one piece of the app that can't run on GitHub Pages itself
(Pages only serves static files) and requires manual deployment with your own
Cloudflare account — it can't be deployed as part of the repo's automated
build.

## Deploy

```bash
cd proxy
npx wrangler login       # one-time, opens a browser to authorize
npx wrangler deploy
```

This prints a URL like `https://ride-openai-proxy.<your-subdomain>.workers.dev`.

## Wire it up

Add that URL to `.env` at the repo root:

```
VITE_AI_PROXY_URL=https://ride-openai-proxy.<your-subdomain>.workers.dev
```

Rebuild the app (`npm run build`) so Vite bakes the URL into the static
bundle. Without this variable set, the AI Insight button will show an error
explaining the proxy isn't configured instead of silently failing.

## Before going live

Edit `ALLOWED_ORIGIN` in `worker.js` from `'*'` to your actual GitHub Pages
origin (e.g. `https://sudhanshumukherjeexx.github.io`) and redeploy, so the
proxy only accepts requests from your site.
