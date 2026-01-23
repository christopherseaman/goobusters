# Cloudflare Worker Setup for CORS and Access Headers

This worker handles CORS preflight requests and automatically adds Cloudflare Access service token headers to all requests.

## Prerequisites

- Cloudflare account with Workers enabled
- `wrangler` CLI installed: `npm install -g wrangler`
- Service token Client ID and Client Secret from Cloudflare Access

## Setup Steps

### 1. Create Worker Project

```bash
cd /Users/christopher/Documents/goobusters
mkdir cloudflare-worker
cd cloudflare-worker
npm init -y
npm install --save-dev wrangler
```

### 2. Create `wrangler.toml`

```toml
name = "goobusters-cors-proxy"
main = "worker.js"
compatibility_date = "2024-01-01"

[env.production]
routes = [
  { pattern = "goo.badmath.org/*", zone_name = "badmath.org" }
]

[vars]
# These will be set as secrets in step 4
```

### 3. Copy Worker Code

Copy `cloudflare-worker-cors.js` to `worker.js` in the worker directory.

### 4. Set Environment Variables (Secrets)

```bash
cd cloudflare-worker
npx wrangler secret put CF_ACCESS_CLIENT_ID
# Paste: 7349a901a19c2250d0e7c906823cdcf8.access

npx wrangler secret put CF_ACCESS_CLIENT_SECRET
# Paste: ff451245c5cd6131dc9c8019a0245cac296beb77520759b07660629efe6bb793
```

### 5. Deploy Worker

```bash
npx wrangler deploy
```

### 6. Configure Route

The route should be automatically configured if you set it in `wrangler.toml`. Otherwise:

1. Go to Cloudflare Dashboard → Workers & Pages
2. Select your worker
3. Go to Triggers → Routes
4. Add route: `goo.badmath.org/*`

### 7. Configure Cloudflare Access

In Cloudflare Access, you may need to:
- Ensure the application allows Service Auth policies
- The service token should be added to the application's policies

## Alternative: Configure Access to Bypass OPTIONS

If you prefer not to use a Worker, you can configure Cloudflare Access directly:

1. Go to Cloudflare One → Access controls → Applications
2. Select `goo.badmath.org` application
3. Go to Advanced settings → Cross-Origin Resource Sharing (CORS) settings
4. Turn on **Bypass options requests to origin**

This allows OPTIONS requests to pass through without authentication, while actual requests still require the service token headers.

## Testing

After deployment, test with:

```bash
# Test preflight
curl -X OPTIONS https://goo.badmath.org/api/series/next \
  -H "Origin: http://localhost:8080" \
  -H "Access-Control-Request-Method: GET" \
  -H "Access-Control-Request-Headers: CF-Access-Client-Id,CF-Access-Client-Secret" \
  -v

# Should return 204 with CORS headers
```

## Notes

- The worker automatically adds Cloudflare Access headers to all requests
- Preflight OPTIONS requests are handled without requiring authentication
- CORS headers are added to all responses
- The worker proxies requests to the origin server
