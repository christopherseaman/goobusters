/**
 * Cloudflare Worker to handle CORS preflight and add Cloudflare Access headers
 * 
 * Deploy this worker and route all requests to https://goo.badmath.org through it.
 * 
 * Setup:
 * 1. Deploy this worker to Cloudflare
 * 2. Add route: goo.badmath.org/*
 * 3. Set environment variables:
 *    - CF_ACCESS_CLIENT_ID: Your service token Client ID
 *    - CF_ACCESS_CLIENT_SECRET: Your service token Client Secret
 */

export default {
  async fetch(request, env) {
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*", // Or specific origin like "http://localhost:8080"
      "Access-Control-Allow-Methods": "GET, HEAD, POST, PUT, DELETE, PATCH, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, X-User-Email, X-Previous-Version-ID, X-Editor, CF-Access-Client-Id, CF-Access-Client-Secret",
      "Access-Control-Allow-Credentials": "true",
      "Access-Control-Max-Age": "86400",
    };

    // Handle CORS preflight requests
    if (request.method === "OPTIONS") {
      const origin = request.headers.get("Origin");
      const requestMethod = request.headers.get("Access-Control-Request-Method");
      const requestHeaders = request.headers.get("Access-Control-Request-Headers");

      if (origin && requestMethod && requestHeaders) {
        // CORS preflight - return allowed headers/methods
        return new Response(null, {
          status: 204,
          headers: {
            ...corsHeaders,
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": requestMethod,
            "Access-Control-Allow-Headers": requestHeaders,
          },
        });
      } else {
        // Standard OPTIONS request
        return new Response(null, {
          status: 204,
          headers: {
            Allow: "GET, HEAD, POST, PUT, DELETE, PATCH, OPTIONS",
          },
        });
      }
    }

    // For actual requests, add Cloudflare Access headers and proxy to origin
    const url = new URL(request.url);
    
    // Create new request to origin with Cloudflare Access headers
    const originRequest = new Request(request);
    
    // Add Cloudflare Access service token headers if configured
    if (env.CF_ACCESS_CLIENT_ID && env.CF_ACCESS_CLIENT_SECRET) {
      originRequest.headers.set("CF-Access-Client-Id", env.CF_ACCESS_CLIENT_ID);
      originRequest.headers.set("CF-Access-Client-Secret", env.CF_ACCESS_CLIENT_SECRET);
    }

    // Forward request to origin
    const response = await fetch(originRequest);

    // Clone response to modify headers
    const modifiedResponse = new Response(response.body, response);

    // Add CORS headers to response
    const origin = request.headers.get("Origin");
    if (origin) {
      modifiedResponse.headers.set("Access-Control-Allow-Origin", origin);
      modifiedResponse.headers.append("Vary", "Origin");
    }

    // Copy other CORS headers
    Object.entries(corsHeaders).forEach(([key, value]) => {
      if (key !== "Access-Control-Allow-Origin") {
        modifiedResponse.headers.set(key, value);
      }
    });

    return modifiedResponse;
  },
};
