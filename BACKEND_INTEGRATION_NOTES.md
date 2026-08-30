# BioGuard Backend Integration Notes

## Current contract

The current frontend is a static HTML/JavaScript client served by FastAPI. Its API base is same-origin (`/api/v1`), so no frontend change is required when Nginx proxies the complete site to the backend.

Client route preservation:

- Browser route: `GET /datasets/{dataset_id}`
- Dataset API: `GET /api/v1/datasets/{dataset_id}`

These are intentionally different routes and both remain available.

## If the frontend is hosted separately

No change is required for the current same-origin deployment. If a future deployment serves the static frontend from a different origin, the frontend owner should:

1. Add a plain-JavaScript runtime setting such as `window.BIOGUARD_API_BASE_URL` before loading `backend/static/bioguard_v2.js`.
2. Build API requests from that value instead of the current same-origin `/api/v1` value.
3. Add the exact frontend origin to backend `ALLOWED_ORIGINS`.
4. Keep `/datasets/{dataset_id}` navigation on the frontend host or configure its web server with an SPA fallback.

Do not use a wildcard CORS origin in production.

## Reverse proxy recommendation

The preferred deployment proxies `/`, `/datasets/...`, `/static/...`, and `/api/v1/...` through the same public hostname. This preserves the existing frontend contract and avoids a cross-origin dependency.
