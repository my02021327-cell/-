# BioGuard Backend Deployment

## Architecture

Development/LAN:

```text
Client -> http://SERVER_IP:8000 -> Uvicorn/FastAPI -> SQLite + Parquet/upload storage
```

Production:

```text
Client -> HTTPS -> Nginx -> 127.0.0.1:8000 -> FastAPI -> persistent DATA_DIR
```

FastAPI serves both the existing HMI routes and `/api/v1` dataset APIs. SQLite stores dataset and forecast metadata; uploaded workbooks, processed Parquet files, and prediction audit JSON files live under `DATA_DIR`.

## Local development

Create a virtual environment and install the existing backend dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
```

Run with environment defaults (localhost only):

```bash
python -m backend.run
```

Run on all local interfaces for LAN testing:

```bash
HOST=0.0.0.0 PORT=8000 python -m backend.run
```

Alternatively, copy `.env.example` to a local `.env` and use Uvicorn's environment-file support:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

Do not commit the real `.env` file.

## LAN access

1. Bind with `HOST=0.0.0.0`.
2. Find the server's LAN address, for example `192.168.0.20`.
3. Permit inbound TCP/8000 only on the trusted LAN firewall profile.
4. Open `http://192.168.0.20:8000` or `http://192.168.0.20:8000/datasets/<dataset_id>` from another computer.

`0.0.0.0` is a bind address, not a browser destination. Use the server's real LAN IP from the client computer.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | Uvicorn bind address used by `python -m backend.run` |
| `PORT` | `8000` | Uvicorn port |
| `ENV` | `development` | `development` or `production` configuration mode |
| `DATA_DIR` | `./backend/storage` | Persistent upload, Parquet, prediction, and default SQLite root |
| `DATABASE_URL` | SQLite under `DATA_DIR` | SQLite URL such as `sqlite:////var/lib/bioguard/bioguard.sqlite3` |
| `ALLOWED_ORIGINS` | bounded local origins in development; empty in production | Comma-separated exact CORS origins |
| `FORWARDED_ALLOW_IPS` | `127.0.0.1` | Reverse proxies trusted for forwarded headers |
| `MODEL_BUNDLE_PATH` | project deployment artifact | Optional model-bundle override |
| `RELIABILITY_SEED_PATH` | project deployment artifact | Optional reliability-seed override |

The current repository supports SQLite only. A non-SQLite `DATABASE_URL` fails at startup rather than silently selecting another database.

## Production deployment

1. Copy the project and model artifacts to the Linux server.
2. Create a dedicated unprivileged service account.
3. Create a virtual environment and install `backend/requirements.txt`.
4. Create a persistent directory such as `/var/lib/bioguard`, owned by the service account.
5. Create a local `.env` with `ENV=production`, `HOST=127.0.0.1`, the persistent `DATA_DIR`, and exact `ALLOWED_ORIGINS` if a separate frontend origin is used.
6. Replace `@SERVICE_USER@`, `@SERVICE_GROUP@`, and `@PROJECT_DIR@` in `deploy/systemd/bioguard-backend.service.example`, install it under `/etc/systemd/system/`, then enable it.
7. Replace `@SERVER_NAME@` in `deploy/nginx/bioguard-backend.conf.example`, enable the Nginx site, and validate with `nginx -t`.

Typical service commands:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bioguard-backend
sudo systemctl status bioguard-backend
```

## Firewall

- LAN-only development: TCP/8000 may be allowed from the trusted subnet.
- Production with Nginx: keep Uvicorn bound to `127.0.0.1`; expose only TCP/80 and TCP/443.
- Do not expose SQLite files, `DATA_DIR`, model artifacts, or `.env` through the web server.

## HTTPS

Certificates are not stored in this repository. After DNS points to the server and the HTTP Nginx site works, install Certbot using the operating system's supported package and request a certificate for the real hostname. For example on a Debian/Ubuntu host with the Nginx plugin:

```bash
sudo certbot --nginx -d bioguard.example.com
sudo certbot renew --dry-run
```

## Verification

Run after deployment:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/api/v1/health
curl -fsS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/datasets/c3d8d5b85256fb91
curl -fsS http://127.0.0.1:8000/api/v1/datasets/c3d8d5b85256fb91
```

Through Nginx, repeat the checks with the public HTTPS hostname. Back up `DATA_DIR` consistently; the SQLite database and referenced Parquet/upload files belong to one persistent dataset state.
