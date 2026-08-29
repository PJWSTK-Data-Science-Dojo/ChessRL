# Web deployment

The production web process loads one format-v2 checkpoint, creates the Flask application through `create_app()`, and fails at startup if the model cannot be loaded. The service keeps browser sessions and game state in memory, so it must run as one process. A threaded worker still accepts concurrent HTTP requests; model inference is serialized by the engine service.

## Required runtime state

- A readable `temp/latest.pth.tar` or another explicit format-v2 checkpoint.
- A Python 3.11 or 3.12 environment with the project and Gunicorn installed.
- A random `LUNA_WEB_SECRET` of at least 32 characters.
- The correct `DEVICE` for the host: `cuda`, `mps`, or `cpu`.

No checkpoint or secret is built into the container image.

## Gunicorn

Install the project and production server:

```bash
uv sync --extra web
```

Configure the process without placing secrets in command history:

```bash
export LUNA_WEB_SECRET="$(uv run python -c 'import secrets; print(secrets.token_hex(32))')"
export CHECKPOINT_PATH="$(pwd)/temp/latest.pth.tar"
export DEVICE=cuda
export SEARCH_SIMULATIONS=96
export COMPILE_INFERENCE=true
export HOST=127.0.0.1
export PORT=5000

uv run gunicorn --config gunicorn.conf.py wsgi:app
```

Readiness is successful only after the checkpoint has loaded:

```bash
curl --fail --silent http://127.0.0.1:5000/api/v1/health
```

The Gunicorn configuration deliberately uses one `gthread` worker. Do not increase the worker count: each process would load another model and maintain an independent game registry. Increase `THREADS` only for HTTP concurrency; neural searches remain serialized.

### Environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `HOST` | `127.0.0.1` | Gunicorn bind address |
| `PORT` | `5000` | Gunicorn port |
| `CHECKPOINT_PATH` | `./temp/latest.pth.tar` | Versioned runtime checkpoint |
| `DEVICE` | `cuda` | `cuda`, `mps`, or `cpu` |
| `SEARCH_SIMULATIONS` | `96` | Base search profile budget; minimum `8` |
| `COMPILE_INFERENCE` | `true` | Enable compiled inference where supported |
| `LUNA_WEB_SECRET` | none | Required session-signing secret, at least 32 characters |
| `SESSION_COOKIE_SECURE` | `true` | Send the session cookie only over HTTPS |
| `THREADS` | `4` | HTTP threads in the single worker |
| `TIMEOUT` | `180` | Hard request timeout in seconds |
| `GRACEFUL_TIMEOUT` | `30` | Graceful shutdown timeout |
| `LOG_LEVEL` | `info` | Gunicorn log level |

For direct local HTTP testing only, set `SESSION_COOKIE_SECURE=false`. Keep it enabled behind TLS.

## Docker Compose

Create a private local environment file:

```bash
cp .env.example .env
uv run python -c 'import secrets; print(secrets.token_hex(32))'
```

Paste the generated value into `LUNA_WEB_SECRET` in `.env`, select the device, and verify that `temp/latest.pth.tar` exists. Then run:

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f luna-web
```

The Compose service mounts `./temp` read-only, probes `/api/v1/health`, and publishes only on `127.0.0.1` by default. Set `BIND_ADDRESS` deliberately if a reverse proxy runs on another host. Secure cookies are enabled by default; for direct local HTTP testing only, set `SESSION_COOKIE_SECURE=false`. The default Compose device is CPU for portability. To expose an NVIDIA-compatible accelerator, create an untracked `compose.override.yml`:

```yaml
services:
  luna-web:
    gpus: all
    environment:
      DEVICE: cuda
      COMPILE_INFERENCE: "true"
```

Restart after publishing a new runtime checkpoint because the model is loaded only at process startup:

```bash
docker compose restart luna-web
```

## Reverse proxy

Terminate TLS at a reverse proxy and keep Gunicorn bound to a private address. A minimal Nginx location is:

```nginx
location / {
    proxy_pass http://127.0.0.1:5000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 180s;
    proxy_send_timeout 180s;
}
```

Add request-rate limiting at the proxy before exposing the service publicly. The application uses signed, HTTP-only, same-site session cookies and per-session game ownership; it is not an account or identity system.

## systemd example

```ini
[Unit]
Description=Luna Chess web service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=luna
Group=luna
WorkingDirectory=/srv/luna-chess
EnvironmentFile=/srv/luna-chess/.env
ExecStart=/srv/luna-chess/.venv/bin/gunicorn --config gunicorn.conf.py wsgi:app
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Protect the environment file with `chmod 600 .env`. Keep `HOST=127.0.0.1` when a same-host proxy fronts the service.

## Operational checks

```bash
test -r temp/latest.pth.tar
curl --fail --silent https://your-host.example/api/v1/health
docker compose logs --tail=100 -f luna-web
```

The health response includes the loaded checkpoint filename and search profiles, but never filesystem paths or secrets. A missing, legacy, or architecture-incompatible checkpoint prevents startup instead of serving random weights.

Back up only numbered checkpoints and evaluation metadata that you intend to retain. `latest.pth.tar` is a publication pointer and `best.pth.tar` exists only after an external benchmark promotes it.

For Lichess Bot API deployment, use the separate token-safe workflow in [LICHESS.md](LICHESS.md).
