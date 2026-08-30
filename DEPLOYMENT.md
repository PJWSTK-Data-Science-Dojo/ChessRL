# Web deployment

Luna's production web stack is deliberately small: one Gunicorn worker owns one
inference network and an in-memory registry of browser games. The container runs as a
non-root user with a read-only filesystem, verifies an immutable checkpoint before
startup, and exposes its port only on host loopback. The public overlay adds accelerator
access and an outbound Cloudflare named tunnel; no inbound origin port is required.

The browser service is anonymous and its games are ephemeral. A process restart clears
active games. This is appropriate for a public demonstration, not for an account-backed
chess service.

The two supported serving paths and their isolation boundary are captured in the
[editable Excalidraw architecture](docs/public-serving.excalidraw).

## Deployment invariants

- Deploy only a numbered checkpoint that has completed external evaluation. Do not
  mount the live training directory or publish `latest.pth.tar`.
- Keep Gunicorn at one worker. Multiple workers duplicate both the model and game state.
- Do not run the standalone Lichess UCI process and the web process independently on
  one accelerator. Their locks are process-local, so they can contend for compute and
  memory. Use separate capacity or a future shared inference service.
- Do not train on the serving accelerator. Publish an evaluated artifact after training,
  then restart the inference service onto that immutable release.
- Require HTTPS, an exact trusted hostname, edge rate limits, and the application's
  bounded inference admission policy before accepting public traffic.

## 1. Publish an evaluated checkpoint

The maintained training flow requires a separately installed Stockfish executable and
checks it before a scheduled benchmark. Follow the preflight in
[README.md](README.md#quick-start); the Python dependency alone does not contain the
engine binary.

No checkpoint is bundled with the repository or container image. After external
evaluation has promoted `runs/luna-main/best.pth.tar`, create a versioned release copy with an ID
that identifies the evaluated iteration:

```bash
make release-web-model RELEASE_ID=iteration-400
make verify-web-model RELEASE_ID=iteration-400
```

The release target:

- fails if the evaluated source does not exist;
- refuses to replace an existing release ID;
- creates `release/luna-iteration-400.pth.tar` as read-only;
- creates the matching `.sha256` manifest.

To publish another evaluated numbered snapshot explicitly, override the source:

```bash
make release-web-model \
  RELEASE_ID=iteration-425 \
  RELEASE_SOURCE=./runs/luna-main/checkpoint_425.pth.tar
```

Only use that override after the named snapshot has passed the fixed external benchmark.
Record the artifact path and the 64-character digest in the private deployment
environment. The container entry point recomputes the digest and refuses startup on a
mismatch.

## 2. Run the private container

Create the local runtime configuration:

```bash
test ! -e .env && install -m 0600 .env.example .env
uv run python -c "import secrets; print(secrets.token_hex(32))"
```

Paste the generated secret into `.env`, then set `MODEL_PATH` and `MODEL_SHA256` from the
release created above. Keep `SESSION_COOKIE_SECURE=false` only for direct loopback HTTP.
Validate and launch:

```bash
make web-config
make web-up
curl --fail --silent http://127.0.0.1:5000/api/v1/health
make web-logs
```

The image is built from `uv.lock` with `uv sync --frozen --no-dev --extra perf --extra
web`. The locked uv environment is copied into the non-root runtime image. The minimal
C++ toolchain retained in the runtime is required by PyTorch's trusted runtime code
generation when compiled inference is enabled explicitly. It is disabled by default so
an unmeasured compilation phase cannot prevent the worker from becoming healthy. The
checkpoint is a single read-only bind mount rather than part of the build context.

Stop the private stack with:

```bash
make web-down
```

## 3. Create the public named tunnel

The public overlay uses a remotely managed Cloudflare named tunnel. Cloudflare Tunnel is
outbound-only, so the origin does not need a public IP or an inbound firewall rule. A
Cloudflare-managed domain is required.

In the Cloudflare dashboard:

1. Create a named tunnel for the production deployment.
2. Add a published-application route for the intended hostname, such as
   `play.example.com`.
3. Set the route's service URL to `http://luna-web:5000`.
4. Copy only the tunnel token from the connector command.
5. Enable Cloudflare's **Always Use HTTPS** setting or an equivalent redirect rule for
   the published hostname. Do not expose the application over edge HTTP.

Store the token without echoing it or adding it to command history:

```bash
install -d -m 0700 secrets
read -rsp 'Cloudflare tunnel token: ' CLOUDFLARE_TUNNEL_TOKEN && printf '\n'
umask 077
printf '%s' "$CLOUDFLARE_TUNNEL_TOKEN" > secrets/cloudflare-tunnel.token
unset CLOUDFLARE_TUNNEL_TOKEN
chmod 0600 secrets/cloudflare-tunnel.token
```

The token file is mounted as a Compose secret and never placed in an environment
variable, image layer, Git file, or process argument. Anyone with this token can run the
tunnel, so rotate it immediately if the file is exposed.

Prepare the public environment:

```bash
test ! -e .env.public && install -m 0600 .env.public.example .env.public
uv run python -c "import secrets; print(secrets.token_hex(32))"
```

Set all placeholders in `.env.public`:

- `PUBLIC_HOSTNAME` must contain the exact published hostname. The origin trusts that name plus
  loopback for its container-local health probe;
- `LUNA_WEB_SECRET` must contain the generated random secret;
- `MODEL_PATH` must identify the immutable evaluated release;
- `MODEL_SHA256` must match its manifest;
- `CLOUDFLARE_TUNNEL_TOKEN_FILE` must identify the owner-only token file;
- `CLOUDFLARED_UID` and `CLOUDFLARED_GID` must match that file's non-root owner,
  as reported by `id -u` and `id -g` in the shell that created it.

Compose file-backed secrets preserve the host file's numeric ownership. Running the
tunnel process as the same non-root UID/GID keeps the token at mode `0600` while making
it readable inside the container. Do not set either value to `0`.

Then validate and launch the merged stack:

```bash
make web-public-config
make web-public-up
make web-public-logs
curl --fail --silent https://play.example.com/api/v1/health
curl --head http://play.example.com/
curl --head https://play.example.com/
```

The HTTP response must redirect to HTTPS. The HTTPS response must include the
`Strict-Transport-Security` header before the site is shared publicly.

The overlay removes the private stack's loopback port publication, pins
`cloudflare/cloudflared` release `2026.8.2` and its multi-platform manifest digest,
enables the accelerator for Luna, forces secure cookies, and requires trusted-host
configuration. The application is reachable only through the tunnel's internal Compose
network. Compiled inference remains disabled unless `COMPILE_INFERENCE=true` is set
after measuring cold startup and steady-state latency on the deployment host. Review
upstream releases and image digests deliberately before changing any pin; do not use
floating tags.

Configure Cloudflare WAF rate-limiting rules for the expensive mutation routes under
`/api/v1/games/`, especially `moves`, `hint`, and `engine-move`. Use a short block
period and tune the thresholds from measured legitimate traffic. Keep API responses out
of edge caches. Application admission control remains necessary because edge identity
can be rotated and trusted traffic can still arrive concurrently.

Stop the public stack with:

```bash
make web-public-down
```

## 4. Hosted-machine layout

The same public Compose overlay works on a dedicated GPU VM or pod with Docker Compose
and the NVIDIA Container Toolkit. The host needs only outbound connectivity for the
tunnel and image pulls. Keep SSH restricted to an administrator network; do not expose
port 5000 publicly.

Use Docker Compose 2.24.4 or newer. The public overlay relies on Compose's `!reset` tag
to remove the loopback port published by the private stack.

Use persistent storage for the versioned release artifact and secret files, but do not
use a provider's ephemeral container disk as the only checkpoint copy. Keep an external
backup of the evaluated artifact and its digest. Start with one always-warm instance;
serverless GPU hosting requires a stateless game API and cold-start work that this
process-local Flask architecture does not currently provide.

## 5. Update a deployed model

Never modify a mounted release in place. Create a new release ID, verify it, update
`MODEL_PATH` and `MODEL_SHA256`, and let Compose recreate the web service:

```bash
make release-web-model RELEASE_ID=iteration-425
make verify-web-model RELEASE_ID=iteration-425
make web-public-up
```

The model is loaded once at process startup. Existing games are intentionally discarded
when the container is replaced. Retain the prior release and environment values until
the new health check and a complete browser game succeed, so rollback is a configuration
change rather than a model-file mutation.

## Runtime configuration

| Variable | Required | Purpose |
|---|---:|---|
| `MODEL_PATH` | yes | Host path to one immutable evaluated checkpoint |
| `MODEL_SHA256` | yes | Expected SHA-256 digest enforced before startup |
| `LUNA_WEB_SECRET` | yes | Random Flask session-signing secret |
| `PUBLIC_HOSTNAME` | public | Cloudflare published-application hostname |
| `CLOUDFLARE_TUNNEL_TOKEN_FILE` | public | Owner-only named-tunnel token file |
| `CLOUDFLARED_UID` | public | Non-root UID that owns the tunnel token file |
| `CLOUDFLARED_GID` | public | GID that owns the tunnel token file |
| `DEVICE` | private | Inference backend; public overlay selects CUDA |
| `SEARCH_SIMULATIONS` | no | Base web search budget, minimum 8 |
| `COMPILE_INFERENCE` | no | Opt into compiled inference after measuring cold startup; public default is off |
| `THREADS` | no | HTTP concurrency in the single Gunicorn worker |
| `TIMEOUT` | no | Gunicorn worker-liveness timeout; not an HTTP request deadline |
| `PROXY_HOPS` | no | Number of trusted reverse proxies; public overlay fixes this to one |
| `HSTS_MAX_AGE_SECONDS` | no | HSTS duration; enabled by the public overlay |
| `WEB_CPU_LIMIT` | no | Compose CPU limit for the web container |
| `WEB_MEMORY_LIMIT` | no | Compose memory limit for the web container |
| `WEB_PIDS_LIMIT` | no | Compose process limit for the web container |

With Gunicorn's threaded worker, `TIMEOUT` monitors worker liveness; it does not impose
a per-request deadline. Luna's search deadline and the browser request timeout bound
normal move requests. Keep both below the edge proxy's request limit, and measure search
budgets on the actual host. If compiled inference is enabled, `TIMEOUT` and the health
check start period must also accommodate measured cold worker startup.

## Security and operational checks

- `.env`, `.env.public`, `release/`, and `secrets/` are ignored by Git and the Docker
  build context.
- The web and tunnel containers use read-only roots, dropped Linux capabilities,
  `no-new-privileges`, PID/memory/CPU bounds, temporary writable filesystems, and rotated
  local logs.
- The private stack binds the origin to host loopback. The public overlay publishes no
  host port; public traffic reaches the origin only through the private Compose network.
- Gunicorn access logging is disabled so anonymous client addresses, game identifiers,
  referrers, and user agents are not retained by the application container.
- A missing checkpoint, invalid digest, incompatible model, weak web secret, or invalid
  trusted-host configuration must fail startup.
- Monitor health, request latency, rejected searches, container restarts, GPU memory,
  and tunnel connectivity. Treat repeated timeouts as a capacity or search-budget issue,
  not as a reason to increase queues indefinitely.

Cloudflare references:

- [Cloudflare Tunnel overview](https://developers.cloudflare.com/tunnel/)
- [Named-tunnel setup](https://developers.cloudflare.com/tunnel/setup/)
- [`--token-file` run parameter](https://developers.cloudflare.com/tunnel/advanced/run-parameters/)
- [Rate-limiting rules](https://developers.cloudflare.com/waf/rate-limiting-rules/)

For the separate Bot API workflow, see [LICHESS.md](LICHESS.md). Its UCI process must not
be launched beside this web stack on the same accelerator until both use one shared
inference owner.
