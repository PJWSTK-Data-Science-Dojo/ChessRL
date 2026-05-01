# Luna Chess Web App - Deployment Guide

## Production Deployment Options

### Option 1: Gunicorn (Recommended for Linux/Mac)

#### Prerequisites
```bash
pip install gunicorn
```

#### Basic Deployment
```bash
# Production server with 4 workers
gunicorn --config gunicorn.conf.py wsgi:app
```

#### Custom Configuration
```bash
# Set environment variables
export DEVICE=cuda
export MCTS_SIMS=50
export PORT=8000
export WORKERS=8

# Run with custom config
gunicorn --config gunicorn.conf.py wsgi:app
```

#### Using .env File
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings
nano .env

# Load env vars and run
source .env
gunicorn --config gunicorn.conf.py wsgi:app
```

#### Systemd Service (Linux)

Create `/etc/systemd/system/luna-chess.service`:

```ini
[Unit]
Description=Luna Chess Web Service
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/path/to/ChessRL
Environment="PATH=/path/to/ChessRL/.venv/bin"
Environment="DEVICE=cuda"
Environment="MCTS_SIMS=50"
ExecStart=/path/to/ChessRL/.venv/bin/gunicorn --config gunicorn.conf.py wsgi:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable luna-chess
sudo systemctl start luna-chess
sudo systemctl status luna-chess
```

### Option 2: Docker

#### Build and Run
```bash
# Build image
docker build -f Dockerfile.web -t luna-chess-web .

# Run container
docker run -d \
  --name luna-chess \
  -p 5000:5000 \
  -v $(pwd)/temp:/app/temp:ro \
  -e DEVICE=cpu \
  -e MCTS_SIMS=50 \
  luna-chess-web
```

#### With Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### With GPU Support
```bash
# Install nvidia-docker2
# Then run with GPU
docker run -d \
  --name luna-chess \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/temp:/app/temp:ro \
  -e DEVICE=cuda \
  -e MCTS_SIMS=100 \
  luna-chess-web
```

### Option 3: Nginx Reverse Proxy

#### Nginx Configuration

Create `/etc/nginx/sites-available/luna-chess`:

```nginx
upstream luna_chess {
    server 127.0.0.1:5000;
}

server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 10M;

    location / {
        proxy_pass http://luna_chess;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long MCTS calculations
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }

    location /static {
        alias /path/to/ChessRL/src/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/luna-chess /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### With SSL (Let's Encrypt)
```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Option 4: Cloud Deployment

#### AWS EC2

1. **Launch Instance**
   - Choose Ubuntu 22.04 AMI
   - Instance type: t3.medium (CPU) or g4dn.xlarge (GPU)
   - Storage: 50GB SSD

2. **Setup**
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Clone repo
git clone https://github.com/your-repo/ChessRL.git
cd ChessRL

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Setup project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-web.txt

# Run with gunicorn
gunicorn --config gunicorn.conf.py wsgi:app
```

3. **Configure Security Group**
   - Allow inbound: Port 80 (HTTP), 443 (HTTPS)
   - Allow SSH from your IP only

#### Heroku

Create `Procfile`:
```
web: gunicorn --config gunicorn.conf.py wsgi:app
```

Deploy:
```bash
heroku create luna-chess-app
git push heroku main
heroku config:set DEVICE=cpu MCTS_SIMS=25
heroku ps:scale web=1
```

#### Google Cloud Run

Build and deploy:
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/luna-chess
gcloud run deploy luna-chess \
  --image gcr.io/PROJECT-ID/luna-chess \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 120 \
  --set-env-vars DEVICE=cpu,MCTS_SIMS=50
```

## Performance Optimization

### 1. Adjust Worker Count

**CPU-bound tasks (MCTS calculation):**
```bash
# Use fewer workers for CPU
WORKERS=2 gunicorn --config gunicorn.conf.py wsgi:app
```

**I/O-bound tasks:**
```bash
# Use more workers
WORKERS=8 gunicorn --config gunicorn.conf.py wsgi:app
```

### 2. Configure MCTS Simulations

**Fast response (weaker play):**
```bash
export MCTS_SIMS=10
```

**Balanced:**
```bash
export MCTS_SIMS=50
```

**Strong play (slower):**
```bash
export MCTS_SIMS=200
```

### 3. Enable Caching

Add to Nginx config:
```nginx
location /static {
    alias /path/to/ChessRL/src/static;
    expires 1y;
    add_header Cache-Control "public, immutable";
}

location / {
    # ... existing proxy settings ...
    proxy_cache_bypass $http_upgrade;
}
```

### 4. Use GPU for Better Performance

```bash
export DEVICE=cuda
export MCTS_SIMS=100
gunicorn --config gunicorn.conf.py wsgi:app
```

## Monitoring

### Basic Health Check
```bash
curl http://localhost:5000/model_info
```

### Logs

**Gunicorn logs:**
```bash
# View access logs
tail -f /var/log/luna-chess/access.log

# View error logs
tail -f /var/log/luna-chess/error.log
```

**Systemd logs:**
```bash
sudo journalctl -u luna-chess -f
```

**Docker logs:**
```bash
docker logs -f luna-chess
```

### Metrics

Add Prometheus metrics (future enhancement):
```python
from prometheus_flask_exporter import PrometheusMetrics
metrics = PrometheusMetrics(app)
```

## Security

### 1. Rate Limiting

Install Flask-Limiter:
```bash
pip install Flask-Limiter
```

Add to web_app.py:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)
```

### 2. CORS (if needed)

```bash
pip install flask-cors
```

```python
from flask_cors import CORS
CORS(app, origins=["https://your-frontend.com"])
```

### 3. HTTPS Only

Nginx config:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # ... rest of config
}
```

### 4. Firewall Rules

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

## Scaling

### Horizontal Scaling

1. **Load Balancer** - Distribute across multiple instances
2. **Shared Model Storage** - Mount checkpoint from S3/NFS
3. **Session Affinity** - Stick user to same worker

### Vertical Scaling

1. **Larger Instance** - More CPU/RAM
2. **GPU Instance** - Faster inference
3. **More Workers** - Parallel request handling

## Troubleshooting

### Issue: "Model not found"
```bash
# Ensure checkpoint exists
ls -la temp/best.pth.tar

# Or train a model
make train
```

### Issue: "Timeout on moves"
```bash
# Increase timeout
export TIMEOUT=300
gunicorn --config gunicorn.conf.py wsgi:app
```

### Issue: "High memory usage"
```bash
# Reduce workers
export WORKERS=2
gunicorn --config gunicorn.conf.py wsgi:app
```

### Issue: "Port already in use"
```bash
# Use different port
export PORT=8000
gunicorn --config gunicorn.conf.py wsgi:app
```

## Maintenance

### Update Model
```bash
# Copy new checkpoint
cp new_model.pth.tar temp/best.pth.tar

# Restart service
sudo systemctl restart luna-chess
# or
docker-compose restart
```

### Update Code
```bash
# Pull latest code
git pull

# Restart service
sudo systemctl restart luna-chess
```

### Backup
```bash
# Backup checkpoints
tar -czf luna-backup.tar.gz temp/

# Backup logs
tar -czf logs-backup.tar.gz /var/log/luna-chess/
```

## Production Checklist

- [ ] Model checkpoint trained and placed in `temp/`
- [ ] Environment variables configured
- [ ] Gunicorn installed and configured
- [ ] Nginx reverse proxy setup (optional)
- [ ] SSL certificate installed (if public)
- [ ] Firewall rules configured
- [ ] Systemd service created and enabled
- [ ] Health check endpoint verified
- [ ] Logs configured and rotating
- [ ] Monitoring setup (optional)
- [ ] Backup strategy in place
- [ ] Documentation for team

## Support

For issues or questions:
- Check logs: `journalctl -u luna-chess -f`
- Test endpoints: `curl http://localhost:5000/model_info`
- Review documentation: `WEB_UI_README.md`

## License

Same as main ChessRL project.
