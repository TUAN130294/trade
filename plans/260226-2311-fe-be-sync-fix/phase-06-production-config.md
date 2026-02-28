# Phase 06: Production Deployment Config

**Priority:** P1 (HIGH)
**Effort:** 1h
**Status:** Pending

## Context Links
- Frontend: `vn-quant-web/src/App.jsx` (line 5: const API_URL = '/api')
- Backend: `run_autonomous_paper_trading.py` (line 85-96: CORS localhost only)
- Docs: `docs/deployment-guide.md` (Docker deployment)
- Audit: "Production deployment config - Add .env.production with API_URL, configure nginx"

## Overview
Current setup hardcodes localhost, works only in dev mode. Production needs:
- Environment-specific API URLs (dev vs prod)
- Nginx reverse proxy config
- CORS properly configured
- Build-time environment variables

**Goal:** Create production-ready deployment config for VPS/cloud hosting.

## Key Insights
- Use Vite's `import.meta.env` for build-time env vars
- Create `.env.production` for prod API URL
- Nginx serves FE static files + proxies /api to BE
- Backend CORS allows production domain
- Keep localhost for dev mode

## Requirements

### Functional
- Dev mode: FE calls http://localhost:8100/api
- Prod mode: FE calls https://yourdomain.com/api (nginx proxies to BE)
- Environment switcher via VITE_API_URL
- Nginx config serves FE + proxies BE

### Non-Functional
- Zero code changes needed between dev and prod
- Build once, deploy anywhere (via env vars)
- Nginx handles SSL termination
- Backend stays on localhost:8100 (not exposed)

## Architecture

### Development Mode
```
┌─────────────────┐
│ React Dev       │
│ localhost:5176  │──┐
└─────────────────┘  │
                     │ Direct API calls
                     ↓
┌─────────────────┐
│ FastAPI         │
│ localhost:8100  │
└─────────────────┘
```

### Production Mode
```
┌─────────────────┐
│ User Browser    │
│ yourdomain.com  │
└────────┬────────┘
         │ HTTPS
         ↓
┌─────────────────────────────┐
│ Nginx (Port 80/443)         │
│ ├─ / → Static files (FE)    │
│ └─ /api → Proxy to BE       │
└────────┬────────────────────┘
         │ /api/* → localhost:8100
         ↓
┌─────────────────┐
│ FastAPI         │
│ localhost:8100  │
└─────────────────┘
```

## Related Code Files

**Frontend (Create):**
- `vn-quant-web/.env.production` - Production env vars

**Frontend (Modify):**
- `vn-quant-web/src/App.jsx` - Use import.meta.env.VITE_API_URL

**Backend (Modify):**
- `run_autonomous_paper_trading.py` - Add production CORS origin from env var

**Infrastructure (Create):**
- `nginx/vn-quant.conf` - Nginx reverse proxy config
- `.env.production` (root) - Backend production env vars

## Implementation Steps

### Step 1: Create Frontend Production Env
Create `vn-quant-web/.env.production`:

```bash
# Production API URL (nginx will proxy /api to backend)
VITE_API_URL=/api

# Or if backend on different domain:
# VITE_API_URL=https://api.yourdomain.com/api
```

Create `vn-quant-web/.env.development` (for clarity):

```bash
# Development API URL (direct to backend)
VITE_API_URL=http://localhost:8100/api
```

### Step 2: Update Frontend to Use Env Var
Modify `vn-quant-web/src/App.jsx`:

```javascript
// BEFORE:
const API_URL = '/api'

// AFTER:
const API_URL = import.meta.env.VITE_API_URL || '/api'

// Also update WebSocket URL in use-websocket.js:
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8100/ws/autonomous'

// For production WebSocket through nginx:
// Use wss:// protocol and same domain
```

Update `vn-quant-web/src/hooks/use-websocket.js`:

```javascript
// Add dynamic WS URL based on env
const getWebSocketURL = () => {
  const wsUrl = import.meta.env.VITE_WS_URL

  if (wsUrl) return wsUrl

  // Auto-detect in production: use wss:// with current domain
  if (import.meta.env.PROD) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${protocol}//${window.location.host}/ws/autonomous`
  }

  // Development default
  return 'ws://localhost:8100/ws/autonomous'
}

export function useWebSocket(url = getWebSocketURL(), options = {}) {
  // ... rest of hook
}
```

### Step 3: Create Nginx Config
Create `nginx/vn-quant.conf`:

```nginx
# VN-Quant Production Nginx Config

upstream backend {
    server 127.0.0.1:8100;
}

server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    # Redirect HTTP to HTTPS (uncomment after SSL setup)
    # return 301 https://$server_name$request_uri;

    # Root directory for static files
    root /var/www/vn-quant/dist;
    index index.html;

    # Enable gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    gzip_min_length 1000;

    # API proxy to backend
    location /api/ {
        proxy_pass http://backend/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # CORS headers (if backend doesn't handle)
        add_header Access-Control-Allow-Origin $http_origin always;
        add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS' always;
        add_header Access-Control-Allow-Headers 'Content-Type, X-API-Key' always;

        # Handle preflight
        if ($request_method = OPTIONS) {
            return 204;
        }
    }

    # WebSocket proxy
    location /ws/ {
        proxy_pass http://backend/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 86400; # 24 hours for WebSocket
    }

    # Static files with caching
    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # SPA fallback - serve index.html for all other routes
    location / {
        try_files $uri $uri/ /index.html;
    }
}

# HTTPS server (uncomment after SSL setup with Let's Encrypt)
# server {
#     listen 443 ssl http2;
#     server_name yourdomain.com www.yourdomain.com;
#
#     ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
#     ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
#
#     # Same config as HTTP server above
#     root /var/www/vn-quant/dist;
#     index index.html;
#     # ... (copy all location blocks from above)
# }
```

### Step 4: Update Backend CORS for Production
Modify `run_autonomous_paper_trading.py`:

```python
import os

# CORS origins from environment
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5176,http://localhost:8100"
).split(",")

# Add production domain if in production
if os.getenv("ENVIRONMENT") == "production":
    prod_origin = os.getenv("PRODUCTION_ORIGIN", "https://yourdomain.com")
    ALLOWED_ORIGINS.append(prod_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Create `.env.production` (root):

```bash
# Backend Production Config
ENVIRONMENT=production
PRODUCTION_ORIGIN=https://yourdomain.com
ALLOWED_ORIGINS=https://yourdomain.com

# API settings
API_HOST=127.0.0.1  # Only listen on localhost (nginx proxies)
API_PORT=8100

# Authentication
API_KEY=your_production_api_key_here

# Trading settings
TRADING_MODE=paper
INITIAL_CAPITAL=100000000
```

### Step 5: Create Deployment Script
Create `scripts/deploy-production.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 VN-Quant Production Deployment"

# Build frontend
echo "📦 Building frontend..."
cd vn-quant-web
npm run build
cd ..

# Copy to nginx directory
echo "📁 Copying files to /var/www/vn-quant..."
sudo mkdir -p /var/www/vn-quant
sudo cp -r vn-quant-web/dist/* /var/www/vn-quant/

# Copy nginx config
echo "⚙️  Configuring nginx..."
sudo cp nginx/vn-quant.conf /etc/nginx/sites-available/vn-quant
sudo ln -sf /etc/nginx/sites-available/vn-quant /etc/nginx/sites-enabled/vn-quant

# Test nginx config
sudo nginx -t

# Reload nginx
echo "🔄 Reloading nginx..."
sudo systemctl reload nginx

# Restart backend service
echo "🔄 Restarting backend..."
sudo systemctl restart vn-quant-backend

echo "✅ Deployment complete!"
echo "🌐 Visit: https://yourdomain.com"
```

Create systemd service file `scripts/vn-quant-backend.service`:

```ini
[Unit]
Description=VN-Quant Autonomous Trading Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/vn-quant
Environment="PATH=/opt/vn-quant/venv/bin"
EnvironmentFile=/opt/vn-quant/.env.production
ExecStart=/opt/vn-quant/venv/bin/python run_autonomous_paper_trading.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Step 6: Test Production Build Locally
```bash
# Build frontend
cd vn-quant-web
npm run build

# Serve with simple HTTP server
npx serve -s dist -p 3000

# Start backend
cd ..
python run_autonomous_paper_trading.py

# Test API calls work through production build
curl http://localhost:3000/api/status
```

## Todo List
- [ ] Create vn-quant-web/.env.production
- [ ] Create vn-quant-web/.env.development
- [ ] Update App.jsx to use import.meta.env.VITE_API_URL
- [ ] Update WebSocket hook to use dynamic URL
- [ ] Create nginx/vn-quant.conf
- [ ] Update backend CORS to read from env
- [ ] Create .env.production (root)
- [ ] Create scripts/deploy-production.sh
- [ ] Create scripts/vn-quant-backend.service
- [ ] Test production build locally
- [ ] Document deployment steps in docs/deployment-guide.md

## Success Criteria
- [ ] Frontend builds successfully with `npm run build`
- [ ] Production build uses correct API URL from env var
- [ ] Nginx config proxies /api to backend
- [ ] WebSocket works through nginx proxy
- [ ] CORS allows production domain
- [ ] Backend runs as systemd service
- [ ] Deployment script completes without errors
- [ ] Zero code changes needed between dev and prod

## Risk Assessment
- **Risk:** CORS blocks production requests → Mitigated: Test with real domain before go-live
- **Risk:** WebSocket fails through nginx → Mitigated: Test WS proxy config separately
- **Risk:** SSL certificate issues → Mitigated: Use Let's Encrypt with certbot

## Security Considerations
- Backend binds to 127.0.0.1 only (not exposed to internet)
- Nginx handles SSL termination
- API key required for protected endpoints (Phase 01)
- Use HTTPS in production (Let's Encrypt free SSL)
- Keep .env.production out of git (add to .gitignore)

## Next Steps
After completing this phase:
1. Update plan.md with completion status
2. Proceed to Phase 07 (Modularize frontend)
3. Test full production deployment on staging server
4. Document deployment runbook in docs/
5. Set up monitoring and logging for production
