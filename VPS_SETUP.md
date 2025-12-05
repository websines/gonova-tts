# VPS Reverse Proxy Setup Guide

Complete guide to setting up a VPS as a reverse proxy for your voice agent services.

## Architecture

```
Internet
  ↓
VPS (Public IP, ports 80/443 open)
  ↓ SSH Reverse Tunnel (encrypted)
Your WSL Server (behind NAT/firewall)
  ↓
nginx Load Balancer
  ↓
ASR & TTS Services (GPU-accelerated)
```

## Prerequisites

- VPS with public IP ($3-6/month)
- SSH access to VPS
- Your WSL server running ASR/TTS services
- Domain names: asr.gonova.one, tts.gonova.one

## Part 1: VPS Setup

### 1.1 Rent a VPS

**Recommended providers** (choose closest to your location):

| Provider | Location | Cost | Specs |
|----------|----------|------|-------|
| [Vultr](https://vultr.com) | Multiple regions | $3.50/mo | 512MB RAM, 1 CPU |
| [Hetzner](https://hetzner.com) | EU, US | €3.79/mo | 2GB RAM, 1 CPU |
| [DigitalOcean](https://digitalocean.com) | Multiple regions | $4/mo | 512MB RAM, 1 CPU |
| [Linode](https://linode.com) | Multiple regions | $5/mo | 1GB RAM, 1 CPU |

**Choose VPS in same region as your WSL server for lowest latency!**

### 1.2 Initial VPS Configuration

```bash
# SSH into your VPS
ssh root@<VPS_IP>

# Update system
apt update && apt upgrade -y

# Install nginx
apt install -y nginx

# Install certbot for SSL (optional but recommended)
apt install -y certbot python3-certbot-nginx

# Create a non-root user
adduser tunnel
usermod -aG sudo tunnel

# Setup SSH key authentication
mkdir -p /home/tunnel/.ssh
chmod 700 /home/tunnel/.ssh
# Copy your WSL public key to /home/tunnel/.ssh/authorized_keys
chmod 600 /home/tunnel/.ssh/authorized_keys
chown -R tunnel:tunnel /home/tunnel/.ssh
```

### 1.3 Configure nginx on VPS

Create `/etc/nginx/nginx.conf`:

```nginx
user www-data;
worker_processes auto;
pid /run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    # Basic settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # WebSocket support
    map $http_upgrade $connection_upgrade {
        default upgrade;
        '' close;
    }

    # Logging
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    # ASR Service - asr.gonova.one
    server {
        listen 80;
        server_name asr.gonova.one;

        # WebSocket endpoint
        location /v1/stream/asr {
            proxy_pass http://localhost:8080;

            # WebSocket headers
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;

            # Preserve client info
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts for streaming
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;

            # Disable buffering
            proxy_buffering off;
        }

        # Health check
        location /health {
            proxy_pass http://localhost:8080;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
        }

        # Metrics
        location /metrics {
            proxy_pass http://localhost:8080;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
        }
    }

    # TTS Service - tts.gonova.one
    server {
        listen 80;
        server_name tts.gonova.one;

        # WebSocket endpoint
        location /v1/stream/tts {
            proxy_pass http://localhost:8081;

            # WebSocket headers
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;

            # Preserve client info
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;

            # Disable buffering
            proxy_buffering off;
        }

        # Health check
        location /health {
            proxy_pass http://localhost:8081;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
        }

        # Metrics
        location /metrics {
            proxy_pass http://localhost:8081;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
        }
    }
}
```

```bash
# Test nginx config
nginx -t

# Restart nginx
systemctl restart nginx
```

## Part 2: WSL Setup

### 2.1 Install autossh on WSL

```bash
# In your WSL terminal
sudo apt update
sudo apt install -y autossh
```

### 2.2 Generate SSH key (if you haven't)

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "wsl-tunnel"

# Copy public key
cat ~/.ssh/id_ed25519.pub
# Copy this output and add it to VPS /home/tunnel/.ssh/authorized_keys
```

### 2.3 Test SSH connection

```bash
# Test connection to VPS
ssh tunnel@<VPS_IP>

# If it works, exit and continue
exit
```

### 2.4 Create SSH tunnel script

Create `/home/<your-user>/tunnel_start.sh`:

```bash
#!/bin/bash
# SSH Reverse Tunnel for Voice Agent

VPS_IP="<YOUR_VPS_IP>"
VPS_USER="tunnel"
SSH_KEY="$HOME/.ssh/id_ed25519"

# Start autossh with reverse tunnels
# Port 8080 on VPS -> Port 80 on WSL (ASR)
# Port 8081 on VPS -> Port 80 on WSL (TTS)
autossh -M 0 \
    -o "ServerAliveInterval 30" \
    -o "ServerAliveCountMax 3" \
    -o "StrictHostKeyChecking=no" \
    -o "ExitOnForwardFailure yes" \
    -i "$SSH_KEY" \
    -N \
    -R 8080:localhost:80 \
    -R 8081:localhost:80 \
    "$VPS_USER@$VPS_IP"
```

```bash
# Make executable
chmod +x ~/tunnel_start.sh
```

### 2.5 Create systemd service for tunnel

Create `/etc/systemd/system/vps-tunnel.service`:

```ini
[Unit]
Description=SSH Reverse Tunnel to VPS
After=network.target

[Service]
Type=simple
User=<YOUR_WSL_USERNAME>
ExecStart=/home/<YOUR_WSL_USERNAME>/tunnel_start.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable vps-tunnel
sudo systemctl start vps-tunnel

# Check status
sudo systemctl status vps-tunnel
```

## Part 3: DNS Configuration

### 3.1 Update DNS Records

**Option A: Direct DNS (No Cloudflare proxy)**

Point your domains directly to VPS IP:

```
Type: A
Name: asr
Content: <VPS_IP>
Proxy: OFF (gray cloud)

Type: A
Name: tts
Content: <VPS_IP>
Proxy: OFF (gray cloud)
```

**Option B: With Cloudflare proxy (Recommended)**

Point domains to VPS, but keep Cloudflare proxy ON:

```
Type: A
Name: asr
Content: <VPS_IP>
Proxy: ON (orange cloud)

Type: A
Name: tts
Content: <VPS_IP>
Proxy: ON (orange cloud)
```

Then in Cloudflare SSL/TLS settings:
- Set to "Flexible" (Cloudflare → VPS uses HTTP)

### 3.2 Wait for DNS propagation

```bash
# Check DNS (wait 1-5 minutes)
dig +short asr.gonova.one
dig +short tts.gonova.one

# Should return your VPS IP
```

## Part 4: Testing

### 4.1 Test from VPS

```bash
# SSH into VPS
ssh tunnel@<VPS_IP>

# Test if tunnels are working
curl http://localhost:8080/health
curl http://localhost:8081/health

# Should return JSON health responses
```

### 4.2 Test from internet

```bash
# From any computer
curl http://asr.gonova.one/health
curl http://tts.gonova.one/health

# Or with HTTPS if using Cloudflare
curl https://asr.gonova.one/health
curl https://tts.gonova.one/health
```

## Part 5: Optional SSL Setup (Direct to VPS)

If NOT using Cloudflare, set up Let's Encrypt on VPS:

```bash
# On VPS
sudo certbot --nginx -d asr.gonova.one -d tts.gonova.one

# Follow prompts
# Certbot will automatically configure nginx for HTTPS
```

## Monitoring & Management

### Check tunnel status (WSL)

```bash
# Check if tunnel is running
sudo systemctl status vps-tunnel

# View tunnel logs
sudo journalctl -u vps-tunnel -f

# Restart tunnel
sudo systemctl restart vps-tunnel
```

### Check nginx on VPS

```bash
# SSH into VPS
ssh tunnel@<VPS_IP>

# Check nginx status
systemctl status nginx

# View nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log

# Restart nginx
systemctl restart nginx
```

### Monitor tunnel connections

```bash
# On VPS, check active SSH tunnels
netstat -tlnp | grep 8080
netstat -tlnp | grep 8081

# Should show sshd listening on localhost:8080 and :8081
```

## Troubleshooting

### Tunnel disconnects frequently

```bash
# Increase SSH timeouts in tunnel_start.sh
-o "ServerAliveInterval 15"  # More frequent keepalives
-o "ServerAliveCountMax 6"   # More retries
```

### Services not responding

```bash
# Check if nginx is running on WSL
curl http://localhost:80/health -H "Host: asr.gonova.one"

# Check if tunnel is established on VPS
ssh tunnel@<VPS_IP> "netstat -tlnp | grep 8080"

# Check nginx on VPS
ssh tunnel@<VPS_IP> "systemctl status nginx"
```

### High latency

```bash
# Test latency from WSL to VPS
ping <VPS_IP>

# Should be < 50ms for same region
# If > 100ms, consider VPS in closer location
```

## Cost Summary

| Item | Monthly Cost |
|------|-------------|
| VPS (Vultr/Hetzner) | $3.50 - $6.00 |
| Domain (if new) | ~$1.00 |
| Cloudflare (optional) | Free |
| **Total** | **$4 - $7/month** |

## Performance

**Expected latency added:**
- VPS same city: +5-15ms
- VPS same country: +15-30ms
- VPS same continent: +30-60ms

**Total latency (typical):**
- Network: 50-100ms
- ASR processing: 200-400ms
- TTS processing: 300-600ms
- **Total: 550-1100ms** (acceptable for voice)

## Security Considerations

1. **Firewall on VPS**: Only open ports 22, 80, 443
2. **SSH key authentication**: Disable password auth
3. **Regular updates**: Keep VPS updated
4. **Monitor logs**: Watch for suspicious activity
5. **Rate limiting**: Add to nginx if needed

## Advantages vs Cloudflare Tunnel

✅ Full control over infrastructure
✅ Can add custom middleware on VPS
✅ Can use for other services too
✅ Learn valuable devops skills
✅ Standard production architecture

## Next Steps

1. ✓ Rent VPS in your region
2. ✓ Configure nginx on VPS
3. ✓ Set up SSH tunnel from WSL
4. ✓ Update DNS records
5. ✓ Test services
6. ✓ Monitor and optimize

---

**Questions?** Check the troubleshooting section or review logs.
