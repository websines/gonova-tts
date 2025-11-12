#!/bin/bash
# Setup SSL certificates for asr.gonova.one and tts.gonova.one

set -e

echo "========================================="
echo "SSL Setup for Voice Agent"
echo "========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo)"
    exit 1
fi

# Install certbot
echo "1. Installing certbot..."
apt update
apt install -y certbot python3-certbot-nginx

echo ""
echo "2. Make sure your DNS is configured:"
echo "   asr.gonova.one  -> Your server IP"
echo "   tts.gonova.one  -> Your server IP"
echo ""
read -p "Press Enter when DNS is configured..."

# Get certificates
echo ""
echo "3. Getting SSL certificates..."
echo ""

# Get cert for asr.gonova.one
echo "Getting certificate for asr.gonova.one..."
certbot certonly --standalone \
    -d asr.gonova.one \
    --non-interactive \
    --agree-tos \
    --email admin@gonova.one \
    --http-01-port 80

echo ""

# Get cert for tts.gonova.one
echo "Getting certificate for tts.gonova.one..."
certbot certonly --standalone \
    -d tts.gonova.one \
    --non-interactive \
    --agree-tos \
    --email admin@gonova.one \
    --http-01-port 80

echo ""
echo "========================================="
echo "SSL Certificates Installed!"
echo "========================================="
echo ""
echo "Certificates location:"
echo "  asr.gonova.one: /etc/letsencrypt/live/asr.gonova.one/"
echo "  tts.gonova.one: /etc/letsencrypt/live/tts.gonova.one/"
echo ""
echo "Next steps:"
echo "  1. Copy production nginx config:"
echo "     cp config/nginx-production.conf /etc/nginx/nginx.conf"
echo ""
echo "  2. Test nginx config:"
echo "     nginx -t"
echo ""
echo "  3. Restart nginx:"
echo "     systemctl restart nginx"
echo ""
echo "  4. Enable auto-renewal:"
echo "     systemctl enable certbot.timer"
echo "     systemctl start certbot.timer"
echo ""
echo "Certificates will auto-renew before expiry."
echo ""
