#!/bin/bash
# Diagnose nginx routing issues

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Nginx Diagnostics${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 1. Check if nginx is running
echo -e "${YELLOW}1. Checking if nginx is running...${NC}"
if pgrep -x nginx > /dev/null; then
    echo -e "   ${GREEN}✓ nginx is running${NC}"
    ps aux | grep "[n]ginx" | head -5
else
    echo -e "   ${RED}✗ nginx is NOT running${NC}"
    echo "   Start it with: sudo systemctl start nginx"
fi
echo ""

# 2. Check which config nginx is using
echo -e "${YELLOW}2. Checking nginx config...${NC}"
if command -v nginx &> /dev/null; then
    nginx -V 2>&1 | grep "configure arguments"
    echo ""
    echo "   Default config: /etc/nginx/nginx.conf"
    if [ -f /etc/nginx/nginx.conf ]; then
        echo -e "   ${GREEN}✓ Config exists${NC}"
    else
        echo -e "   ${RED}✗ Config missing${NC}"
    fi
else
    echo -e "   ${RED}✗ nginx command not found${NC}"
fi
echo ""

# 3. Test nginx config
echo -e "${YELLOW}3. Testing nginx config...${NC}"
sudo nginx -t 2>&1 | head -10
echo ""

# 4. Check if ports 80/443 are listening
echo -e "${YELLOW}4. Checking nginx ports...${NC}"
echo "   Port 80 (HTTP):"
if sudo netstat -tlnp 2>/dev/null | grep ":80 " | grep nginx; then
    echo -e "   ${GREEN}✓ nginx listening on port 80${NC}"
else
    echo -e "   ${RED}✗ nginx NOT listening on port 80${NC}"
fi
echo ""

# 5. Test local backend services (bypass nginx)
echo -e "${YELLOW}5. Testing backend services directly...${NC}"
echo "   ASR port 8001:"
if curl -s -f http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓ ASR instance 1 responding${NC}"
    curl -s http://localhost:8001/health | head -1
else
    echo -e "   ${RED}✗ ASR instance 1 not responding${NC}"
fi
echo ""

echo "   ASR port 8011:"
if curl -s -f http://localhost:8011/health > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓ ASR instance 2 responding${NC}"
    curl -s http://localhost:8011/health | head -1
else
    echo -e "   ${RED}✗ ASR instance 2 not responding${NC}"
fi
echo ""

echo "   TTS port 8002:"
if curl -s -f http://localhost:8002/health > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓ TTS instance 1 responding${NC}"
    curl -s http://localhost:8002/health | head -1
else
    echo -e "   ${RED}✗ TTS instance 1 not responding${NC}"
fi
echo ""

# 6. Test nginx routing (localhost)
echo -e "${YELLOW}6. Testing nginx routing (localhost with Host header)...${NC}"
echo "   Testing ASR route:"
RESPONSE=$(curl -s -w "\n%{http_code}" -H "Host: asr.gonova.one" http://localhost/health 2>&1)
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "   ${GREEN}✓ ASR route working (HTTP $HTTP_CODE)${NC}"
    echo "   Response: $BODY"
else
    echo -e "   ${RED}✗ ASR route failed (HTTP $HTTP_CODE)${NC}"
    echo "   Response: $BODY"
fi
echo ""

# 7. Check nginx error logs
echo -e "${YELLOW}7. Recent nginx error logs:${NC}"
if [ -f /var/log/nginx/error.log ]; then
    sudo tail -20 /var/log/nginx/error.log | tail -10
else
    echo -e "   ${YELLOW}No error log found${NC}"
fi
echo ""

# 8. Check if using correct nginx config
echo -e "${YELLOW}8. Checking active nginx config...${NC}"
if [ -f /etc/nginx/nginx.conf ]; then
    echo "   Checking for upstream definitions:"
    if sudo grep -q "upstream asr_backend" /etc/nginx/nginx.conf; then
        echo -e "   ${GREEN}✓ ASR upstream defined${NC}"
    else
        echo -e "   ${RED}✗ ASR upstream NOT defined${NC}"
    fi

    if sudo grep -q "upstream tts_backend" /etc/nginx/nginx.conf; then
        echo -e "   ${GREEN}✓ TTS upstream defined${NC}"
    else
        echo -e "   ${RED}✗ TTS upstream NOT defined${NC}"
    fi

    echo ""
    echo "   Checking for server blocks:"
    if sudo grep -q "server_name asr.gonova.one" /etc/nginx/nginx.conf; then
        echo -e "   ${GREEN}✓ asr.gonova.one server block found${NC}"
    else
        echo -e "   ${RED}✗ asr.gonova.one server block NOT found${NC}"
    fi

    if sudo grep -q "server_name tts.gonova.one" /etc/nginx/nginx.conf; then
        echo -e "   ${GREEN}✓ tts.gonova.one server block found${NC}"
    else
        echo -e "   ${RED}✗ tts.gonova.one server block NOT found${NC}"
    fi
fi
echo ""

# 9. Summary and recommendations
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "If nginx routing is failing, you may need to:"
echo ""
echo "1. Copy the correct config:"
echo "   sudo cp config/nginx-cloudflare.conf /etc/nginx/nginx.conf"
echo ""
echo "2. Test the config:"
echo "   sudo nginx -t"
echo ""
echo "3. Restart nginx:"
echo "   sudo systemctl restart nginx"
echo ""
echo "4. Check if Cloudflare is properly configured:"
echo "   - DNS: asr.gonova.one -> Your server IP"
echo "   - SSL/TLS mode: Flexible (Cloudflare handles SSL)"
echo ""
