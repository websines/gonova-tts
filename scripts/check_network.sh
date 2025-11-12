#!/bin/bash
# Check network connectivity and firewall

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Network & Firewall Diagnostics${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 1. Check if port 80 is actually listening (alternative methods)
echo -e "${YELLOW}1. Checking port 80 (alternative methods)...${NC}"
echo "   Using ss:"
sudo ss -tlnp | grep ":80 "
echo ""
echo "   Using lsof:"
sudo lsof -i :80 2>/dev/null || echo "   lsof not available"
echo ""

# 2. Check firewall status
echo -e "${YELLOW}2. Checking firewall...${NC}"
if command -v ufw &> /dev/null; then
    echo "   UFW status:"
    sudo ufw status
elif command -v iptables &> /dev/null; then
    echo "   iptables rules:"
    sudo iptables -L -n | grep "80"
else
    echo "   No firewall tool found"
fi
echo ""

# 3. Test local connection to port 80
echo -e "${YELLOW}3. Testing connection to localhost:80...${NC}"
if curl -s -f http://localhost/ > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓ Can connect to localhost:80${NC}"
else
    echo -e "   ${RED}✗ Cannot connect to localhost:80${NC}"
fi
echo ""

# 4. Test with actual domain (from server)
echo -e "${YELLOW}4. Testing domain from server...${NC}"
echo "   Testing asr.gonova.one/health:"
RESPONSE=$(curl -s -w "\n%{http_code}" https://asr.gonova.one/health 2>&1)
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "   ${GREEN}✓ Working (HTTP $HTTP_CODE)${NC}"
else
    echo -e "   ${RED}✗ Failed (HTTP $HTTP_CODE)${NC}"
fi
echo "   Response: $BODY"
echo ""

# 5. Check DNS resolution
echo -e "${YELLOW}5. Checking DNS...${NC}"
echo "   asr.gonova.one resolves to:"
dig +short asr.gonova.one || nslookup asr.gonova.one | grep "Address:" | tail -1
echo ""
echo "   tts.gonova.one resolves to:"
dig +short tts.gonova.one || nslookup tts.gonova.one | grep "Address:" | tail -1
echo ""

# 6. Check server's public IP
echo -e "${YELLOW}6. Server's public IP:${NC}"
curl -s ifconfig.me || curl -s icanhazip.com
echo ""
echo ""

# 7. Test if port 80 is reachable from outside
echo -e "${YELLOW}7. Testing external connectivity...${NC}"
echo "   Attempting to connect to port 80 from this server:"
timeout 5 bash -c "echo test | nc -w 2 localhost 80" 2>&1 && echo -e "   ${GREEN}✓ Port 80 accepts connections${NC}" || echo -e "   ${RED}✗ Port 80 connection failed${NC}"
echo ""

# 8. Summary
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Troubleshooting Steps${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "If the domain still doesn't work:"
echo ""
echo "1. Check Cloudflare DNS:"
echo "   - Go to Cloudflare dashboard"
echo "   - Verify A records:"
echo "     asr.gonova.one -> Your server IP"
echo "     tts.gonova.one -> Your server IP"
echo "   - Set proxy status (orange cloud)"
echo ""
echo "2. Check Cloudflare SSL/TLS settings:"
echo "   - Go to SSL/TLS > Overview"
echo "   - Set to 'Flexible' (Cloudflare handles SSL)"
echo ""
echo "3. Open firewall if needed:"
echo "   sudo ufw allow 80/tcp"
echo "   sudo ufw allow 443/tcp"
echo ""
echo "4. Restart nginx:"
echo "   sudo systemctl restart nginx"
echo ""
