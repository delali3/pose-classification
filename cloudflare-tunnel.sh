#!/bin/bash

# Setup Cloudflare Tunnel for SSL without domain
set -e

echo "🌩️ Setting up Cloudflare Tunnel..."

# Download cloudflared
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

echo "✅ Cloudflared installed!"
echo "📝 Next steps:"
echo "1. Visit https://one.dash.cloudflare.com/"
echo "2. Go to Zero Trust > Access > Tunnels"
echo "3. Create a new tunnel"
echo "4. Follow the instructions to get your tunnel token"
echo "5. Run: cloudflared service install <your-token>"
echo "6. Configure the tunnel to point to localhost:8501"
echo ""
echo "This will give you a secure HTTPS URL like: https://random-name.your-account.workers.dev"
