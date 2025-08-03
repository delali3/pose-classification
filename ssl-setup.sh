#!/bin/bash

# Quick SSL setup using nip.io (free wildcard DNS)
set -e

echo "🔒 Setting up SSL with nip.io..."

# Use nip.io domain
DROPLET_IP="159.223.131.64"
DOMAIN="pose-app.159.223.131.64.nip.io"

echo "🌐 Using domain: pose-app.159.223.131.64.nip.io"

# Update nginx config with the nip.io domain
sudo tee /etc/nginx/sites-available/pose-classification > /dev/null <<EOF
server {
    listen 80;
    server_name pose-app.159.223.131.64.nip.io;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        
        # WebSocket support for Streamlit
        proxy_set_header Connection "upgrade";
        proxy_set_header Upgrade \$http_upgrade;
    }
    
    # Handle Streamlit's _stcore endpoints
    location /_stcore/ {
        proxy_pass http://localhost:8501/_stcore/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

# Enable the site
sudo ln -sf /etc/nginx/sites-available/pose-classification /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Install SSL certificate
echo "🔒 Installing SSL certificate..."
sudo certbot --nginx -d pose-app.159.223.131.64.nip.io --non-interactive --agree-tos --email nusetorfoster77@gmail.com

echo "✅ SSL setup complete!"
echo "🌍 Your app is now accessible at: https://pose-app.159.223.131.64.nip.io"
echo "🔓 HTTP redirect: http://pose-app.159.223.131.64.nip.io"
