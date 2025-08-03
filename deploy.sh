#!/bin/bash

# Deployment script for pose-classification app
set -e

echo "🚀 Starting deployment of pose-classification app..."

# Variables
APP_DIR="/home/$USER/pose-classification"
DOMAIN="your-domain.com"  # Replace with your actual domain

# Create app directory
echo "📁 Creating application directory..."
mkdir -p $APP_DIR
cd $APP_DIR

# Clone or update the repository (replace with your repo URL)
if [ ! -d ".git" ]; then
    echo "📥 Cloning repository..."
    # git clone https://github.com/yourusername/pose-classification.git .
    echo "Please upload your files to $APP_DIR"
else
    echo "🔄 Updating repository..."
    git pull origin main
fi

# Build and start the application
echo "🏗️ Building Docker container..."
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Configure Nginx
echo "🌐 Configuring Nginx..."
sudo cp nginx.conf /etc/nginx/sites-available/pose-classification
sudo ln -sf /etc/nginx/sites-available/pose-classification /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Setup SSL certificate (optional but recommended)
if [ "$DOMAIN" != "your-domain.com" ]; then
    echo "🔒 Setting up SSL certificate..."
    sudo certbot --nginx -d $DOMAIN
fi

# Setup firewall
echo "🔥 Configuring firewall..."
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

echo "✅ Deployment complete!"
echo "🌍 Your app should be accessible at: http://$DOMAIN"
echo "📊 Check logs with: docker-compose logs -f"
echo "🔄 To restart: docker-compose restart"
