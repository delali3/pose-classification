#!/bin/bash

# Pose Classification App - Server Setup Script
set -e

echo "🚀 Setting up server for pose-classification app..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
echo "🐙 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install nginx (for reverse proxy)
echo "🌐 Installing Nginx..."
sudo apt install nginx -y

# Install certbot for SSL
echo "🔒 Installing Certbot for SSL..."
sudo apt install certbot python3-certbot-nginx -y

# Install additional utilities
echo "🛠️ Installing additional utilities..."
sudo apt install curl wget git htop unzip -y

# Start and enable services
echo "🔧 Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

echo "🔧 Starting Nginx service..."
sudo systemctl start nginx
sudo systemctl enable nginx

# Clean up
rm -f get-docker.sh

echo "✅ Installation complete!"
echo "📝 Next steps:"
echo "1. Log out and log back in for Docker permissions to take effect"
echo "2. Upload your pose-classification code to the server"
echo "3. Run the deploy.sh script to start your application"
echo ""
echo "🔍 Verify installation:"
echo "docker --version"
echo "docker-compose --version"
echo "nginx -v"
