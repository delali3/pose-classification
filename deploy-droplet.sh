#!/bin/bash

# Droplet deployment script for pose-classification app
set -e

echo "🚀 Starting deployment of pose-classification app on DigitalOcean Droplet..."

# Variables
APP_DIR="/root/pose-classification"
DOMAIN="ghprofit.com"
DOCKER_COMPOSE_FILE="docker-compose.droplet.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root (use sudo)"
    exit 1
fi

# Navigate to app directory
print_status "📁 Navigating to app directory: $APP_DIR"
cd $APP_DIR

# Update repository
print_status "🔄 Updating repository..."
git pull origin main

# Clean up Docker to free space
print_status "🧹 Cleaning up Docker to free space..."
docker system prune -af --volumes || true
docker builder prune -af || true

# Create necessary directories
print_status "📁 Creating necessary directories..."
mkdir -p logs data sound images/good images/bad

# Stop existing containers
print_status "⏹️ Stopping existing containers..."
docker-compose -f $DOCKER_COMPOSE_FILE down || true

# Build with no cache to ensure fresh build
print_status "🏗️ Building Docker container (this may take several minutes)..."
docker-compose -f $DOCKER_COMPOSE_FILE build --no-cache

# Start the application
print_status "🚀 Starting the application..."
docker-compose -f $DOCKER_COMPOSE_FILE up -d

# Wait for container to be healthy
print_status "⏳ Waiting for application to be ready..."
sleep 30

# Check if container is running
if docker-compose -f $DOCKER_COMPOSE_FILE ps | grep -q "Up"; then
    print_status "✅ Container is running"
else
    print_error "❌ Container failed to start"
    docker-compose -f $DOCKER_COMPOSE_FILE logs
    exit 1
fi

# Configure Nginx
print_status "🌐 Configuring Nginx..."

# Backup existing nginx config if it exists
if [ -f /etc/nginx/sites-enabled/default ]; then
    print_status "📋 Backing up default nginx config..."
    mv /etc/nginx/sites-enabled/default /etc/nginx/sites-enabled/default.backup
fi

# Copy new nginx configuration
cp nginx.droplet.conf /etc/nginx/sites-available/pose-classification
ln -sf /etc/nginx/sites-available/pose-classification /etc/nginx/sites-enabled/

# Test nginx configuration
print_status "🧪 Testing Nginx configuration..."
if nginx -t; then
    print_status "✅ Nginx configuration is valid"
    systemctl reload nginx
else
    print_error "❌ Nginx configuration is invalid"
    exit 1
fi

# Setup firewall
print_status "🔥 Configuring firewall..."
ufw allow 22/tcp   # SSH
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS
ufw --force enable

# Setup SSL certificate if domain is not placeholder
if [ "$DOMAIN" != "your-domain.com" ] && [ "$DOMAIN" != "example.com" ]; then
    print_status "🔒 Setting up SSL certificate for $DOMAIN..."
    
    # Check if domain resolves to this server
    DOMAIN_IP=$(dig +short $DOMAIN | tail -n1)
    SERVER_IP=$(curl -s ifconfig.me)
    
    if [ "$DOMAIN_IP" = "$SERVER_IP" ]; then
        print_status "✅ Domain DNS is correctly configured"
        certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN || print_warning "SSL setup failed, continuing without SSL"
    else
        print_warning "⚠️ Domain $DOMAIN does not point to this server ($SERVER_IP). Skipping SSL setup."
        print_warning "Please update your DNS records to point to $SERVER_IP"
    fi
else
    print_warning "⚠️ Using placeholder domain. Update DOMAIN variable for SSL setup."
fi

# Final status check
print_status "🔍 Performing final health check..."
sleep 10

if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    print_status "✅ Application health check passed"
else
    print_warning "⚠️ Application health check failed, but deployment may still work"
fi

# Display final information
echo ""
echo "========================================"
echo "🎉 DEPLOYMENT COMPLETE!"
echo "========================================"
echo ""
print_status "🌍 Your app should be accessible at:"
if [ "$DOMAIN" != "your-domain.com" ] && [ "$DOMAIN" != "example.com" ]; then
    echo "   • https://$DOMAIN (if SSL was configured)"
    echo "   • http://$DOMAIN"
else
    echo "   • http://$(curl -s ifconfig.me):80"
fi
echo ""
print_status "📊 Useful commands:"
echo "   • Check logs: docker-compose -f $DOCKER_COMPOSE_FILE logs -f"
echo "   • Restart app: docker-compose -f $DOCKER_COMPOSE_FILE restart"
echo "   • Stop app: docker-compose -f $DOCKER_COMPOSE_FILE down"
echo "   • Update app: git pull && $0"
echo ""
print_status "🔧 Troubleshooting:"
echo "   • Check container status: docker-compose -f $DOCKER_COMPOSE_FILE ps"
echo "   • Check nginx status: systemctl status nginx"
echo "   • Check application logs: docker-compose -f $DOCKER_COMPOSE_FILE logs pose-app"
echo ""

# Show current status
print_status "📈 Current Status:"
docker-compose -f $DOCKER_COMPOSE_FILE ps
