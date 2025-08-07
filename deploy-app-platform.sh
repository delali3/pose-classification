#!/bin/bash

# DigitalOcean App Platform deployment script
echo "🚀 Deploying to DigitalOcean App Platform..."

# Install doctl if not present
if ! command -v doctl &> /dev/null; then
    echo "📦 Installing doctl..."
    curl -sL https://github.com/digitalocean/doctl/releases/download/v1.98.0/doctl-1.98.0-linux-amd64.tar.gz | tar -xzv
    sudo mv doctl /usr/local/bin
fi

# Authenticate (you'll need to set DIGITALOCEAN_ACCESS_TOKEN environment variable)
echo "🔑 Authenticating with DigitalOcean..."
doctl auth init

# Create app from spec
echo "🏗️ Creating/updating app..."
doctl apps create --spec .do/app.yaml

echo "✅ Deployment submitted to App Platform!"
echo "🌐 Check your app status at: https://cloud.digitalocean.com/apps"
echo "📊 Your app will be available at: https://ghprofit.com"
