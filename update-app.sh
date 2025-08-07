#!/bin/bash

echo "🔄 Updating Pose Classification App..."

# Navigate to app directory
cd /root/pose-classification

# Pull latest changes from GitHub
echo "📥 Pulling latest changes..."
git pull origin main

# Stop current container
echo "⏹️ Stopping current container..."
docker stop pose-classification 2>/dev/null || true
docker rm pose-classification 2>/dev/null || true

# Rebuild with latest changes
echo "🏗️ Rebuilding container..."
docker build -f Dockerfile.simple -t pose-app .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Start updated container
    echo "🚀 Starting updated container..."
    docker run -d \
        --name pose-classification \
        -p 8501:8501 \
        --restart unless-stopped \
        pose-app
    
    # Wait and check
    sleep 10
    
    echo "📊 Container status:"
    docker ps -f name=pose-classification
    
    echo "🧪 Testing health..."
    if curl -f http://localhost:8501/_stcore/health 2>/dev/null; then
        echo "✅ Update successful!"
        echo "🌐 App accessible at: http://68.183.113.248:8501"
    else
        echo "❌ Health check failed"
        docker logs pose-classification --tail 20
    fi
else
    echo "❌ Build failed"
fi
