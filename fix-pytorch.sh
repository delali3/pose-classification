#!/bin/bash

# Fix PyTorch 2.6+ model loading issue
set -e

echo "🔧 Fixing PyTorch model loading issue..."

# Stop the current container
docker-compose down

echo "📝 Updating requirements.txt to use compatible PyTorch version..."

# Update requirements.txt to use PyTorch 2.1.0 (more stable for YOLO)
cat > requirements.txt << 'EOF'
torch==2.1.0
torchvision==0.16.0
ultralytics==8.0.196
streamlit==1.28.1
numpy==1.24.3
opencv-python==4.8.1.78
Pillow==10.0.1
scikit-learn==1.3.0
joblib==1.3.2
pandas==2.0.3
plotly==5.17.0
pygame==2.5.2
EOF

echo "🏗️ Rebuilding Docker container with fixed PyTorch version..."
docker-compose build --no-cache

echo "🚀 Starting the application..."
docker-compose up -d

echo "⏳ Waiting for application to start..."
sleep 10

echo "📊 Checking container status..."
docker-compose ps

echo "📋 Checking logs..."
docker-compose logs --tail=20

echo "✅ Fix applied!"
echo "🌍 Your app should now be accessible at:"
echo "   HTTP: http://pose-app.159.223.131.64.nip.io"
echo "   HTTPS: https://pose-app.159.223.131.64.nip.io"
echo ""
echo "🔍 To monitor logs: docker-compose logs -f"
