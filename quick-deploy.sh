#!/bin/bash

# QUICK FIX DEPLOYMENT SCRIPT
# Use this if the main deployment script fails

echo "🚀 Quick Fix Deployment for Pose Classification App"

# Check available space
echo "📊 Checking available disk space..."
df -h

# Clean up aggressively
echo "🧹 Aggressive cleanup..."
docker system prune -af --volumes
apt autoremove -y
apt autoclean
journalctl --vacuum-time=1d

# Option 1: Try with simplified requirements
echo "📦 Option 1: Building with simplified requirements..."
docker build -f Dockerfile.minimal -t pose-app:simple . || echo "Option 1 failed"

# Option 2: Try with explicit CPU PyTorch
if [ $? -ne 0 ]; then
    echo "📦 Option 2: Building with explicit CPU PyTorch..."
    docker build -f Dockerfile.cpu -t pose-app:cpu . || echo "Option 2 failed"
fi

# Option 3: Manual pip install approach
if [ $? -ne 0 ]; then
    echo "📦 Option 3: Creating minimal container..."
    cat > Dockerfile.emergency << 'EOF'
FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install streamlit numpy pandas plotly opencv-python-headless Pillow scikit-learn joblib
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install ultralytics pygame
COPY app.py .
COPY pose_classifier.pkl .
COPY yolo11n-pose.pt .
RUN mkdir -p sound images/good images/bad
EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
EOF
    docker build -f Dockerfile.emergency -t pose-app:emergency .
fi

# Start the container
echo "🚀 Starting container..."
docker run -d --name pose-classification -p 8501:8080 pose-app:simple || \
docker run -d --name pose-classification -p 8501:8080 pose-app:cpu || \
docker run -d --name pose-classification -p 8501:8080 pose-app:emergency

echo "✅ Check if container is running:"
docker ps

echo "📊 Check logs:"
docker logs pose-classification

echo "🌐 Test the app:"
echo "curl http://localhost:8501/_stcore/health"
