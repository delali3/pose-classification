#!/bin/bash

echo "🚀 Quick Deploy - Pose Classification App"
echo "DigitalOcean Droplet: 68.183.113.248"

# Clean up
echo "🧹 Cleaning up..."
docker stop pose-classification 2>/dev/null || true
docker rm pose-classification 2>/dev/null || true
docker system prune -f

# Build with simple Dockerfile
echo "🏗️ Building with simple Dockerfile..."
docker build -f Dockerfile.simple -t pose-app .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Run the container
    echo "🚀 Starting container..."
    docker run -d \
        --name pose-classification \
        -p 8501:8501 \
        --restart unless-stopped \
        pose-app
    
    # Wait a moment
    sleep 10
    
    # Check status
    echo "📊 Container status:"
    docker ps -f name=pose-classification
    
    echo "📋 Container logs:"
    docker logs pose-classification --tail 20
    
    # Test the app
    echo "🧪 Testing the app..."
    sleep 5
    curl -f http://localhost:8501/_stcore/health && echo "✅ Health check passed!" || echo "❌ Health check failed"
    
    echo ""
    echo "🎉 Deployment complete!"
    echo "🌐 Your app should be accessible at:"
    echo "   http://68.183.113.248:8501"
    echo ""
    echo "📊 Useful commands:"
    echo "   docker logs pose-classification -f"
    echo "   docker restart pose-classification"
    echo "   docker stop pose-classification"
    
else
    echo "❌ Build failed. Let's try the emergency approach..."
    
    # Emergency Dockerfile
    cat > Dockerfile.emergency << 'EOF'
FROM python:3.9-slim
WORKDIR /app

# Install system dependencies including OpenGL libraries
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages - use headless OpenCV
RUN pip install streamlit numpy pandas plotly scikit-learn joblib pygame
RUN pip install opencv-python-headless Pillow
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install ultralytics

COPY app.py pose_classifier.pkl yolo11n-pose.pt ./
RUN mkdir -p sound images/good images/bad && touch sound/sound.wav
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

    echo "🚨 Building emergency version..."
    docker build -f Dockerfile.emergency -t pose-app .
    
    if [ $? -eq 0 ]; then
        echo "✅ Emergency build successful!"
        docker run -d --name pose-classification -p 8501:8501 --restart unless-stopped pose-app
        sleep 10
        docker ps -f name=pose-classification
        curl -f http://localhost:8501/_stcore/health && echo "✅ Emergency deployment successful!"
    else
        echo "❌ All builds failed. Check the logs above for errors."
    fi
fi
