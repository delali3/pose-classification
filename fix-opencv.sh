#!/bin/bash

echo "🔧 Quick Fix for OpenGL/OpenCV Issue"

# Stop current container
echo "⏹️ Stopping current container..."
docker stop pose-classification 2>/dev/null || true
docker rm pose-classification 2>/dev/null || true

# Create a fixed Dockerfile with OpenGL support
echo "📝 Creating fixed Dockerfile..."
cat > Dockerfile.fixed << 'EOF'
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies including OpenGL libraries
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages with explicit headless OpenCV
RUN pip install --no-cache-dir \
    streamlit \
    numpy \
    pandas \
    plotly \
    scikit-learn \
    joblib \
    pygame \
    Pillow

# Install OpenCV headless (no GUI dependencies)
RUN pip install --no-cache-dir opencv-python-headless

# Install PyTorch CPU version
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Ultralytics
RUN pip install --no-cache-dir ultralytics

# Copy application files
COPY app.py pose_classifier.pkl yolo11n-pose.pt ./

# Create directories and dummy sound file
RUN mkdir -p sound images/good images/bad && \
    touch sound/sound.wav

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

# Build the fixed container
echo "🏗️ Building fixed container..."
docker build -f Dockerfile.fixed -t pose-app-fixed .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Run the fixed container
    echo "🚀 Starting fixed container..."
    docker run -d \
        --name pose-classification \
        -p 8501:8501 \
        --restart unless-stopped \
        pose-app-fixed
    
    # Wait and check
    sleep 15
    
    echo "📊 Container status:"
    docker ps -f name=pose-classification
    
    echo ""
    echo "📋 Recent logs:"
    docker logs pose-classification --tail 10
    
    echo ""
    echo "🧪 Testing health..."
    sleep 5
    if curl -f http://localhost:8501/_stcore/health 2>/dev/null; then
        echo "✅ Health check passed!"
        echo ""
        echo "🎉 Fix successful!"
        echo "🌐 App accessible at: http://68.183.113.248:8501"
    else
        echo "❌ Health check failed. Check logs:"
        docker logs pose-classification --tail 20
    fi
else
    echo "❌ Build failed. Check the error messages above."
fi
