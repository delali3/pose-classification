#!/bin/bash

# Quick fix for the TypeError issue
set -e

echo "🔧 Fixing TypeError in app.py..."

# Stop the application
docker-compose down

# Upload the fixed app.py and requirements.txt from local machine
echo "📁 You need to upload the fixed files first:"
echo "From your local machine, run:"
echo "scp \"c:\\xampp\\htdocs\\era\\pose-classification\\app.py\" root@159.223.131.64:/root/pose-classification/"
echo "scp \"c:\\xampp\\htdocs\\era\\pose-classification\\requirements.txt\" root@159.223.131.64:/root/pose-classification/"
echo ""
echo "Press Enter when files are uploaded..."
read

# Rebuild and restart
echo "🏗️ Rebuilding container with fixes..."
docker-compose build --no-cache

echo "🚀 Starting application..."
docker-compose up -d

echo "⏳ Waiting for startup..."
sleep 15

echo "📊 Checking status..."
docker-compose ps

echo "📋 Recent logs..."
docker-compose logs --tail=10

echo "✅ Fix applied!"
echo "🌍 Test your app at: https://pose-app.159.223.131.64.nip.io"
