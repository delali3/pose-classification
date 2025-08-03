# Pose Classification App

A real-time pose classification application using YOLO and machine learning to detect good and bad posture.

## Features

- Real-time pose detection using YOLO11
- Posture classification (Good/Bad)
- Audio alerts for bad posture
- Interactive Streamlit dashboard
- Analytics and logging
- Export functionality

## Local Development

### Prerequisites

- Python 3.8+
- Webcam access

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd pose-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run visualize.py
```

## Digital Ocean Deployment

### Prerequisites

- Digital Ocean droplet (Ubuntu 22.04 LTS, minimum 2GB RAM)
- Domain name (optional but recommended)
- SSH access to your droplet

### Step 1: Initial Server Setup

1. Connect to your droplet:
```bash
ssh root@your-droplet-ip
```

2. Run the server setup script:
```bash
curl -O https://raw.githubusercontent.com/yourusername/pose-classification/main/setup-server.sh
chmod +x setup-server.sh
./setup-server.sh
```

3. Log out and log back in for Docker permissions to take effect.

### Step 2: Deploy the Application

1. Upload your application files to the server:
```bash
# Option 1: Using git (recommended)
git clone https://github.com/yourusername/pose-classification.git
cd pose-classification

# Option 2: Using SCP
scp -r /path/to/your/pose-classification root@your-droplet-ip:/home/root/
```

2. Update the domain in nginx.conf:
```bash
nano nginx.conf
# Replace 'your-domain.com' with your actual domain
```

3. Run the deployment script:
```bash
chmod +x deploy.sh
./deploy.sh
```

### Step 3: Access Your Application

- Without domain: `http://your-droplet-ip:8501`
- With domain: `http://your-domain.com`
- With SSL: `https://your-domain.com`

## Management Commands

### Check application status:
```bash
docker-compose ps
```

### View logs:
```bash
docker-compose logs -f
```

### Restart application:
```bash
docker-compose restart
```

### Update application:
```bash
git pull origin main
docker-compose build --no-cache
docker-compose up -d
```

### Stop application:
```bash
docker-compose down
```

## File Structure

```
pose-classification/
├── app.py                 # Training script
├── visualize.py          # Main Streamlit application
├── mc.py                 # Additional utilities
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
├── nginx.conf           # Nginx reverse proxy configuration
├── setup-server.sh      # Server setup script
├── deploy.sh            # Deployment script
├── yolo11n-pose.pt      # YOLO model weights
├── pose_classifier.pkl  # Trained classifier
├── posture_log.csv      # Logging data
├── images/              # Training images
│   ├── good/           # Good posture images
│   └── bad/            # Bad posture images
└── sound/              # Audio files
    └── sound.wav       # Alert sound
```

## Troubleshooting

### Common Issues

1. **Port already in use**:
```bash
sudo lsof -i :8501
sudo kill -9 <PID>
```

2. **Docker permission denied**:
```bash
sudo usermod -aG docker $USER
# Log out and log back in
```

3. **Nginx configuration errors**:
```bash
sudo nginx -t
sudo systemctl reload nginx
```

4. **SSL certificate issues**:
```bash
sudo certbot --nginx -d your-domain.com
```

### Monitoring

- **System resources**: `htop`
- **Docker stats**: `docker stats`
- **Nginx logs**: `sudo tail -f /var/log/nginx/error.log`
- **Application logs**: `docker-compose logs -f`

## Security Considerations

- Keep your system updated: `sudo apt update && sudo apt upgrade`
- Use strong SSH keys and disable password authentication
- Configure fail2ban for SSH protection
- Regularly backup your data and models
- Monitor application logs for suspicious activity

## Performance Optimization

- Use a droplet with at least 4GB RAM for better ML model performance
- Consider using GPU-enabled droplets for faster inference
- Implement caching for frequently accessed data
- Monitor CPU and memory usage regularly

## Support

For issues and questions, please check the logs first:
```bash
docker-compose logs -f
```

If you need help, please create an issue in the repository with:
- Error messages
- System information
- Steps to reproduce the problem
