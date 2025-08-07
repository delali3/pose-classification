# Enhanced Streamlit Realtime ProActive Watch Dashboard

import streamlit as st
st.set_page_config(
    page_title="ProActive Watch Dashboard", 
    layout="wide",
    page_icon="🧍‍♂️",
    initial_sidebar_state="expanded"
)

import cv2
import numpy as np
import joblib
import tempfile
import time
import os
from datetime import datetime, timedelta
from ultralytics import YOLO
import threading
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Configuration & Constants
# ------------------------------
class Config:
    NOTIFICATION_THRESHOLD = 5  # seconds
    ALARM_THRESHOLD = 10  # seconds
    LOG_FILE = "posture_log.csv"
    MODEL_FILE = "yolo11n-pose.pt"
    CLASSIFIER_FILE = "pose_classifier.pkl"
    SOUND_FILE = "sound/sound.wav"  # Changed to WAV for better compatibility
    LABELS = ["good", "bad"]
    
    # Pose skeleton connections
    SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

# ------------------------------
# Session State Management
# ------------------------------
def initialize_session_state():
    """Initialize all session state variables"""
    if 'posture_timer' not in st.session_state:
        st.session_state.posture_timer = {"start": None}
    if 'notification_shown' not in st.session_state:
        st.session_state.notification_shown = False
    if 'alarm_shown' not in st.session_state:
        st.session_state.alarm_shown = False
    if 'success_shown' not in st.session_state:
        st.session_state.success_shown = False
    if 'beep_active' not in st.session_state:
        st.session_state.beep_active = False
    if 'alarm_thread' not in st.session_state:
        st.session_state.alarm_thread = None
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    if 'session_stats' not in st.session_state:
        st.session_state.session_stats = {
            'good_count': 0,
            'bad_count': 0,
            'session_start': datetime.now(),
            'total_alerts': 0
        }

# ------------------------------
# Model Loading with Error Handling
# ------------------------------
@st.cache_resource
def load_models():
    """Load YOLO model and posture classifier with error handling"""
    try:
        # Add safe globals for YOLO model loading to handle PyTorch 2.6+ security restrictions
        import torch
        import ultralytics.nn.tasks
        
        # Allow ultralytics classes to be loaded safely
        torch.serialization.add_safe_globals([
            ultralytics.nn.tasks.PoseModel,
            ultralytics.nn.tasks.DetectionModel,
            ultralytics.models.yolo.pose.PosePredictor,
            ultralytics.models.yolo.detect.DetectionPredictor
        ])
        
        model = YOLO(Config.MODEL_FILE)
        if os.path.exists(Config.CLASSIFIER_FILE):
            clf = joblib.load(Config.CLASSIFIER_FILE)
            return model, clf, None
        else:
            return model, None, "Classifier file not found. Please ensure pose_classifier.pkl exists."
    except Exception as e:
        return None, None, f"Error loading models: {str(e)}"

# ------------------------------
# ENHANCED Audio System with WAV Support
# ------------------------------
def play_wav_with_pygame(file_path, repeat=3):
    """Play WAV file with pygame - More reliable than MP3"""
    try:
        import pygame
        
        # Initialize pygame mixer with settings optimized for WAV
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        # Load WAV file (more reliable than music for WAV)
        sound = pygame.mixer.Sound(file_path)
        
        for i in range(repeat):
            print(f"🔊 Playing WAV with pygame {i+1}/{repeat}...")
            sound.play()
            
            # Wait for sound to finish
            while pygame.mixer.get_busy():
                pygame.time.wait(100)
            
            if i < repeat - 1:  # Don't wait after last play
                pygame.time.wait(500)
        
        pygame.mixer.quit()
        print("✅ WAV played successfully with pygame!")
        return True
        
    except ImportError:
        print("❌ pygame not available")
        return False
    except Exception as e:
        print(f"❌ pygame WAV error: {e}")
        return False

def play_wav_with_playsound(file_path, repeat=3):
    """Play WAV with playsound library"""
    try:
        from playsound import playsound
        
        for i in range(repeat):
            print(f"🔊 Playing WAV with playsound {i+1}/{repeat}...")
            playsound(file_path)
            if i < repeat - 1:
                time.sleep(0.5)
        
        print("✅ WAV played successfully with playsound!")
        return True
        
    except ImportError:
        print("❌ playsound not available")
        return False
    except Exception as e:
        print(f"❌ playsound WAV error: {e}")
        return False

def play_wav_with_os(file_path, repeat=3):
    """Play WAV using OS-specific commands"""
    try:
        import platform
        system = platform.system()
        
        for i in range(repeat):
            print(f"🔊 Playing WAV with OS command {i+1}/{repeat}...")
            
            if system == "Windows":
                # Windows can play WAV files directly
                import winsound
                winsound.PlaySound(file_path, winsound.SND_FILENAME)
            elif system == "Darwin":  # macOS
                os.system(f"afplay '{file_path}'")
            elif system == "Linux":
                # Try multiple Linux audio commands
                if os.system(f"aplay '{file_path}' 2>/dev/null") != 0:
                    if os.system(f"paplay '{file_path}' 2>/dev/null") != 0:
                        os.system(f"ffplay -nodisp -autoexit '{file_path}' 2>/dev/null")
            
            if i < repeat - 1:
                time.sleep(0.5)
        
        print("✅ WAV played successfully with OS commands!")
        return True
        
    except Exception as e:
        print(f"❌ OS WAV playback error: {e}")
        return False

def play_system_sound():
    """Play system fallback sound"""
    try:
        import platform
        system = platform.system()
        
        if system == "Windows":
            try:
                import winsound
                print("🔊 Playing Windows system beep...")
                for _ in range(5):
                    winsound.Beep(1000, 500)
                    time.sleep(0.2)
                print("✅ Windows beep successful!")
                return True
            except:
                print("❌ Windows beep failed")
                
        elif system == "Darwin":  # macOS
            try:
                print("🔊 Playing macOS system sound...")
                os.system("afplay /System/Library/Sounds/Sosumi.aiff")
                print("✅ macOS sound successful!")
                return True
            except:
                print("❌ macOS sound failed")
                
        elif system == "Linux":
            try:
                print("🔊 Playing Linux system sound...")
                os.system("paplay /usr/share/sounds/alsa/Front_Left.wav")
                print("✅ Linux sound successful!")
                return True
            except:
                print("❌ Linux sound failed")
        
        return False
        
    except Exception as e:
        print(f"❌ System sound error: {e}")
        return False

def play_alarm_sound():
    """Main alarm function - OPTIMIZED for WAV files"""
    print(f"🚨 ALARM TRIGGERED! Trying to play: {Config.SOUND_FILE}")
    
    # Check if custom sound file exists
    if not os.path.exists(Config.SOUND_FILE):
        print(f"❌ Custom sound not found: {Config.SOUND_FILE}")
        print("🔄 Using system fallback...")
        if not play_system_sound():
            print("🚨 ALARM - NO AUDIO AVAILABLE!")
        return
    
    print(f"✅ Custom sound file found: {Config.SOUND_FILE}")
    file_ext = os.path.splitext(Config.SOUND_FILE)[1].lower()
    print(f"📄 File type: {file_ext}")
    
    # Try different methods based on file type
    if file_ext == '.wav':
        print("🎵 Detected WAV file - using optimized WAV playback")
        
        # Method 1: pygame with Sound (best for WAV)
        if play_wav_with_pygame(Config.SOUND_FILE):
            return
        
        # Method 2: OS-specific commands (very reliable for WAV)
        if play_wav_with_os(Config.SOUND_FILE):
            return
            
        # Method 3: playsound
        if play_wav_with_playsound(Config.SOUND_FILE):
            return
            
    else:  # MP3 or other formats
        print("🎵 Non-WAV file detected - using general playback")
        
        # Method 1: playsound (better for MP3)
        if play_wav_with_playsound(Config.SOUND_FILE):
            return
            
        # Method 2: pygame
        if play_wav_with_pygame(Config.SOUND_FILE):
            return
    
    # All custom methods failed, use system sound
    print("🔄 All custom sound methods failed, using system fallback...")
    if not play_system_sound():
        print("🚨 ALARM - NO AUDIO AVAILABLE!")

def test_wav_only():
    """Test ONLY custom WAV file - optimized for WAV"""
    if not os.path.exists(Config.SOUND_FILE):
        print(f"❌ Sound file not found: {Config.SOUND_FILE}")
        return False
    
    print(f"🎵 Testing sound file: {Config.SOUND_FILE}")
    file_ext = os.path.splitext(Config.SOUND_FILE)[1].lower()
    print(f"📄 File extension: {file_ext}")
    
    if file_ext == '.wav':
        print("🎵 Testing WAV file with optimized methods...")
        
        # Test pygame WAV
        if play_wav_with_pygame(Config.SOUND_FILE, repeat=1):
            return True
        
        # Test OS WAV commands
        if play_wav_with_os(Config.SOUND_FILE, repeat=1):
            return True
            
        # Test playsound
        if play_wav_with_playsound(Config.SOUND_FILE, repeat=1):
            return True
    else:
        print("🎵 Testing non-WAV file...")
        
        # Test playsound first for MP3
        if play_wav_with_playsound(Config.SOUND_FILE, repeat=1):
            return True
            
        # Test pygame
        if play_wav_with_pygame(Config.SOUND_FILE, repeat=1):
            return True
    
    print("❌ All test methods failed!")
    return False

# ------------------------------
# Simple Notification System
# ------------------------------
def show_notification(message, alert_type="info"):
    """Simple notification using Streamlit's built-in methods"""
    if alert_type == "notification":
        st.warning(f"📢 POSTURE ALERT: {message}")
    elif alert_type == "alarm":
        st.error(f"🚨 POSTURE ALARM: {message}")
    elif alert_type == "success":
        st.success(f"✅ GREAT POSTURE: {message}")
    else:
        st.info(f"ℹ️ {message}")

# ------------------------------
# FIXED Posture Tracking
# ------------------------------
def track_poor_posture(is_poor):
    """Enhanced posture tracking - FIXED to always return float"""
    current_time = time.time()
    
    if is_poor:
        # Update session stats
        st.session_state.session_stats['bad_count'] += 1
        
        if st.session_state.posture_timer['start'] is None:
            st.session_state.posture_timer['start'] = current_time
            st.session_state.notification_shown = False
            st.session_state.alarm_shown = False
            st.session_state.success_shown = False
            return 0.0  # Just started
        else:
            elapsed_time = current_time - st.session_state.posture_timer['start']
            
            # Stage 1: Notification
            if (elapsed_time >= Config.NOTIFICATION_THRESHOLD and 
                elapsed_time < Config.ALARM_THRESHOLD and 
                not st.session_state.notification_shown):
                
                st.session_state.notification_shown = True
                st.session_state.session_stats['total_alerts'] += 1
                remaining_time = Config.ALARM_THRESHOLD - elapsed_time
                show_notification(f"Alarm in {remaining_time:.1f} seconds", "notification")
                
            # Stage 2: Alarm
            elif elapsed_time >= Config.ALARM_THRESHOLD and not st.session_state.alarm_shown:
                st.session_state.alarm_shown = True
                st.session_state.beep_active = True
                show_notification(f"Poor posture for {elapsed_time:.1f} seconds", "alarm")
                
                # Start alarm in background thread
                alarm_thread = threading.Thread(target=play_alarm_sound, daemon=True)
                alarm_thread.start()
                
            return elapsed_time
    else:
        # Good posture detected
        st.session_state.session_stats['good_count'] += 1
        
        if (st.session_state.posture_timer['start'] is not None and 
            not st.session_state.success_shown):
            st.session_state.success_shown = True
            show_notification("Excellent posture!", "success")
        
        # Reset timer and flags
        st.session_state.posture_timer['start'] = None
        st.session_state.notification_shown = False
        st.session_state.alarm_shown = False
        st.session_state.beep_active = False
        
        return 0.0  # Good posture

# ------------------------------
# Logging System
# ------------------------------
def log_event(label, confidence=None):
    """Enhanced logging with confidence scores"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create log entry
    log_entry = f"{timestamp},{label}"
    if confidence is not None:
        log_entry += f",{confidence:.3f}"
    log_entry += "\n"
    
    # Append to log file
    with open(Config.LOG_FILE, "a") as f:
        f.write(log_entry)

# ------------------------------
# Pose Drawing
# ------------------------------
def draw_enhanced_pose(img, kps, label, confidence=None):
    """Enhanced pose drawing with confidence and better visualization"""
    # Draw keypoints
    for i, (x, y) in enumerate(kps):
        color = (0, 255, 0) if label == "good" else (0, 0, 255)
        cv2.circle(img, (int(x), int(y)), 6, color, -1)
        cv2.circle(img, (int(x), int(y)), 8, (255, 255, 255), 2)
    
    # Draw skeleton
    for i, j in Config.SKELETON:
        if i < len(kps) and j < len(kps):
            x1, y1 = kps[i]
            x2, y2 = kps[j]
            color = (0, 255, 0) if label == "good" else (0, 0, 255)
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    
    # Add status overlay
    h, w = img.shape[:2]
    overlay = img.copy()
    
    # Status box
    status_color = (0, 255, 0) if label == "good" else (0, 0, 255)
    cv2.rectangle(overlay, (10, 10), (w-10, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Status text
    status_text = f"Posture: {label.upper()}"
    cv2.putText(img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2)
    
    if confidence is not None:
        conf_text = f"Confidence: {confidence:.1%}"
        cv2.putText(img, conf_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img

# ------------------------------
# Data Analysis Functions
# ------------------------------
def load_log_data():
    """Load and process log data with error handling"""
    if not os.path.exists(Config.LOG_FILE):
        return None
    
    try:
        # Try to read with confidence column
        df = pd.read_csv(Config.LOG_FILE, names=["Timestamp", "Posture", "Confidence"])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except:
        # Fallback to basic format
        try:
            df = pd.read_csv(Config.LOG_FILE, names=["Timestamp", "Posture"])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Confidence'] = None
            return df
        except Exception as e:
            st.error(f"Error loading log data: {e}")
            return None

def create_analytics_dashboard(df):
    """Create comprehensive analytics dashboard"""
    if df is None or df.empty:
        st.info("No data available for analysis")
        return 0, 0, 0, 0.0  # Return default values instead of None
    
    # Summary statistics
    total_records = len(df)
    good_count = len(df[df['Posture'] == 'good'])
    bad_count = len(df[df['Posture'] == 'bad'])
    good_percentage = (good_count / total_records) * 100 if total_records > 0 else 0
    
    # Time-based analysis
    df['Hour'] = df['Timestamp'].dt.hour
    df['Date'] = df['Timestamp'].dt.date
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Posture distribution pie chart
        fig_pie = px.pie(
            values=[good_count, bad_count], 
            names=['Good Posture', 'Bad Posture'],
            title="Posture Distribution",
            color_discrete_map={'Good Posture': '#28a745', 'Bad Posture': '#dc3545'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Hourly posture trends
        hourly_stats = df.groupby(['Hour', 'Posture']).size().unstack(fill_value=0)
        
        # Ensure both 'good' and 'bad' columns exist
        if 'good' not in hourly_stats.columns:
            hourly_stats['good'] = 0
        if 'bad' not in hourly_stats.columns:
            hourly_stats['bad'] = 0
        
        # Reorder columns to ensure consistent ordering
        hourly_stats = hourly_stats[['good', 'bad']]
        
        fig_bar = px.bar(
            hourly_stats.reset_index(), 
            x='Hour', 
            y=['good', 'bad'],
            title="Posture Trends by Hour",
            labels={'value': 'Count', 'variable': 'Posture'},
            color_discrete_map={'good': '#28a745', 'bad': '#dc3545'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Daily trends
    daily_stats = df.groupby(['Date', 'Posture']).size().unstack(fill_value=0)
    if not daily_stats.empty:
        # Ensure both 'good' and 'bad' columns exist
        if 'good' not in daily_stats.columns:
            daily_stats['good'] = 0
        if 'bad' not in daily_stats.columns:
            daily_stats['bad'] = 0
        
        # Reorder columns to ensure consistent ordering
        daily_stats = daily_stats[['good', 'bad']]
        
        fig_daily = px.line(
            daily_stats.reset_index(), 
            x='Date', 
            y=['good', 'bad'],
            title="Daily Posture Trends",
            labels={'value': 'Count', 'variable': 'Posture'},
            color_discrete_map={'good': '#28a745', 'bad': '#dc3545'}
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    
    return total_records, good_count, bad_count, good_percentage

# ------------------------------
# Main Application
# ------------------------------
def main():
    initialize_session_state()
    
    # Load models
    model, clf, error = load_models()
    if error:
        st.error(error)
        st.stop()
    
    # Check for sound file
    if not os.path.exists(Config.SOUND_FILE):
        st.warning(f"⚠️ Custom sound file not found at: {Config.SOUND_FILE}")
        st.info("The app will use system default sounds as fallback")
    else:
        st.success(f"✅ Custom sound file loaded: {Config.SOUND_FILE}")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown(
        '<div class="main-header">🧍‍♂️ ProActive Watch Dashboard</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/standing-man.png", width=80)
        st.markdown("## 🧍‍♂️ ProActive Watch")
        st.markdown("---")
        
        # Session statistics
        st.markdown("### Session Stats")
        session_duration = datetime.now() - st.session_state.session_stats['session_start']
        st.metric("Session Duration", f"{session_duration.seconds // 60}m {session_duration.seconds % 60}s")
        st.metric("Good Posture Count", st.session_state.session_stats['good_count'])
        st.metric("Bad Posture Count", st.session_state.session_stats['bad_count'])
        st.metric("Total Alerts", st.session_state.session_stats['total_alerts'])
        
        st.markdown("---")
        
        # Settings
        st.markdown("### Settings")
        notification_threshold = st.slider("Notification Threshold (s)", 3, 10, Config.NOTIFICATION_THRESHOLD)
        alarm_threshold = st.slider("Alarm Threshold (s)", 5, 20, Config.ALARM_THRESHOLD)
        
        # Update config
        Config.NOTIFICATION_THRESHOLD = notification_threshold
        Config.ALARM_THRESHOLD = alarm_threshold
        
        if st.button("Reset Session Stats"):
            st.session_state.session_stats = {
                'good_count': 0,
                'bad_count': 0,
                'session_start': datetime.now(),
                'total_alerts': 0
            }
            st.success("Session stats reset!")
    
    # Main tabs
    tabs = st.tabs(["📊 Live Monitor", "📈 Analytics", "📋 Logs", "⚙️ Settings"])
    
    with tabs[0]:
        st.markdown("### Real-time Posture Monitoring")
        
        # Server deployment notice
        st.info("""
        🌐 **Server Deployment Notice**: 
        - **Webcam**: May not work on server deployments (expected behavior)
        - **Image Upload**: ✅ Works perfectly! Recommended for server use
        - **Best Practice**: Take photos with your phone/camera and upload them
        """)
        
        # Alert system explanation
        with st.expander("🔔 Alert System Guide", expanded=False):
            st.markdown(f"""
            **Smart Alert System:**
            - 🟡 **0-{Config.NOTIFICATION_THRESHOLD}s**: Monitoring (status updates only)
            - 🟠 **{Config.NOTIFICATION_THRESHOLD}s**: One-time notification popup
            - 🔴 **{Config.ALARM_THRESHOLD}s**: Emergency alarm popup + audio
            - ✅ **Good posture**: Success message when improved
            - 🔄 **Smart logic**: No spam, shows only on state changes
            - 🔊 **Custom sound**: Uses {Config.SOUND_FILE} for alarms
            """)
        
        # Controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            use_webcam = st.checkbox("🎥 Use Webcam (Live Detection)")
            # Add server deployment notice
            if use_webcam:
                st.info("💡 **Note**: Webcam access may not work on server deployments. Use image upload for best results!")
        with col2:
            if st.button("🔇 Stop Alarm"):
                st.session_state.beep_active = False
                st.success("Alarm stopped!")
        with col3:
            if st.button("📊 Reset Timer"):
                st.session_state.posture_timer['start'] = None
                st.info("Timer reset!")
        
        # File upload
        if not use_webcam:
            uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
            st.info("📱 **Tip**: Take a photo with your phone and upload it here for posture analysis!")
        
        # Status display
        status_placeholder = st.empty()
        frame_placeholder = st.empty()
        
        # Main monitoring logic
        if use_webcam:
            if st.button("🔴 Start Monitoring") or st.session_state.monitoring_active:
                st.session_state.monitoring_active = True
                
                if st.button("⏹️ Stop Monitoring"):
                    st.session_state.monitoring_active = False
                    st.rerun()
                
                # Webcam processing
                try:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("❌ **Webcam Access Failed**")
                        st.error("This is expected on server deployments (like DigitalOcean).")
                        st.info("🔄 **Solution**: Use the Image Upload feature below instead!")
                        st.session_state.monitoring_active = False
                        st.rerun()
                        return
                    
                    frame_count = 0
                    
                    while st.session_state.monitoring_active and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.error("❌ **Failed to access webcam**")
                            st.info("💡 **This is normal on server deployments**")
                            st.info("📱 **Please use Image Upload instead** - it works perfectly!")
                            break
                        
                        # Process every 3rd frame for performance
                        if frame_count % 3 == 0:
                            results = model.predict(source=frame, save=False, verbose=False)
                            
                            if len(results[0].keypoints.data) > 0:
                                kps = results[0].keypoints.data[0].cpu().numpy()[:, :2]
                                
                                # Calculate distances for classification
                                distances = []
                                for i in range(len(kps)):
                                    for j in range(i + 1, len(kps)):
                                        dist = np.linalg.norm(kps[i] - kps[j])
                                        distances.append(dist)
                                
                                if len(distances) > 0 and clf is not None:
                                    prediction = clf.predict([distances])[0]
                                    label = Config.LABELS[prediction]
                                    confidence = max(clf.predict_proba([distances])[0])
                                    
                                    # Draw enhanced pose
                                    frame = draw_enhanced_pose(frame, kps, label, confidence)
                                    
                                    # Log event
                                    log_event(label, confidence)
                                    
                                    # Track posture and get elapsed time (FIXED - always returns float)
                                    elapsed_time = track_poor_posture(label == "bad")
                                    
                                    # Update status
                                    if label == "bad" and elapsed_time > 0:
                                        if elapsed_time < Config.NOTIFICATION_THRESHOLD:
                                            remaining_time = Config.NOTIFICATION_THRESHOLD - elapsed_time
                                            status_placeholder.info(f"⚠️ Poor posture detected for {elapsed_time:.1f}s. Notification in {remaining_time:.1f}s")
                                        elif elapsed_time < Config.ALARM_THRESHOLD:
                                            remaining_time = Config.ALARM_THRESHOLD - elapsed_time
                                            status_placeholder.warning(f"📢 Notification shown. Alarm in {remaining_time:.1f}s")
                                        else:
                                            status_placeholder.error(f"🚨 Alarm triggered! Duration: {elapsed_time:.1f}s")
                                    else:
                                        status_placeholder.success("✅ Good posture maintained")
                                else:
                                    status_placeholder.warning("⚠️ Unable to classify posture")
                            else:
                                status_placeholder.warning("⚠️ No person detected")
                        
                        # Display frame
                        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        frame_count += 1
                        
                        # Small delay to prevent overwhelming
                        time.sleep(0.1)
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    
                except Exception as e:
                    st.error("❌ **Webcam Error**")
                    st.error(f"Error details: {str(e)}")
                    st.info("💡 **This is expected on server deployments**")
                    st.info("📱 **Please use Image Upload feature instead** - it works great!")
                    st.session_state.monitoring_active = False
        
        elif uploaded_file:
            # Process uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.read())
                image_path = tmp_file.name
            
            image = cv2.imread(image_path)
            results = model.predict(source=image, save=False, verbose=False)
            
            if len(results[0].keypoints.data) > 0:
                kps = results[0].keypoints.data[0].cpu().numpy()[:, :2]
                
                # Calculate distances
                distances = []
                for i in range(len(kps)):
                    for j in range(i + 1, len(kps)):
                        dist = np.linalg.norm(kps[i] - kps[j])
                        distances.append(dist)
                
                if len(distances) > 0 and clf is not None:
                    prediction = clf.predict([distances])[0]
                    label = Config.LABELS[prediction]
                    confidence = max(clf.predict_proba([distances])[0])
                    
                    # Draw enhanced pose
                    image = draw_enhanced_pose(image, kps, label, confidence)
                    
                    # Display result
                    frame_placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
                    # Show status
                    if label == "good":
                        status_placeholder.success(f"✅ Good posture detected! (Confidence: {confidence:.1%})")
                    else:
                        status_placeholder.error(f"❌ Poor posture detected! (Confidence: {confidence:.1%})")
                    
                    # Log event
                    log_event(label, confidence)
                else:
                    status_placeholder.error("❌ Unable to extract features for classification")
            else:
                status_placeholder.error("❌ No person detected in the image")
            
            # Clean up
            os.unlink(image_path)
    
    with tabs[1]:
        st.markdown("### Analytics Dashboard")
        
        # Load and analyze data
        df = load_log_data()
        if df is not None and not df.empty:
            total_records, good_count, bad_count, good_percentage = create_analytics_dashboard(df)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 2rem; color: #28a745;">✅</div>
                    <div class="metric-value" style="color: #28a745;">{good_count}</div>
                    <div>Good Posture</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 2rem; color: #dc3545;">❌</div>
                    <div class="metric-value" style="color: #dc3545;">{bad_count}</div>
                    <div>Bad Posture</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 2rem; color: #007bff;">📊</div>
                    <div class="metric-value" style="color: #007bff;">{total_records}</div>
                    <div>Total Records</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 2rem; color: #ffc107;">📈</div>
                    <div class="metric-value" style="color: #ffc107;">{good_percentage:.1f}%</div>
                    <div>Good Posture Rate</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No analytics data available yet. Start monitoring to generate insights!")
    
    with tabs[2]:
        st.markdown("### Detailed Logs")
        
        df = load_log_data()
        if df is not None:
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                posture_filter = st.selectbox("Filter by Posture", ["All", "Good", "Bad"])
            with col2:
                date_filter = st.date_input("Filter by Date", value=None)
            with col3:
                records_limit = st.selectbox("Show Records", [50, 100, 200, 500, "All"])
            
            # Apply filters
            filtered_df = df.copy()
            
            if posture_filter != "All":
                filtered_df = filtered_df[filtered_df['Posture'].str.title() == posture_filter]
            
            if date_filter:
                filtered_df = filtered_df[filtered_df['Timestamp'].dt.date == date_filter]
            
            if records_limit != "All":
                filtered_df = filtered_df.tail(records_limit)
            
            # Display filtered data
            if not filtered_df.empty:
                # Format the dataframe for better display
                display_df = filtered_df.copy()
                display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                display_df['Posture'] = display_df['Posture'].str.title()
                
                if 'Confidence' in display_df.columns:
                    display_df['Confidence'] = display_df['Confidence'].apply(
                        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                    )
                
                # Color coding for posture - FIXED deprecated applymap
                def highlight_posture(val):
                    if val == 'Good':
                        return 'background-color: #d4edda; color: #155724'
                    elif val == 'Bad':
                        return 'background-color: #f8d7da; color: #721c24'
                    return ''
                
                # Use map instead of deprecated applymap
                styled_df = display_df.style.map(highlight_posture, subset=['Posture'])
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Export functionality
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Export to CSV"):
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"posture_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("🗑️ Clear All Logs"):
                        if st.checkbox("I understand this will delete all log data"):
                            if os.path.exists(Config.LOG_FILE):
                                os.remove(Config.LOG_FILE)
                                st.success("All logs cleared!")
                                st.rerun()
            else:
                st.info("No records match the selected filters.")
        else:
            st.info("No log data available yet. Start monitoring to generate logs!")
    
    with tabs[3]:
        st.markdown("### Settings & Configuration")
        
        # Sound settings - COMPLETELY REWRITTEN
        st.markdown("#### 🔊 Audio Configuration")
        
        # Sound file path setting
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Sound File Settings**")
            current_sound = st.text_input(
                "Custom Sound File Path", 
                value=Config.SOUND_FILE,
                help="Path to your custom MP3/WAV sound file"
            )
            
            # Update config when path changes
            Config.SOUND_FILE = current_sound
            
            # Sound file status with detailed info
            if os.path.exists(current_sound):
                file_size = os.path.getsize(current_sound) / 1024
                st.success(f"✅ Sound file found")
                st.info(f"📁 Size: {file_size:.1f} KB")
                st.info(f"📂 Path: {current_sound}")
            else:
                st.error(f"❌ Sound file not found")
                st.warning("Make sure the file path is correct and the file exists")
                st.info(f"Expected: {current_sound}")
        
        with col2:
            st.markdown("**Audio Test Controls**")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🎵 Test WAV Only"):
                    st.info("🎵 Testing ONLY your WAV file...")
                    with st.spinner("Playing sound..."):
                        test_thread = threading.Thread(target=test_wav_only, daemon=True)
                        test_thread.start()
                        test_thread.join(timeout=10)  # Wait max 10 seconds
                    st.success("Test completed! Check console for details.")
            
            with col_b:
                if st.button("🚨 Test Full Alarm"):
                    st.info("🚨 Testing full alarm sequence...")
                    with st.spinner("Playing alarm..."):
                        test_thread = threading.Thread(target=play_alarm_sound, daemon=True)
                        test_thread.start()
                        test_thread.join(timeout=15)  # Wait max 15 seconds
                    st.success("Alarm test completed!")
            
            # Audio debug information
            if st.button("🔧 Audio Debug Info"):
                st.markdown("**Debug Information:**")
                
                # Check libraries
                try:
                    import pygame
                    st.success("✅ pygame available")
                except ImportError:
                    st.error("❌ pygame not available")
                    st.code("pip install pygame")
                
                try:
                    from playsound import playsound
                    st.success("✅ playsound available")
                except ImportError:
                    st.error("❌ playsound not available")
                    st.code("pip install playsound")
                
                # Check system
                import platform
                st.info(f"🖥️ System: {platform.system()}")
                
                # Check file details
                if os.path.exists(Config.SOUND_FILE):
                    import mimetypes
                    file_type = mimetypes.guess_type(Config.SOUND_FILE)[0]
                    st.info(f"📄 File type: {file_type}")
                    
                    # Try to get file info
                    try:
                        with open(Config.SOUND_FILE, 'rb') as f:
                            first_bytes = f.read(10)
                        st.info(f"📊 First bytes: {first_bytes.hex()}")
                    except Exception as e:
                        st.error(f"❌ Cannot read file: {e}")
                else:
                    st.error("❌ Sound file not found for debugging")
        
        st.markdown("---")
        
        # Installation instructions
        st.markdown("#### 📦 Audio Library Installation")
        st.info("""
        **For best WAV support, install pygame:**
        ```bash
        pip install pygame
        ```
        
        **Optional (for MP3 support):**
        ```bash
        pip install playsound
        ```
        
        **Supported formats:** WAV (recommended), MP3, OGG
        **Best compatibility:** 22.05kHz, 16-bit, stereo WAV files
        """)
        
        st.markdown("---")
        
        # File format recommendation
        st.markdown("#### 🎵 Audio File Recommendations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **✅ RECOMMENDED: WAV Format**
            - Better compatibility
            - No codec issues
            - Works with pygame Sound()
            - Faster loading
            """)
        
        with col2:
            st.warning("""
            **⚠️ MP3 Format**
            - May require additional codecs
            - Slower loading
            - Compression artifacts
            - Platform-dependent
            """)
        
        # Quick file converter tip
        if Config.SOUND_FILE.endswith('.mp3'):
            st.info("""
            💡 **Quick Tip**: Convert your MP3 to WAV for better compatibility:
            - Use online converters or
            - `ffmpeg -i sound.mp3 sound.wav`
            """)
            
            # Option to switch to WAV
            if st.button("🔄 Switch to WAV file"):
                wav_path = Config.SOUND_FILE.replace('.mp3', '.wav')
                if os.path.exists(wav_path):
                    Config.SOUND_FILE = wav_path
                    st.success(f"Switched to: {wav_path}")
                    st.rerun()
                else:
                    st.error(f"WAV file not found: {wav_path}")
                    st.info("Please convert your MP3 to WAV first")
        
        st.markdown("---")
        
        # Alert settings
        st.markdown("#### ⏰ Alert Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            new_notification_threshold = st.slider(
                "Notification Threshold (seconds)", 
                min_value=3, 
                max_value=15, 
                value=Config.NOTIFICATION_THRESHOLD,
                help="Time before showing notification popup"
            )
            
            new_alarm_threshold = st.slider(
                "Alarm Threshold (seconds)", 
                min_value=5, 
                max_value=30, 
                value=Config.ALARM_THRESHOLD,
                help="Time before triggering audio alarm"
            )
        
        with col2:
            st.markdown("##### Current Settings")
            st.info(f"📢 Notification: {Config.NOTIFICATION_THRESHOLD}s")
            st.info(f"🚨 Alarm: {Config.ALARM_THRESHOLD}s")
            st.info(f"🔊 Sound: {os.path.basename(Config.SOUND_FILE)}")
            
            if st.button("💾 Save Settings"):
                Config.NOTIFICATION_THRESHOLD = new_notification_threshold
                Config.ALARM_THRESHOLD = new_alarm_threshold
                Config.SOUND_FILE = current_sound
                st.success("Settings saved!")
        
        st.markdown("---")
        
        # Model information
        st.markdown("#### 🤖 Model Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**YOLO Model:**")
            if model:
                st.success(f"✅ Loaded: {Config.MODEL_FILE}")
            else:
                st.error(f"❌ Failed to load: {Config.MODEL_FILE}")
        
        with col2:
            st.markdown("**Posture Classifier:**")
            if clf:
                st.success(f"✅ Loaded: {Config.CLASSIFIER_FILE}")
            else:
                st.error(f"❌ Failed to load: {Config.CLASSIFIER_FILE}")
        
        st.markdown("---")
        
        # System information
        st.markdown("#### 💻 System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Performance:**")
            st.info(f"Session started: {st.session_state.session_stats['session_start'].strftime('%H:%M:%S')}")
            st.info(f"Monitoring: {'Active' if st.session_state.monitoring_active else 'Inactive'}")
        
        with col2:
            st.markdown("**Data:**")
            log_size = os.path.getsize(Config.LOG_FILE) if os.path.exists(Config.LOG_FILE) else 0
            st.info(f"Log file size: {log_size / 1024:.1f} KB")
            st.info(f"Total records: {len(load_log_data()) if load_log_data() is not None else 0}")
        
        st.markdown("---")
        
        # Reset options
        st.markdown("#### 🔄 Reset Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reset Session"):
                st.session_state.session_stats = {
                    'good_count': 0,
                    'bad_count': 0,
                    'session_start': datetime.now(),
                    'total_alerts': 0
                }
                st.session_state.posture_timer = {"start": None}
                st.session_state.notification_shown = False
                st.session_state.alarm_shown = False
                st.session_state.success_shown = False
                st.success("Session reset successfully!")
        
        with col2:
            if st.button("🔧 Reset Settings"):
                Config.NOTIFICATION_THRESHOLD = 5
                Config.ALARM_THRESHOLD = 10
                Config.SOUND_FILE = "sound/sound.wav"  # Default to WAV
                st.success("Settings reset to defaults!")
        
        with col3:
            if st.button("🗑️ Clear All Data"):
                if st.checkbox("Confirm data deletion"):
                    if os.path.exists(Config.LOG_FILE):
                        os.remove(Config.LOG_FILE)
                    # Reset session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.success("All data cleared!")
                    st.rerun()

# ------------------------------
# Footer
# ------------------------------
def show_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>🧍‍♂️ ProActive Watch Dashboard | Built with Streamlit & YOLO</p>
            <p>Monitor your posture • Stay healthy • Be productive</p>
            <p><strong>🔊 Audio Requirements:</strong> pip install pygame</p>
            <p><strong>📁 Sound File:</strong> Place your WAV file at sound/sound.wav</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------
# Run Application
# ------------------------------
if __name__ == "__main__":
    main()
    show_footer()