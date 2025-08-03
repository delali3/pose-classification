# Streamlit Realtime ProActive Watch

import streamlit as st
import cv2
import numpy as np
import joblib
import tempfile
import time
from datetime import datetime
from ultralytics import YOLO
import winsound  # For Windows audio beep
import threading

# ------------------------------
# Initialization
# ------------------------------
model = YOLO("yolo11n-pose.pt")
clf = joblib.load("pose_classifier.pkl")
labels = ["good", "bad"]
NOTIFICATION_THRESHOLD = 5  # seconds - show screen notification
ALARM_THRESHOLD = 10  # seconds - play audio alarm (5s notification + 5s warning)
beep_active = False
notification_shown = False
alarm_shown = False
success_shown = False
posture_timer = {"start": None}
alarm_thread = None

# ------------------------------
# Helper Functions
# ------------------------------
def show_popup_notification(message, alert_type="warning"):
    """Show a modal-style pop-up notification that auto-hides after 2 seconds"""
    popup_id = f"popup_{alert_type}_{int(time.time() * 1000)}"  # Unique ID
    
    if alert_type == "notification":
        # Yellow warning pop-up with auto-hide
        st.markdown("""
        <div id="{}" style="position: fixed; top: 20%; left: 50%; transform: translateX(-50%); z-index: 9999; background-color: #ffc107; padding: 25px; border-radius: 15px; border: 4px solid #e0a800; text-align: center; color: black; font-size: 20px; font-weight: bold; box-shadow: 0 10px 20px rgba(0,0,0,0.5); min-width: 400px; animation: popup-show 0.3s ease-out;">
            📢 POSTURE NOTIFICATION<br><br>
            {}<br><br>
            <small style="font-size: 14px;">Please adjust your posture</small>
        </div>
        <style>
        @keyframes popup-show {{
            0% {{ opacity: 0; transform: translateX(-50%) translateY(-20px) scale(0.9); }}
            100% {{ opacity: 1; transform: translateX(-50%) translateY(0) scale(1); }}
        }}
        @keyframes popup-hide {{
            0% {{ opacity: 1; transform: translateX(-50%) scale(1); }}
            100% {{ opacity: 0; transform: translateX(-50%) scale(0.9); }}
        }}
        </style>
        <script>
        setTimeout(function() {{
            var popup = document.getElementById('{}');
            if (popup) {{
                popup.style.animation = 'popup-hide 0.3s ease-in';
                setTimeout(function() {{
                    popup.style.display = 'none';
                }}, 300);
            }}
        }}, 2000);
        </script>
        """.format(popup_id, message, popup_id), unsafe_allow_html=True)
    
    elif alert_type == "alarm":
        # Red alarm pop-up with blinking and auto-hide
        st.markdown("""
        <div id="{}" style="position: fixed; top: 15%; left: 50%; transform: translateX(-50%); z-index: 9999; background-color: #dc143c; padding: 30px; border-radius: 20px; border: 6px solid #8b0000; text-align: center; color: white; font-size: 24px; font-weight: bold; box-shadow: 0 15px 30px rgba(0,0,0,0.7); min-width: 500px; animation: emergency-blink 0.5s linear infinite;">
            🚨 EMERGENCY ALARM 🚨<br><br>
            {}<br><br>
            <small style="font-size: 16px;">ADJUST YOUR POSITION NOW!</small>
        </div>
        <style>
        @keyframes emergency-blink {{
            0% {{ background-color: #dc143c; transform: translateX(-50%) scale(1); }}
            50% {{ background-color: #ff0000; transform: translateX(-50%) scale(1.02); }}
            100% {{ background-color: #dc143c; transform: translateX(-50%) scale(1); }}
        }}
        @keyframes popup-hide {{
            0% {{ opacity: 1; transform: translateX(-50%) scale(1); }}
            100% {{ opacity: 0; transform: translateX(-50%) scale(0.9); }}
        }}
        </style>
        <script>
        setTimeout(function() {{
            var popup = document.getElementById('{}');
            if (popup) {{
                popup.style.animation = 'popup-hide 0.3s ease-in';
                setTimeout(function() {{
                    popup.style.display = 'none';
                }}, 300);
            }}
        }}, 2000);
        </script>
        """.format(popup_id, message, popup_id), unsafe_allow_html=True)
    
    elif alert_type == "success":
        # Green success pop-up with auto-hide
        st.markdown("""
        <div id="{}" style="position: fixed; top: 25%; left: 50%; transform: translateX(-50%); z-index: 9999; background-color: #28a745; padding: 20px; border-radius: 12px; border: 3px solid #1e7e34; text-align: center; color: white; font-size: 18px; font-weight: bold; box-shadow: 0 8px 16px rgba(0,0,0,0.4); min-width: 350px; animation: popup-show 0.3s ease-out;">
            ✅ POSTURE RESTORED<br><br>
            {}<br><br>
            <small style="font-size: 14px;">Keep up the good work!</small>
        </div>
        <style>
        @keyframes popup-show {{
            0% {{ opacity: 0; transform: translateX(-50%) translateY(-20px) scale(0.9); }}
            100% {{ opacity: 1; transform: translateX(-50%) translateY(0) scale(1); }}
        }}
        @keyframes popup-hide {{
            0% {{ opacity: 1; transform: translateX(-50%) scale(1); }}
            100% {{ opacity: 0; transform: translateX(-50%) scale(0.9); }}
        }}
        </style>
        <script>
        setTimeout(function() {{
            var popup = document.getElementById('{}');
            if (popup) {{
                popup.style.animation = 'popup-hide 0.3s ease-in';
                setTimeout(function() {{
                    popup.style.display = 'none';
                }}, 300);
            }}
        }}, 2000);
        </script>
        """.format(popup_id, message, popup_id), unsafe_allow_html=True)

def play_alarm_sound():
    """Play a continuous beeping sound until stopped"""
    try:
        for _ in range(5):  # Beep 5 times
            winsound.Beep(1000, 500)  # 1000Hz frequency, 500ms duration
            time.sleep(0.2)  # Short pause between beeps
    except:
        print("Could not play alarm sound")

def stop_alarm():
    """Stop the alarm sound and reset notification state"""
    global alarm_thread, beep_active, notification_shown, alarm_shown, success_shown
    beep_active = False
    notification_shown = False
    alarm_shown = False
    success_shown = False
    if alarm_thread and alarm_thread.is_alive():
        alarm_thread.join(timeout=1)

def draw_pose(img, kps, label):
    skeleton = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    for x, y in kps:
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
    for i, j in skeleton:
        x1, y1 = kps[i]
        x2, y2 = kps[j]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(img, f"Posture: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img

def track_poor_posture(is_poor):
    global posture_timer, beep_active, notification_shown, alarm_shown, success_shown, alarm_thread
    current_time = time.time()
    
    if is_poor:
        if posture_timer['start'] is None:
            posture_timer['start'] = current_time
            # Reset all notification flags when starting new bad posture session
            notification_shown = False
            alarm_shown = False
            success_shown = False
            st.info(f"⚠️ Poor posture detected. Monitoring...")
        else:
            elapsed_time = current_time - posture_timer['start']
            
            if elapsed_time >= NOTIFICATION_THRESHOLD and elapsed_time < ALARM_THRESHOLD:
                # Stage 1: Pop-up notification (5-10 seconds) - show only once
                if not notification_shown:
                    notification_shown = True
                    remaining_alarm_time = max(0, ALARM_THRESHOLD - elapsed_time)
                    # Show pop-up notification once
                    show_popup_notification(f"Alarm in {remaining_alarm_time:.1f} seconds", "notification")
                
                # Show status without pop-up
                remaining_alarm_time = max(0, ALARM_THRESHOLD - elapsed_time)
                st.warning(f"📢 Notification shown. Alarm in {remaining_alarm_time:.1f}s")
                
            elif elapsed_time >= ALARM_THRESHOLD:
                # Stage 2: Audio alarm pop-up (after 10 seconds) - show only once
                if not alarm_shown:
                    alarm_shown = True
                    beep_active = True
                    # Show emergency pop-up alarm once
                    show_popup_notification(f"Poor posture for {elapsed_time:.1f} seconds", "alarm")
                    # Start alarm sound in a separate thread
                    alarm_thread = threading.Thread(target=play_alarm_sound, daemon=True)
                    alarm_thread.start()
                
                # Show status without pop-up
                st.error(f"🚨 Alarm triggered! Duration: {elapsed_time:.1f}s")
                
            elif elapsed_time < NOTIFICATION_THRESHOLD:
                # Before notification stage
                remaining_notification_time = max(0, NOTIFICATION_THRESHOLD - elapsed_time)
                st.info(f"⚠️ Poor posture detected for {elapsed_time:.1f}s. Notification in {remaining_notification_time:.1f}s")
    else:
        # Good posture detected - reset everything
        if posture_timer['start'] is not None:
            # Show success pop-up only once when transitioning from bad to good
            if not success_shown:
                success_shown = True
                show_popup_notification("Excellent posture!", "success")
        
        # Reset timer and flags
        posture_timer['start'] = None
        notification_shown = False
        alarm_shown = False
        beep_active = False
        
        # Stop any running alarm
        if alarm_thread and alarm_thread.is_alive():
            alarm_thread.join(timeout=0.1)

def log_event(label, path="posture_log.csv"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, mode="a") as f:
        f.write(f"{now},{label}\n")

# ------------------------------
# Streamlit Interface
# ------------------------------
st.title("🧍‍♂️ ProActive Watch - Posture Monitoring")
st.markdown("### Smart Pop-up Alert System (Shows once per stage)")
st.markdown("""
**How it works:**
- 🟡 **0-5s**: Monitoring poor posture (status updates)
- 🟠 **5s**: **One-time pop-up notification** appears and auto-hides after 2s
- 🔴 **10s**: **One-time emergency alarm** appears and auto-hides after 2s + audio beep
- ✅ **Good posture**: **One-time success message** when posture improves
- 🔄 **Smart logic**: No repeated pop-ups, only shows when state changes
""")

# Add control panel
col1, col2 = st.columns(2)
with col1:
    use_webcam = st.checkbox("Use webcam (live detection)")
with col2:
    if st.button("🔇 Stop Alarm"):
        stop_alarm()
        st.success("Alarm stopped!")

uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

# Status display
status_placeholder = st.empty()
FRAME_WINDOW = st.image([])

if use_webcam:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, save=False, verbose=False)

        if len(results[0].keypoints.data) > 0:
            kps = results[0].keypoints.data[0].cpu().numpy()[:, :2]
            dist = [np.linalg.norm(kps[i] - kps[j]) for i in range(len(kps)) for j in range(i + 1, len(kps))]

            if len(dist) > 0:
                pred = clf.predict([dist])[0]
                label = labels[pred]
                frame = draw_pose(frame, kps, label)
                log_event(label)
                track_poor_posture(label == "bad")
            else:
                label = "unknown"
                cv2.putText(frame, "Unable to extract features", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

elif uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        image_path = tmp_file.name
    image = cv2.imread(image_path)
    results = model.predict(source=image, save=False, verbose=False)

    if len(results[0].keypoints.data) > 0:
        kps = results[0].keypoints.data[0].cpu().numpy()[:, :2]
        dist = [np.linalg.norm(kps[i] - kps[j]) for i in range(len(kps)) for j in range(i + 1, len(kps))]
        if len(dist) > 0:
            pred = clf.predict([dist])[0]
            label = labels[pred]
            image = draw_pose(image, kps, label)
            log_event(label)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Posture: {label}", use_column_width=True)
        else:
            st.error("Unable to extract features from the keypoints.")
    else:
        st.error("No person detected in the image.")
