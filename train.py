
from ultralytics import YOLO
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------
# Step 1: Load YOLO Pose Model
# ------------------------------
# Handle PyTorch 2.6+ security restrictions
import torch
import ultralytics.nn.tasks

# Allow ultralytics classes to be loaded safely
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.PoseModel,
    ultralytics.nn.tasks.DetectionModel,
    ultralytics.models.yolo.pose.PosePredictor,
    ultralytics.models.yolo.detect.DetectionPredictor
])

model = YOLO("yolo11n-pose.pt")  # or yolov8n-pose.pt if using older version

# ------------------------------
# Step 2: Inference on Images
# ------------------------------
image_folder = "images"  # Replace with your folder
labels = {"good": 0, "bad": 1}

features = []
targets = []

for label_name, label_value in labels.items():
    folder_path = os.path.join(image_folder, label_name)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            results = model.predict(source=image_path, save=False, verbose=False)

            if len(results) == 0 or len(results[0].keypoints.data) == 0:
                continue  # Skip if no person detected

            keypoints = results[0].keypoints.data[0].cpu().numpy()[:, :2]  # Take (x, y) only
            
            # Ensure we have exactly 17 keypoints (YOLO pose standard)
            if len(keypoints) != 17:
                print(f"Skipping {filename}: Expected 17 keypoints, got {len(keypoints)}")
                continue

            # Feature: pairwise distances between keypoints
            dist = []
            for i in range(len(keypoints)):
                for j in range(i + 1, len(keypoints)):
                    d = np.linalg.norm(keypoints[i] - keypoints[j])
                    dist.append(d)

            # Verify consistent feature dimension (should be 136 for 17 keypoints)
            if len(dist) == 136:  # C(17,2) = 17*16/2 = 136
                features.append(dist)
                targets.append(label_value)
            else:
                print(f"Skipping {filename}: Expected 136 features, got {len(dist)}")

# ------------------------------
# Step 3: Train Classifier
# ------------------------------
print(f"Total samples collected: {len(features)}")
print(f"Feature dimension: {len(features[0]) if features else 'No features'}")
print(f"Class distribution: {dict(zip(*np.unique(targets, return_counts=True)))}")

if len(features) == 0:
    print("No valid samples found! Check your images and model.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ------------------------------
# Step 4: Evaluate
# ------------------------------

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=list(labels.keys())))

joblib.dump(clf, "pose_classifier.pkl")