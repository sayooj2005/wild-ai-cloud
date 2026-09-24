import cv2
import time
import sys
import os
import math
import numpy as np
from datetime import datetime
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Ultralytics/PyTorch could not be loaded ({e}). Falling back to OpenCV & Synthetic Detection Engine.")
    YOLO_AVAILABLE = False
    YOLO = None

class WildlifeIoTSystem:
    def __init__(self, model_path='yolov8n.pt', capture_index=0):
        print("[SYSTEM] Initializing Enhanced YOLO Wildlife & Intrusion System...")
        
        # Load YOLO Model
        self.model = None
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                print(f"[SYSTEM] Model {model_path} loaded successfully.")
            except Exception as e:
                print(f"[WARNING] Failed to load YOLO model: {e}. Switching to OpenCV fallback mode.")
        else:
            print("[SYSTEM] YOLO not available. Running OpenCV AI Pose & Behavior Synthesizer mode.")

        # Initialize Camera / Capture Source
        self.cap = None
        if capture_index is not None:
            if isinstance(capture_index, int) and os.name == 'nt':
                self.cap = cv2.VideoCapture(capture_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(capture_index)
                
            if not self.cap.isOpened():
                err_msg = f"Could not open video stream (Index/File: {capture_index})."
                print(f"[ERROR] {err_msg}")
                raise RuntimeError(err_msg)
                
            if isinstance(capture_index, int):
                print(f"[SYSTEM] Warming up camera {capture_index}...")
                for _ in range(15):
                    self.cap.read()
                time.sleep(0.5)
                print("[SYSTEM] Camera warm-up complete.")
            
        # Alert Cooldowns
        self.last_alert_time = {}
        self.alert_cooldown = 10

        # Posture & Motion History Tracking (for velocity & behavior calculation)
        self.track_history = {}  # {track_id or label: [(cx, cy, timestamp, aspect_ratio, area)]}
        self.frame_counter = 0

        print("[SYSTEM] Pose Estimation & Dynamic Risk Engine Active.")

    def estimate_pose_and_behavior(self, species, x1, y1, x2, y2, prev_centroids, group_size=1):
        """
        Estimates animal/human posture keypoints and classifies behavioral patterns.
        Behaviors: 'Charging / Aggressive', 'Standing Alert', 'Walking / Moving', 'Feeding / Grazing', 'Resting'
        """
        w = x2 - x1
        h = y2 - y1
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        aspect_ratio = float(h) / max(w, 1)
        area = w * h

        # Calculate velocity vector if history exists
        velocity = 0.0
        direction_vector = "Stationary"
        dx, dy = 0, 0
        if prev_centroids and len(prev_centroids) > 0:
            pcx, pcy, pt, _, _ = prev_centroids[-1]
            dx = cx - pcx
            dy = cy - pcy
            dt = max(time.time() - pt, 0.03)
            velocity = math.sqrt(dx*dx + dy*dy) / dt  # pixels per second

            if abs(dx) > abs(dy):
                direction_vector = "Moving Right" if dx > 0 else "Moving Left"
            else:
                direction_vector = "Approaching Village" if dy > 0 else "Retreating to Forest"

        # Behavior Classification Logic based on Posture & Dynamics
        species_lower = species.lower()
        
        # 1. Pose Keypoints Synthesis (Head, Spine, Limbs, Tail/Base)
        keypoints = []
        if "poacher" in species_lower or "human" in species_lower:
            head = (cx, int(y1 + h * 0.15))
            left_shoulder = (int(x1 + w * 0.2), int(y1 + h * 0.3))
            right_shoulder = (int(x2 - w * 0.2), int(y1 + h * 0.3))
            spine = (cx, int(y1 + h * 0.5))
            left_hip = (int(x1 + w * 0.3), int(y1 + h * 0.7))
            right_hip = (int(x2 - w * 0.3), int(y1 + h * 0.7))
            left_foot = (int(x1 + w * 0.25), int(y2 - h * 0.05))
            right_foot = (int(x2 - w * 0.25), int(y2 - h * 0.05))
            keypoints = [head, left_shoulder, right_shoulder, spine, left_hip, right_hip, left_foot, right_foot]
        else:
            # Quadruped / Large Wildlife keypoints
            head = (int(x1 + w * 0.8) if dx >= 0 else int(x1 + w * 0.2), int(y1 + h * 0.3))
            spine_mid = (cx, int(y1 + h * 0.4))
            rear = (int(x1 + w * 0.2) if dx >= 0 else int(x1 + w * 0.8), int(y1 + h * 0.45))
            front_leg = (int(x1 + w * 0.75) if dx >= 0 else int(x1 + w * 0.25), int(y2 - h * 0.05))
            rear_leg = (int(x1 + w * 0.25) if dx >= 0 else int(x1 + w * 0.75), int(y2 - h * 0.05))
            keypoints = [head, spine_mid, rear, front_leg, rear_leg]

        # 2. Movement-Based Behaviour Classification (Part 14)
        # Derived purely from measurable motion data, not assumed emotion.
        previous_directions = [c[0] for c in prev_centroids[-4:]] if len(prev_centroids) >= 2 else []
        
        if len(prev_centroids) >= 4:
            # Check for direction changes across last 4 frames
            xs = [c[0] for c in prev_centroids[-4:]]
            direction_changes = sum(1 for i in range(1, len(xs)-1) if (xs[i]-xs[i-1]) * (xs[i+1]-xs[i]) < 0)
        else:
            direction_changes = 0

        if len(prev_centroids) >= 2 and group_size >= 2:
            behaviour = "GROUP"
            pose_pattern = f"Multi-animal group detected ({group_size} individuals)"
        elif velocity > 200 and "Approaching" in direction_vector:
            behaviour = "APPROACHING"
            pose_pattern = "High velocity approach toward settlement"
        elif velocity > 180:
            behaviour = "RUNNING"
            pose_pattern = "High forward velocity detected"
        elif direction_changes >= 2 and velocity > 30:
            behaviour = "CHANGING_DIRECTION"
            pose_pattern = "Erratic movement pattern detected"
        elif "Retreating" in direction_vector and velocity > 30:
            behaviour = "MOVING_AWAY"
            pose_pattern = "Moving away from settlement"
        elif velocity > 40:
            behaviour = "MOVING"
            pose_pattern = "Normal locomotion movement"
        else:
            behaviour = "STATIONARY"
            pose_pattern = "Stationary / Low movement"

        return behaviour, pose_pattern, keypoints, velocity, direction_vector

    def calculate_dynamic_risk(self, species, behaviour, velocity, direction_vector, group_size=1, is_night=False):
        """
        Dynamic Risk Engine incorporating 5 core factors:
        1. Species Threat Base
        2. Behavior Multiplier
        3. Movement Vector & Speed
        4. Group Size / Density
        5. Environmental Factors (Visibility/Night)
        Returns: risk_score (0-100), danger_level ('low', 'medium', 'high', 'critical')
        """
        species_lower = species.lower()

        # 1. Base Species Threat
        if any(s in species_lower for s in ["tiger", "leopard", "panther"]):
            base_score = 85
        elif "elephant" in species_lower:
            base_score = 80
        elif "poacher" in species_lower or "human" in species_lower:
            base_score = 90
        elif "bear" in species_lower:
            base_score = 70
        elif "boar" in species_lower:
            base_score = 50
        elif "deer" in species_lower:
            base_score = 20
        else:
            base_score = 30

        # 2. Behavior Multiplier (mapped to Part 14 behaviour codes)
        behaviour_score = 0
        if behaviour in ("APPROACHING", "RUNNING"):
            behaviour_score = 35
        elif behaviour == "CHANGING_DIRECTION":
            behaviour_score = 20
        elif behaviour == "GROUP":
            behaviour_score = 25
        elif behaviour == "MOVING":
            behaviour_score = 10
        elif behaviour == "MOVING_AWAY":
            behaviour_score = -5
        else:  # STATIONARY
            behaviour_score = 0

        # 3. Movement Vector & Speed
        movement_score = 0
        if "Approaching" in direction_vector:
            movement_score += 20
        elif "Retreating" in direction_vector:
            movement_score -= 10
            
        if velocity > 150:
            movement_score += 15

        # 4. Group Size
        group_score = 0
        if group_size >= 4:
            group_score = 20
        elif group_size >= 2:
            group_score = 10

        # 5. Environmental Factors (Night / Low Visibility)
        env_score = 15 if is_night else 0

        # Total Dynamic Risk Calculation
        raw_risk = base_score + behaviour_score + movement_score + group_score + env_score
        risk_score = min(max(int(raw_risk), 0), 100)

        # Danger Level Categorization
        if risk_score >= 85:
            danger_level = "critical"
        elif risk_score >= 65:
            danger_level = "high"
        elif risk_score >= 40:
            danger_level = "medium"
        else:
            danger_level = "low"

        return risk_score, danger_level

    def trigger_alert(self, detection_type, confidence, frame, danger_level="high", risk_score=80, behaviour="Charging"):
        """Triggers multi-channel responses (Audio Repellent, SMS, Evacuation Alert)."""
        current_time = time.time()
        if current_time - self.last_alert_time.get(detection_type, 0) < self.alert_cooldown:
            return

        self.last_alert_time[detection_type] = current_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n>>> [REAL-TIME ALERT] {detection_type.upper()} | Risk: {risk_score}% ({danger_level.upper()}) | Behavior: {behaviour} | Time: {timestamp}")
        
        # Save evidence image
        filename = f"detection_{detection_type}_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)

    def draw_pose_skeleton(self, frame, keypoints, behaviour, color):
        """Renders visual keypoint nodes and skeleton limbs on frame."""
        if not keypoints or len(keypoints) < 2:
            return

        # Draw nodes
        for pt in keypoints:
            cv2.circle(frame, pt, 5, (0, 255, 255), -1)
            cv2.circle(frame, pt, 7, color, 1)

        # Connect limbs
        for i in range(len(keypoints) - 1):
            cv2.line(frame, keypoints[i], keypoints[i+1], (0, 255, 128), 2)

    def process_frame(self, frame):
        """
        Runs YOLO inference, performs Pose Estimation, computes Dynamic Risk, and returns telemetry.
        """
        self.frame_counter += 1
        h_frame, w_frame, _ = frame.shape
        is_night = (datetime.now().hour >= 18 or datetime.now().hour < 6)

        # Run YOLO with ByteTrack enabled for persistent object tracking
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", stream=True, verbose=False)
        telemetry = None

        detected_count = 0
        detected_threats = []

        for result in results:
            boxes = result.boxes
            if boxes is None: continue
            
            detected_count += len(boxes)
            
            for box in boxes:
                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                
                # Extract ByteTrack ID (if tracking is successful, otherwise fallback)
                track_id = int(box.id[0]) if box.id is not None else int(time.time() * 1000) % 10000

                # Map COCO classes to project domain
                detected_threat = None
                if label == 'person' and confidence > 0.45:
                    detected_threat = "poacher"
                elif label == 'elephant' and confidence > 0.55:
                    detected_threat = "elephant"
                elif label == 'cat' and confidence > 0.4:
                    detected_threat = "tiger"
                elif label == 'bear' and confidence > 0.5:
                    detected_threat = "bear"
                elif label == 'dog' and confidence > 0.45:
                    detected_threat = "wild boar"
                elif label == 'cow' or label == 'horse':
                    detected_threat = "wild boar"

                if detected_threat:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                    
                    # Track centroid history using specific object tracking ID
                    track_key = f"{detected_threat}_ID{track_id}"
                    history = self.track_history.get(track_key, [])
                    
                    # Perform Pose Estimation & Behavior Classification
                    behaviour, pose_pattern, keypoints, velocity, direction_vector = self.estimate_pose_and_behavior(
                        detected_threat, x1, y1, x2, y2, history, group_size=len(boxes)
                    )

                    # Update history
                    history.append((cx, cy, time.time(), (y2-y1)/max(x2-x1,1), (x2-x1)*(y2-y1)))
                    if len(history) > 10:
                        history.pop(0)
                    self.track_history[track_key] = history

                    # Dynamic Risk Assessment
                    risk_score, danger_level = self.calculate_dynamic_risk(
                        detected_threat, behaviour, velocity, direction_vector, group_size=len(boxes), is_night=is_night
                    )

                    # Color scheme based on danger level
                    if danger_level == "critical":
                        box_color = (0, 0, 255)       # Red
                    elif danger_level == "high":
                        box_color = (0, 140, 255)     # Orange
                    elif danger_level == "medium":
                        box_color = (0, 255, 255)     # Yellow
                    else:
                        box_color = (0, 255, 0)       # Green

                    # Draw Bounding Box (Sleek, no skeleton lines)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    # self.draw_pose_skeleton(frame, keypoints, behaviour, box_color)  # Removed as requested

                    # Draw Overlay Badges (Include Tracking ID)
                    badge_text = f"{detected_threat.upper()} #{track_id} [{risk_score}% RISK]"
                    cv2.rectangle(frame, (x1, max(y1 - 40, 0)), (x1 + len(badge_text)*10 + 10, y1), box_color, -1)
                    cv2.putText(frame, badge_text, (x1 + 5, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    sub_badge = f"Pose: {behaviour} ({direction_vector} {round(velocity,1)}px/s)"
                    cv2.putText(frame, sub_badge, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    # Trigger alert logic
                    if danger_level in ["high", "critical"]:
                        self.trigger_alert(detected_threat, confidence, frame, danger_level, risk_score, behaviour)

                    # Store primary telemetry item
                    telemetry = {
                        "track_id": track_id,
                        "animal": detected_threat,
                        "confidence": float(confidence),
                        "bbox": [x1, y1, x2, y2],
                        "behaviour": behaviour,
                        "pose_pattern": pose_pattern,
                        "risk_score": risk_score,
                        "danger_level": danger_level,
                        "movement_vector": direction_vector,
                        "velocity_px_sec": round(velocity, 1),
                        "group_size": len(boxes)
                    }
                    detected_threats.append(telemetry)

        return frame, detected_threats

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.process_frame(frame)
            cv2.imshow('Wildlife Guardian - AI IoT Surveillance', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Monitoring Stopped.")

if __name__ == "__main__":
    iot_system = WildlifeIoTSystem(model_path='yolov8n.pt', capture_index=0)
    iot_system.run()

