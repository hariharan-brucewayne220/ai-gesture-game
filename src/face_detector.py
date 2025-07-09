"""
Face Detector - Wink detection for aiming using MediaPipe Face Mesh
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Tuple, Optional

class FaceDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Eye landmarks (MediaPipe Face Mesh indices)
        self.left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Simplified eye landmarks for EAR calculation
        self.left_eye_key_points = [33, 160, 158, 133, 153, 144]  # outer, top, bottom, inner, top2, bottom2
        self.right_eye_key_points = [362, 385, 387, 263, 373, 380]  # outer, top, bottom, inner, top2, bottom2
        
        # Thresholds for eye states - adjusted based on your EAR values
        self.EAR_THRESHOLD = 0.25  # Below this = eye closed
        self.WINK_THRESHOLD = 0.18  # More strict threshold for wink (based on your 0.14-0.16 values)
        self.OPEN_THRESHOLD = 0.25  # Above this = eye definitely open
        
        # Wink state tracking
        self.wink_start_time = None
        self.is_aiming = False
        self.min_wink_duration = 0.1  # Minimum wink duration (seconds)
        self.max_wink_duration = 5.0  # Maximum wink duration before reset
        
        # Stability tracking - require consistent detection
        self.consecutive_winks = 0
        self.consecutive_opens = 0
        self.required_consistency = 3  # Require 3 consecutive detections
        
        # Distance validation
        self.min_face_size = 50  # Minimum face bounding box size
        self.last_face_distance = 0
        
        # Debug mode
        self.debug_mode = True
        
    def calculate_ear(self, eye_landmarks, landmarks):
        """Calculate Eye Aspect Ratio (EAR) for given eye landmarks"""
        if len(eye_landmarks) < 6:
            return 0.0
            
        try:
            # Get landmark coordinates
            points = []
            for idx in eye_landmarks:
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]
                    points.append([lm.x, lm.y])
                else:
                    return 0.0
            
            if len(points) < 6:
                return 0.0
                
            points = np.array(points)
            
            # Calculate vertical distances
            A = np.linalg.norm(points[1] - points[5])  # Top to bottom
            B = np.linalg.norm(points[2] - points[4])  # Top2 to bottom2
            
            # Calculate horizontal distance
            C = np.linalg.norm(points[0] - points[3])  # Outer to inner corner
            
            if C == 0:
                return 0.0
                
            # Eye Aspect Ratio
            ear = (A + B) / (2.0 * C)
            return ear
            
        except Exception as e:
            if self.debug_mode:
                print(f"EAR calculation error: {e}")
            return 0.0
    
    def calculate_face_distance(self, landmarks, image_shape):
        """Estimate face distance based on face landmarks"""
        try:
            # Use face outline landmarks to estimate size
            face_points = [10, 152, 234, 454]  # Top, bottom, left, right of face
            
            coords = []
            for idx in face_points:
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]
                    x = int(lm.x * image_shape[1])
                    y = int(lm.y * image_shape[0])
                    coords.append([x, y])
            
            if len(coords) == 4:
                # Calculate face bounding box size
                coords = np.array(coords)
                width = np.max(coords[:, 0]) - np.min(coords[:, 0])
                height = np.max(coords[:, 1]) - np.min(coords[:, 1])
                face_size = (width + height) / 2
                return face_size
            
        except Exception as e:
            if self.debug_mode:
                print(f"Face distance calculation error: {e}")
        
        return 0
    
    def detect_wink(self, frame) -> Tuple[bool, str, float]:
        """
        Detect wink in frame
        Returns: (is_aiming, wink_type, confidence)
        """
        if frame is None:
            return False, "none", 0.0
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            # No face detected - stop aiming
            if self.is_aiming:
                self.is_aiming = False
                self.wink_start_time = None
                if self.debug_mode:
                    print("👁️ No face detected - stopped aiming")
            return False, "none", 0.0
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Calculate face distance for validation
        face_distance = self.calculate_face_distance(face_landmarks, frame.shape)
        self.last_face_distance = face_distance
        
        if face_distance < self.min_face_size:
            if self.debug_mode and self.is_aiming:
                print(f"👁️ Face too far away: {face_distance:.1f} < {self.min_face_size}")
            return False, "too_far", 0.0
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(self.left_eye_key_points, face_landmarks)
        right_ear = self.calculate_ear(self.right_eye_key_points, face_landmarks)
        
        if left_ear == 0.0 or right_ear == 0.0:
            return False, "calculation_error", 0.0
        
        current_time = time.time()
        
        # Detect wink patterns
        left_closed = left_ear < self.WINK_THRESHOLD
        right_closed = right_ear < self.WINK_THRESHOLD
        left_open = left_ear > self.OPEN_THRESHOLD
        right_open = right_ear > self.OPEN_THRESHOLD
        
        # Check for wink (one eye closed, other open)
        is_winking = (left_closed and right_open) or (right_closed and left_open)
        
        # Check for blink (both eyes closed) - ignore this
        is_blinking = left_closed and right_closed
        
        if self.debug_mode:
            wink_status = ""
            if is_winking:
                if left_closed and right_open:
                    wink_status = "LEFT_WINK"
                elif right_closed and left_open:
                    wink_status = "RIGHT_WINK"
            elif is_blinking:
                wink_status = "BLINK"
            else:
                wink_status = "OPEN"
            
            # Only print debug info occasionally to avoid spam
            if current_time % 1 < 0.1:  # Every ~1 second
                aiming_status = "AIMING" if self.is_aiming else "NOT_AIMING"
                timer_status = f"Timer:{current_time - self.wink_start_time:.2f}s" if self.wink_start_time else "NoTimer"
                stability_status = f"W:{self.consecutive_winks}/O:{self.consecutive_opens}"
                print(f"👁️ L:{left_ear:.3f} R:{right_ear:.3f} | {wink_status} | {aiming_status} | {timer_status} | {stability_status} | Dist:{face_distance:.1f}")
        
        # Update stability counters
        if is_winking:
            self.consecutive_winks += 1
            self.consecutive_opens = 0
        elif not is_winking and not is_blinking:
            self.consecutive_opens += 1
            self.consecutive_winks = 0
        else:
            # Blink - reset both counters
            self.consecutive_winks = 0
            self.consecutive_opens = 0
        
        # State machine for wink detection with stability
        if is_winking and self.consecutive_winks >= self.required_consistency:
            # Stable wink detected
            if not self.is_aiming:
                # Start aiming process
                if self.wink_start_time is None:
                    self.wink_start_time = current_time
                    if self.debug_mode:
                        print(f"👁️ Stable wink detected ({self.consecutive_winks} frames) - starting timer")
                elif current_time - self.wink_start_time > self.min_wink_duration:
                    # Wink held long enough - start aiming
                    self.is_aiming = True
                    wink_type = "left_wink" if left_closed else "right_wink"
                    if self.debug_mode:
                        print(f"👁️ Started aiming with {wink_type}")
                    return True, wink_type, 0.9
            else:
                # Already aiming, continue
                wink_type = "left_wink" if left_closed else "right_wink"
                if self.debug_mode and current_time % 2 < 0.1:  # Debug every 2 seconds
                    print(f"👁️ Continuing aim with {wink_type}")
                return True, wink_type, 0.8
                
        elif not is_winking and not is_blinking and self.consecutive_opens >= self.required_consistency:
            # Stable eyes open - stop aiming
            if self.is_aiming:
                self.is_aiming = False
                self.wink_start_time = None
                if self.debug_mode:
                    print(f"👁️ Stable eyes open ({self.consecutive_opens} frames) - stopped aiming")
                return False, "aim_stop", 0.9
            else:
                # Reset wink timing if not aiming
                self.wink_start_time = None
            
        elif is_blinking:
            # Blink detected - don't change aiming state, just reset timer
            if self.debug_mode and self.consecutive_winks > 0:  # Only show if we were tracking a wink
                print("👁️ Blink detected - ignoring")
            self.wink_start_time = None
            return self.is_aiming, "blink", 0.0
            
        return self.is_aiming, "none", 0.0
    
    def draw_debug_info(self, frame, is_aiming, wink_type, confidence):
        """Draw debug information on frame"""
        if not self.debug_mode:
            return frame
            
        # Draw aiming status
        status_text = f"AIM: {'ON' if is_aiming else 'OFF'}"
        status_color = (0, 255, 0) if is_aiming else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        if wink_type != "none":
            cv2.putText(frame, f"Wink: {wink_type} ({confidence:.2f})", 
                       (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Draw face distance
        cv2.putText(frame, f"Face Distance: {self.last_face_distance:.1f}", 
                   (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def toggle_debug(self):
        """Toggle debug mode"""
        self.debug_mode = not self.debug_mode
        print(f"👁️ Face detector debug: {'ON' if self.debug_mode else 'OFF'}")
        return self.debug_mode

# Example usage
if __name__ == "__main__":
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)
    
    print("👁️ Wink to aim! Close one eye to start aiming, open to stop.")
    print("Press 'q' to quit, 'd' to toggle debug")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # Detect wink
        is_aiming, wink_type, confidence = detector.detect_wink(frame)
        
        # Draw debug info
        frame = detector.draw_debug_info(frame, is_aiming, wink_type, confidence)
        
        cv2.imshow("Wink Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            detector.toggle_debug()
    
    cap.release()
    cv2.destroyAllWindows()