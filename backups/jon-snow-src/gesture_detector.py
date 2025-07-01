import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List, Tuple

class GestureDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Gesture state tracking
        self.current_gesture = None
        self.gesture_confidence = 0.0
        self.gesture_history = []
        
    def get_hand_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None
    
    def extract_features(self, landmarks) -> Dict[str, float]:
        """Extract gesture features from landmarks"""
        if not landmarks:
            return {}
            
        # Convert landmarks to numpy array
        points = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
        
        # Key landmark indices (MediaPipe hand model)
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20
        WRIST = 0
        INDEX_MCP = 5  # Index finger base for direction calculation
        
        # Calculate finger extensions
        # Improved thumb detection - check if thumb is away from palm
        thumb_extended = abs(points[THUMB_TIP][0] - points[WRIST][0]) > 0.06 or abs(points[THUMB_TIP][1] - points[WRIST][1]) > 0.08
        index_extended = points[INDEX_TIP][1] < points[INDEX_TIP - 2][1]   # Up = extended
        middle_extended = points[MIDDLE_TIP][1] < points[MIDDLE_TIP - 2][1]
        ring_extended = points[RING_TIP][1] < points[RING_TIP - 2][1]
        pinky_extended = points[PINKY_TIP][1] < points[PINKY_TIP - 2][1]
        
        # Calculate pointing direction (for directional gestures)
        index_direction_x = points[INDEX_TIP][0] - points[INDEX_MCP][0]  # Left/Right
        index_direction_y = points[INDEX_TIP][1] - points[INDEX_MCP][1]  # Up/Down
        
        # Calculate distances for fist detection
        avg_finger_to_wrist = np.mean([
            np.linalg.norm(points[tip] - points[WRIST]) 
            for tip in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        ])
        
        return {
            'thumb_extended': thumb_extended,
            'index_extended': index_extended,
            'middle_extended': middle_extended,
            'ring_extended': ring_extended,
            'pinky_extended': pinky_extended,
            'avg_distance_to_wrist': avg_finger_to_wrist,
            'extended_fingers': sum([index_extended, middle_extended, ring_extended, pinky_extended]),
            'total_fingers': sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]),
            'index_direction_x': index_direction_x,
            'index_direction_y': index_direction_y
        }
    
    def classify_gesture(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Classify gesture from features"""
        if not features:
            return "none", 0.0
            
        # Gesture classification rules
        extended = features['extended_fingers']
        thumb = features['thumb_extended']
        index = features['index_extended']
        middle = features['middle_extended']
        ring = features['ring_extended']
        pinky = features['pinky_extended']
        direction_x = features['index_direction_x']
        direction_y = features['index_direction_y']
        
        # Open Palm - All fingers extended (Forward)
        if extended >= 4:
            return "open_palm", 0.9
            
        # Closed Fist - No fingers extended (Backward)
        if extended == 0 and features['avg_distance_to_wrist'] < 0.15:
            return "closed_fist", 0.9
            
        # Directional Index Pointing - Only index finger extended
        if index and not middle and not ring and not pinky:
            # Point Left - Index pointing left
            if direction_x < -0.08:
                return "point_left", 0.85
            # Point Right - Index pointing right  
            elif direction_x > 0.08:
                return "point_right", 0.85
            # Point Up - Index pointing up (Jump)
            elif direction_y < -0.08:
                return "point_up", 0.85
            # Point Forward - Index pointing straight
            else:
                return "point_forward", 0.8
            
        # Rock Sign - Index and pinky extended (Attack)
        if index and pinky and not middle and not ring:
            return "rock_sign", 0.8
            
        return "unknown", 0.3
    
    def stabilize_gesture(self, gesture: str, confidence: float) -> Tuple[str, float]:
        """Stabilize gestures using history to reduce noise"""
        self.gesture_history.append((gesture, confidence))
        
        # Keep only recent history
        if len(self.gesture_history) > 5:
            self.gesture_history.pop(0)
            
        # Count occurrences of each gesture
        gesture_counts = {}
        total_confidence = {}
        
        for g, c in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
            total_confidence[g] = total_confidence.get(g, 0) + c
            
        # Find most common gesture with good confidence
        best_gesture = max(gesture_counts.keys(), key=lambda x: gesture_counts[x])
        avg_confidence = total_confidence[best_gesture] / gesture_counts[best_gesture]
        
        # Only return stable gestures
        if gesture_counts[best_gesture] >= 3 and avg_confidence > 0.7:
            return best_gesture, avg_confidence
            
        return "none", 0.0
    
    def process_frame(self) -> Tuple[Optional[np.ndarray], str, float]:
        """Process single frame and return frame, gesture, confidence"""
        ret, frame = self.cap.read()
        if not ret:
            return None, "none", 0.0
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Get hand landmarks
        landmarks = self.get_hand_landmarks(frame)
        
        if landmarks:
            # Draw landmarks on frame
            self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Extract features and classify
            features = self.extract_features(landmarks)
            raw_gesture, raw_confidence = self.classify_gesture(features)
            
            # Stabilize gesture
            stable_gesture, stable_confidence = self.stabilize_gesture(raw_gesture, raw_confidence)
            
            self.current_gesture = stable_gesture
            self.gesture_confidence = stable_confidence
            
            # Add gesture text to frame
            cv2.putText(frame, f"Gesture: {stable_gesture} ({stable_confidence:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Debug info - show finger count and thumb status
            debug_text = f"4-Fingers: {features['extended_fingers']} | Total: {features['total_fingers']} | Thumb: {features['thumb_extended']}"
            cv2.putText(frame, debug_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                       
            return frame, stable_gesture, stable_confidence
        else:
            # No hand detected
            cv2.putText(frame, "No hand detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, "none", 0.0
    
    def release(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()