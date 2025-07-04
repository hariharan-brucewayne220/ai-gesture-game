import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Optional, Dict, Tuple
from pynput.keyboard import Key

# Custom gesture AI import
try:
    from custom_gesture_ai import CustomGestureAI
    CUSTOM_AI_AVAILABLE = True
except ImportError:
    CUSTOM_AI_AVAILABLE = False

class GestureDetector:
    def __init__(self, google_ai_key=None):
        # Initialize MediaPipe for single hand detection (original working config)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Single hand only - original working configuration
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
        
        # Hand camera control - Velocity-based movement
        self.hand_camera_enabled = True  # ON by default
        self.hand_reference_position = None
        self.last_hand_position = None
        self.camera_sensitivity_x = 5.0  # Increased for larger movement range
        self.camera_sensitivity_y = 4.0  # Increased for larger movement range
        self.camera_deadzone = 0.01     # Movement threshold
        
        # Simple smoothing for fluid camera movement
        self.smoothed_velocity_x = 0.0
        self.smoothed_velocity_y = 0.0
        self.smoothing_factor = 0.4  # Higher = more responsive (was too low before)
        
        # Parent controller reference
        self.parent_controller = None
        
        # Training mode (for compatibility with main.py)
        self.training_mode = False
        
        # Custom AI attributes (for compatibility)
        self.custom_ai = None
        self.custom_ai_enabled = False
        
        # Local CNN system for offline recognition
        self.local_trainer = None
        self.use_local_model = False
        self.ai_model_generator = None
        
        # MLP gesture trainer for landmark-based recognition
        self.mlp_trainer = None
        self.use_mlp_model = False
        
        # Initialize MLP trainer
        try:
            from mlp_gesture_trainer import MLPGestureTrainer
            self.mlp_trainer = MLPGestureTrainer()
            if self.mlp_trainer.load_model():
                self.use_mlp_model = True
                print("🎯 MLP gesture system initialized")
            else:
                print("📦 MLP gesture system ready for calibration")
        except Exception as e:
            print(f"⚠️ MLP gesture system not available: {e}")
            self.mlp_trainer = None
        
        
    def get_hand_landmarks(self, frame):
        """Extract single hand landmarks from frame (original working method)"""
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
        
        # Calculate finger curl tightness (for distinguishing closed vs hollow fist)
        # Measure distances between fingertips and palm center
        palm_center = (points[WRIST] + points[INDEX_MCP]) / 2  # Approximate palm center
        finger_curl_distances = [
            np.linalg.norm(points[tip] - palm_center) 
            for tip in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        ]
        max_curl_distance = max(finger_curl_distances)
        avg_curl_distance = np.mean(finger_curl_distances)
        
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
            'index_direction_y': index_direction_y,
            'max_curl_distance': max_curl_distance,
            'avg_curl_distance': avg_curl_distance
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
            
        # Closed Fist - No fingers extended (Backward) - Simple like old code
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
            # Point Up - Index pointing up (Forward) - More sensitive
            elif direction_y < -0.05:
                return "point_up", 0.85
            # Point Forward - Index pointing straight
            else:
                return "point_forward", 0.8
            
        # Rock Sign - Index and pinky extended (Attack)
        if index and pinky and not middle and not ring:
            return "rock_sign", 0.8
        
        # C Shape - Thumb and index extended, curved (Crouch)
        if thumb and index and extended == 2:
            return "c_shape", 0.8
            
        return "unknown", 0.3
    
    def stabilize_gesture(self, gesture: str, confidence: float) -> Tuple[str, float]:
        """Stabilize gestures using history to reduce noise"""
        self.gesture_history.append((gesture, confidence))
        
        # Keep only recent history (reduced for faster response)
        if len(self.gesture_history) > 3:
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
        
        # Only return stable gestures (reduced requirements for faster response)
        if gesture_counts[best_gesture] >= 2 and avg_confidence > 0.6:
            return best_gesture, avg_confidence
            
        return "none", 0.0
    
    def calculate_hand_camera_movement(self, landmarks, image_size):
        """Calculate camera movement based on hand velocity with smooth interpolation"""
        if not landmarks or not self.hand_camera_enabled:
            return 0, 0
            
        # Use wrist position (landmark 0) as tracking point
        wrist = landmarks.landmark[0]
        current_position = (wrist.x, wrist.y)
        
        # Initialize positions if not set
        if self.last_hand_position is None:
            self.last_hand_position = current_position
            return 0, 0
        
        # Calculate raw velocity (movement since last frame)
        raw_velocity_x = current_position[0] - self.last_hand_position[0]
        raw_velocity_y = current_position[1] - self.last_hand_position[1]
        
        # Using simple exponential smoothing for stability
        
        # Apply exponential smoothing for fluid movement
        # Formula: smoothed = (1-α) * old_smoothed + α * raw_value
        self.smoothed_velocity_x = (1 - self.smoothing_factor) * self.smoothed_velocity_x + self.smoothing_factor * raw_velocity_x
        self.smoothed_velocity_y = (1 - self.smoothing_factor) * self.smoothed_velocity_y + self.smoothing_factor * raw_velocity_y
        
        # Update last position for next frame
        self.last_hand_position = current_position
        
        # Simple, responsive movement - back to basics but better
        mouse_dx = 0.0
        mouse_dy = 0.0
        
        # Only move if above deadzone
        if abs(self.smoothed_velocity_x) > self.camera_deadzone:
            # Simple linear scaling with floating point precision
            mouse_dx = self.smoothed_velocity_x * self.camera_sensitivity_x * 800
            
        if abs(self.smoothed_velocity_y) > self.camera_deadzone:
            mouse_dy = self.smoothed_velocity_y * self.camera_sensitivity_y * 800
        
        # Apply floating point precision for micro-movements
        return mouse_dx, mouse_dy
    
    def set_hand_camera_reference(self):
        """Reset hand camera tracking"""
        self.last_hand_position = None
        self.smoothed_velocity_x = 0.0
        self.smoothed_velocity_y = 0.0
        print("📍 Hand camera tracking reset")
    
    def toggle_hand_camera(self):
        """Toggle hand camera control on/off"""
        self.hand_camera_enabled = not self.hand_camera_enabled
        if self.hand_camera_enabled:
            self.set_hand_camera_reference()  # Reset tracking when enabling
            print(f"✅ Hand camera control ENABLED (Simple & responsive)")
        else:
            print("❌ Hand camera control DISABLED")
        return self.hand_camera_enabled
    
    def adjust_camera_sensitivity(self, increase: bool = True):
        """Adjust camera sensitivity up or down"""
        if increase:
            self.camera_sensitivity_x = min(2.0, self.camera_sensitivity_x + 0.1)
            self.camera_sensitivity_y = min(1.6, self.camera_sensitivity_y + 0.08)
        else:
            self.camera_sensitivity_x = max(0.1, self.camera_sensitivity_x - 0.1)
            self.camera_sensitivity_y = max(0.08, self.camera_sensitivity_y - 0.08)
        
        print(f"📹 Hand camera sensitivity: {self.camera_sensitivity_x:.1f}x")
    
    def adjust_camera_smoothing(self, increase: bool = True):
        """Adjust camera smoothing for more/less fluid movement"""
        if increase:
            self.smoothing_factor = min(0.8, self.smoothing_factor + 0.1)
        else:
            self.smoothing_factor = max(0.1, self.smoothing_factor - 0.1)
        
        responsiveness = "More Responsive" if increase else "Smoother"
        print(f"🌊 Camera smoothing: {responsiveness} ({self.smoothing_factor:.1f})")
    
    
    
    
    def start_calibration(self):
        """Start full MLP calibration for WSAD + default gestures"""
        if not self.mlp_trainer:
            print("❌ MLP trainer not available")
            return False
        
        print("🎯 Starting MLP Gesture Calibration")
        success = self.mlp_trainer.start_calibration()
        
        if success:
            self.use_mlp_model = True
            print("✅ MLP calibration completed!")
            return True
        
        print("❌ MLP calibration failed")
        return False
    
    def add_custom_gesture(self, gesture_name: str, key_mapping: str):
        """Add a custom gesture after initial calibration"""
        if not self.mlp_trainer:
            print("❌ MLP trainer not available")
            return False
        
        success = self.mlp_trainer.add_custom_gesture(gesture_name, key_mapping)
        if success:
            self.use_mlp_model = True
            print(f"✅ Custom gesture '{gesture_name}' added!")
        return success

    
    def start_override_training(self, movement_key: str):
        """Start training mode for existing movement keys with AI override"""
        if not self.custom_ai:
            print("❌ Custom AI not available - no API key provided")
            return False
        
        # Map movement keys to AI gesture names
        key_to_ai_name = {
            "w": ("ai_point_up", "AI-trained forward movement - any hand angle"),
            "a": ("ai_point_left", "AI-trained left movement - any hand angle"), 
            "s": ("ai_closed_fist", "AI-trained backward movement - any fist type"),
            "d": ("ai_point_right", "AI-trained right movement - any hand angle"),
            "space": ("ai_open_palm", "AI-trained jump - any open hand"),
            "attack": ("ai_rock_sign", "AI-trained attack - any rock sign")
        }
        
        if movement_key not in key_to_ai_name:
            print(f"❌ Unknown movement key: {movement_key}")
            return False
        
        ai_name, description = key_to_ai_name[movement_key]
        
        self.training_mode = True
        self.training_frames = []
        self.training_gesture_name = ai_name
        self.training_description = description
        self.override_mode = True
        
        print(f"🎯 OVERRIDE Training for movement key: {movement_key.upper()}")
        print(f"🤖 AI Gesture: {ai_name}")
        print(f"📝 Description: {description}")
        print("💡 This will replace the built-in detection with AI!")
        print("👋 Try different hand angles - palm side, back side, etc.")
        print("📸 Press SPACE to capture examples")
        print("🔄 Capture 10-15 examples for best accuracy")
        return True
    
    def capture_training_frame(self, frame):
        """
        Capture training data including landmarks and images
        """
        if not self.training_mode:
            return False
        
        # Get current hand landmarks
        landmarks = self.get_hand_landmarks(frame)
        if landmarks is None:
            print("❌ No hand detected - try again")
            return False
        
        # Store both frame and landmarks
        self.training_frames.append({
            'frame': frame.copy(),
            'landmarks': landmarks,
            'timestamp': time.time()
        })
        
        # Add to local trainer if available
        if self.local_trainer and hasattr(self, 'training_key_mapping'):
            self.local_trainer.add_training_sample(
                landmarks, 
                self.training_gesture_name, 
                self.training_key_mapping
            )
        
        print(f"📸 Captured example {len(self.training_frames)} (landmarks + image)")
        return True
    
    def finish_training(self):
        """
        Finish training with multi-modal approach
        Trains local model and optionally cloud AI for reliability
        """
        if not self.training_mode or not self.training_frames:
            print("❌ No training data to save")
            return False
        
        if len(self.training_frames) < 5:
            print(f"⚠️ Only {len(self.training_frames)} examples captured. Need at least 5 for good accuracy.")
            return False
        
        print("🚀 Starting training process...")
        
        # 1. Train local model (primary method)
        if self.local_trainer:
            print("🧠 Training local model...")
            if self.local_trainer.train_model():
                self.use_local_model = True
                print("✅ Local model trained - instant recognition enabled!")
            
            # Save key mapping
            if hasattr(self, 'training_key_mapping') and self.training_key_mapping:
                if hasattr(self, 'parent_controller'):
                    self.parent_controller.add_gesture_mapping(self.training_gesture_name, self.training_key_mapping)
                    print(f"🎮 Gesture '{self.training_gesture_name}' mapped to key '{self.training_key_mapping.upper()}'")
        
        # 2. Train cloud AI as backup (if available)
        if self.custom_ai and not self.use_local_model:
            print("☁️ Training cloud AI backup...")
            # Extract frames for cloud training
            frame_list = [data['frame'] for data in self.training_frames]
            self.custom_ai.record_gesture_examples(
                self.training_gesture_name,
                self.training_description,
                frame_list
            )
        
        # 3. Generate optimized model architecture using Gemini (if available)
        if self.ai_model_generator and len(self.training_frames) >= 10:
            print("🤖 Analyzing for architecture optimization...")
            try:
                # This could be used for future model improvements
                pass
            except Exception as e:
                print(f"⚠️ Architecture generation skipped: {e}")
        
        # Reset training state
        self.training_mode = False
        self.training_frames = []
        self.training_gesture_name = ""
        self.training_key_mapping = None
        
        print("🎉 Training completed successfully!")
        print("🚀 Your gesture is ready for instant recognition!")
        return True
    
    def cancel_training(self):
        """Cancel current training session"""
        self.training_mode = False
        self.training_frames = []
        self.training_gesture_name = ""
        print("❌ Training cancelled")
    
    def toggle_custom_ai(self):
        """Toggle custom AI recognition on/off"""
        if not self.custom_ai:
            print("❌ Custom AI not available - no API key provided")
            return False
        
        self.custom_ai_enabled = not self.custom_ai_enabled
        status = "ENABLED" if self.custom_ai_enabled else "DISABLED"
        print(f"🤖 Custom AI recognition {status}")
        return self.custom_ai_enabled
    
    def list_custom_gestures(self):
        """List all trained custom gestures"""
        if not self.custom_ai:
            print("❌ Custom AI not available")
            return
        
        self.custom_ai.list_trained_gestures()
    
    def get_cached_ai_result(self, frame):
        """Get AI result with smart caching and fallback logic"""
        if not self.custom_ai_enabled or not self.custom_ai:
            return "none", 0.0
        
        # Generate a simple frame hash for caching
        import hashlib
        import time
        
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        frame_hash = hashlib.md5(frame_bytes).hexdigest()[:8]  # Short hash
        current_time = time.time()
        
        # Check cache first
        if frame_hash in self.ai_cache:
            cached_result, cached_time = self.ai_cache[frame_hash]
            if current_time - cached_time < self.cache_duration:
                return cached_result  # Return cached result
        
        # Call AI for new recognition
        ai_gesture, ai_confidence = self.custom_ai.recognize_gesture(frame)
        
        # Cache the result
        self.ai_cache[frame_hash] = ((ai_gesture, ai_confidence), current_time)
        
        # Clean old cache entries (keep cache small)
        self.ai_cache = {k: v for k, v in self.ai_cache.items() 
                        if current_time - v[1] < self.cache_duration}
        
        return ai_gesture, ai_confidence
    
    def toggle_ai_fallback_mode(self):
        """Toggle between fallback mode and always-on AI mode"""
        self.ai_fallback_only = not self.ai_fallback_only
        mode = "FALLBACK ONLY" if self.ai_fallback_only else "ALWAYS ON"
        print(f"🤖 AI Mode: {mode}")
        if self.ai_fallback_only:
            print("💡 AI will only activate when MediaPipe fails")
        else:
            print("💡 AI will always try to recognize gestures")
        return self.ai_fallback_only
    
    def process_frame(self) -> Tuple[Optional[np.ndarray], str, float, int, int]:
        """Process single frame and return frame, gesture, confidence, hand_mouse_dx, hand_mouse_dy"""
        ret, frame = self.cap.read()
        if not ret:
            return None, "none", 0.0, 0, 0
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Initialize all variables to prevent scope issues
        final_gesture = "none"
        final_confidence = 0.0
        display_gesture = "none"
        display_confidence = 0.0
        hand_mouse_dx, hand_mouse_dy = 0, 0
        features = {}
        ai_used = False
        
        # Get hand landmarks
        landmarks = self.get_hand_landmarks(frame)
        
        if landmarks:
            # Draw hand landmarks
            self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Calculate hand camera movement
            hand_mouse_dx, hand_mouse_dy = self.calculate_hand_camera_movement(landmarks, frame.shape)
            
            # Smart gesture recognition with AI fallback
            # Variables already initialized above
            
            # First try MediaPipe detection
            features = self.extract_features(landmarks)
            raw_gesture, raw_confidence = self.classify_gesture(features)
            
            # MLP model recognition - PRIORITY (check first)
            if self.use_mlp_model and self.mlp_trainer:
                mlp_gesture, mlp_confidence = self.mlp_trainer.predict_gesture(landmarks)
                
                # Check if MLP detected a custom gesture (not in default WSAD set)
                default_gestures = ["forward", "left", "backward", "right", "jump", "attack"]
                is_custom_gesture = mlp_gesture not in default_gestures and mlp_gesture != "none"
                
                if mlp_gesture != "none" and mlp_confidence > 0.65:  # Lower threshold for custom gestures
                    # For custom gestures, prioritize MLP even over high-confidence MediaPipe
                    if is_custom_gesture and mlp_confidence > 0.6:
                        final_gesture = mlp_gesture
                        final_confidence = mlp_confidence
                        display_gesture = f"MLP:{mlp_gesture}"
                        ai_used = True
                    # For default gestures, use normal priority
                    elif not is_custom_gesture:
                        final_gesture = mlp_gesture
                        final_confidence = mlp_confidence
                        display_gesture = f"MLP:{mlp_gesture}"
                        ai_used = True
                    else:
                        # Custom gesture with low confidence, try MediaPipe fallback
                        movement_gestures = ["point_left", "point_right", "point_up", "closed_fist"]
                        
                        # Only allow MediaPipe override for movement gestures with very high confidence
                        if raw_gesture in movement_gestures and raw_confidence > 0.95:  # Very high threshold
                            final_gesture = raw_gesture
                            final_confidence = raw_confidence
                            display_gesture = raw_gesture
                        else:
                            # Still prefer MLP for custom gestures
                            final_gesture = mlp_gesture
                            final_confidence = mlp_confidence
                            display_gesture = f"MLP:{mlp_gesture}"
                            ai_used = True
                else:
                    # MLP failed, try MediaPipe as fallback
                    movement_gestures = ["point_left", "point_right", "point_up", "closed_fist"]
                    
                    if raw_gesture in movement_gestures and raw_confidence > 0.8:  # Higher threshold for MediaPipe fallback
                        # MediaPipe is confident about movement
                        final_gesture = raw_gesture
                        final_confidence = raw_confidence
                        display_gesture = raw_gesture
                    else:
                        # Try action gestures with stabilization
                        stable_gesture, stable_confidence = self.stabilize_gesture(raw_gesture, raw_confidence)
                        if stable_gesture != "none" and stable_confidence > 0.8:  # Higher threshold for MediaPipe fallback
                            final_gesture = stable_gesture
                            final_confidence = stable_confidence
                            display_gesture = stable_gesture
            else:
                # No MLP, use MediaPipe only
                movement_gestures = ["point_left", "point_right", "point_up", "closed_fist"]
                
                if raw_gesture in movement_gestures and raw_confidence > 0.8:
                    # MediaPipe is confident about movement
                    final_gesture = raw_gesture
                    final_confidence = raw_confidence
                    display_gesture = raw_gesture
                else:
                    # Try action gestures with stabilization
                    stable_gesture, stable_confidence = self.stabilize_gesture(raw_gesture, raw_confidence)
                    if stable_gesture != "none" and stable_confidence > 0.8:
                        final_gesture = stable_gesture
                        final_confidence = stable_confidence
                        display_gesture = stable_gesture
            
            # Set final results
            self.current_gesture = final_gesture
            self.gesture_confidence = final_confidence
            display_confidence = final_confidence
            
            
            # Ensure display_gesture matches final_gesture if not set by AI
            if display_gesture == "none":
                display_gesture = final_gesture
            
            # Add gesture text to frame
            color = (0, 255, 0) if display_confidence > 0.7 else (0, 255, 255)
            cv2.putText(frame, f"Gesture: {display_gesture} ({display_confidence:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add training mode status
            if self.training_mode:
                cv2.putText(frame, f"TRAINING: {self.training_gesture_name} ({len(self.training_frames)} examples)", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, "Press SPACE to capture example", 
                           (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Add hand camera status
            if self.hand_camera_enabled:
                cv2.putText(frame, f"Hand Camera: ({hand_mouse_dx:.1f}, {hand_mouse_dy:.1f}) | Smoothing: {self.smoothing_factor:.1f}", 
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Add custom AI status
            if self.custom_ai:
                ai_status = "ON" if self.custom_ai_enabled else "OFF"
                cv2.putText(frame, f"Custom AI: {ai_status} | {self.custom_ai.get_status()}", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Debug info - show finger count and curl detection (only when features exist)
            if features:
                debug_text = f"4-Fingers: {features['extended_fingers']} | Curl: {features['avg_curl_distance']:.3f}"
                cv2.putText(frame, debug_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                # Show fist detection criteria when no fingers extended
                if features['extended_fingers'] == 0:
                    fist_debug = f"Wrist: {features['avg_distance_to_wrist']:.3f} | MaxCurl: {features['max_curl_distance']:.3f}"
                    cv2.putText(frame, fist_debug, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                       
            return frame, display_gesture, self.gesture_confidence, hand_mouse_dx, hand_mouse_dy
        else:
            # No hand detected
            cv2.putText(frame, "No hand detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, "none", 0.0, 0, 0
    
    def release(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()