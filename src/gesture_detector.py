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
        # Initialize MediaPipe for dual hand detection - ULTRA-OPTIMIZED for left hand speed
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Both hands for gaming: left=camera, right=gestures
            min_detection_confidence=0.7,  # Lowered for faster detection
            min_tracking_confidence=0.3    # Much lower for ultra-responsive tracking
        )
        
        # Hand role assignment - dual hand gaming mode
        self.left_hand_camera = True   # Left hand controls camera
        self.right_hand_gestures = True  # Right hand for discrete actions
        self.dual_hand_mode = True     # Enable two-hand gaming system
        self.mp_draw = mp.solutions.drawing_utils
        
        # Video capture - now using threaded approach for better performance
        self.use_threaded_camera = True
        self.threaded_camera = None
        
        # Fallback to direct capture if needed
        self.cap = None
        
        # Gesture state tracking
        self.current_gesture = None
        self.gesture_confidence = 0.0
        self.gesture_history = []
        
        # Hand camera control - Velocity-based movement
        self.hand_camera_enabled = True  # ON by default
        self.hand_reference_position = None
        self.last_hand_position = None
        self.camera_sensitivity_x = 1.5  # Reduced for God of War precision
        self.camera_sensitivity_y = 1.2  # Reduced for God of War precision
        self.camera_deadzone = 0.01     # Movement threshold
        
        # Aggressive smoothing for ultra-responsive camera movement
        self.smoothed_velocity_x = 0.0
        self.smoothed_velocity_y = 0.0
        self.smoothing_factor = 0.85  # Increased for immediate response
        
        # Left hand performance optimization
        self.left_hand_frame_skip = 0  # Skip frames for better performance
        self.left_hand_skip_rate = 1   # Process every frame (no skipping by default)
        self.cached_left_hand_position = None
        self.position_interpolation_factor = 0.9  # Very aggressive interpolation
        
        # Performance monitoring
        self.left_hand_response_times = []
        self.last_left_hand_time = 0
        self.performance_debug = False
        
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
        
        # Gesture-based camera control
        self.gesture_camera_control = True   # Require "cam" gesture for camera movement
        self.camera_control_gesture = "cam"  # Name of gesture that enables camera movement  
        self.camera_control_hand = "left"    # Which hand controls camera
        self.current_camera_gesture_active = False  # Is camera gesture currently detected
        
        # Initialize MLP trainer
        try:
            from mlp_gesture_trainer import MLPGestureTrainer
            self.mlp_trainer = MLPGestureTrainer()
            if self.mlp_trainer.load_model():
                self.use_mlp_model = True
                print("MLP gesture system initialized")
            else:
                self.use_mlp_model = False
                print("MLP gesture system ready for calibration")
        except Exception as e:
            print(f"Warning: MLP gesture system not available: {e}")
            self.mlp_trainer = None
        
        # Initialize threaded camera for better performance
        if self.use_threaded_camera:
            try:
                from threaded_camera import ThreadedCameraProcessor
                self.threaded_camera = ThreadedCameraProcessor(
                    camera_index=0,
                    target_fps=30,  # Will auto-reduce during heavy gaming
                    buffer_size=2
                )
                print("🎥 Threaded camera processor initialized")
            except Exception as e:
                print(f"Warning: Threaded camera not available: {e}")
                self.use_threaded_camera = False
        
        
    def get_hand_landmarks(self, frame):
        """Extract hand landmarks - returns both hands for dual-hand gaming"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # For backwards compatibility, return single hand if only one detected
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 1:
                # Single hand detected - use as right hand (gesture hand)
                return results.multi_hand_landmarks[0]
            else:
                # Multiple hands detected - return the right hand for gestures
                # MediaPipe provides handedness information
                if results.multi_handedness:
                    for i, handedness in enumerate(results.multi_handedness):
                        # MediaPipe gives "Left"/"Right" from camera perspective
                        # "Right" in camera view = user's right hand
                        if handedness.classification[0].label == "Right":
                            return results.multi_hand_landmarks[i]
                # Fallback to first hand if no right hand found
                return results.multi_hand_landmarks[0]
        return None
        
    def get_both_hands(self, frame):
        """Get both hands separately for dual-hand gaming"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        left_hand = None
        right_hand = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            # Debug: Show detected hands info
            detected_hands = []
            for i, handedness in enumerate(results.multi_handedness):
                hand_label = handedness.classification[0].label
                confidence = handedness.classification[0].score
                detected_hands.append(f"{hand_label}({confidence:.2f})")
                
                if hand_label == "Left":  # Camera perspective: user's left hand
                    left_hand = results.multi_hand_landmarks[i]
                elif hand_label == "Right":  # Camera perspective: user's right hand
                    right_hand = results.multi_hand_landmarks[i]
            
            # Store debug info for display
            self.detected_hands_debug = ", ".join(detected_hands)
        else:
            self.detected_hands_debug = "No hands detected"
        
        return left_hand, right_hand, results
    
    
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
    
    def classify_gesture_for_hand(self, landmarks, hand_type: str) -> Tuple[str, float]:
        """Classify gesture for specific hand with trained models"""
        if not landmarks:
            return "none", 0.0
        
        # First try MLP trainer for hand-specific gestures
        if self.mlp_trainer and self.use_mlp_model:
            try:
                # Check if MLP trainer supports hand-specific classification
                if hasattr(self.mlp_trainer, 'classify_gesture_for_hand'):
                    return self.mlp_trainer.classify_gesture_for_hand(landmarks, hand_type)
                else:
                    # Fallback to regular MLP classification using landmarks
                    return self.mlp_trainer.predict_gesture(landmarks)
            except Exception as e:
                print(f"MLP classification error: {e}")
        
        # Fallback to MediaPipe classification (extract features first)
        features = self.extract_features(landmarks)
        return self.classify_gesture(features)
    
    def calculate_hand_camera_movement(self, landmarks, image_size):
        """Optimized camera movement with aggressive responsiveness for left hand"""
        if not landmarks or not self.hand_camera_enabled:
            return 0, 0
            
        # Use wrist position (landmark 0) as tracking point
        wrist = landmarks.landmark[0]
        current_position = (wrist.x, wrist.y)
        
        # Initialize positions if not set
        if self.last_hand_position is None:
            self.last_hand_position = current_position
            self.cached_left_hand_position = current_position
            return 0, 0
        
        # PERFORMANCE OPTIMIZATION: Use cached position when available
        if self.cached_left_hand_position is not None:
            # Interpolate between cached and current for ultra-smooth movement
            interpolated_x = (1 - self.position_interpolation_factor) * self.cached_left_hand_position[0] + self.position_interpolation_factor * current_position[0]
            interpolated_y = (1 - self.position_interpolation_factor) * self.cached_left_hand_position[1] + self.position_interpolation_factor * current_position[1]
            current_position = (interpolated_x, interpolated_y)
        
        # Calculate raw velocity with high precision
        raw_velocity_x = current_position[0] - self.last_hand_position[0]
        raw_velocity_y = current_position[1] - self.last_hand_position[1]
        
        # Ultra-aggressive smoothing for immediate response
        self.smoothed_velocity_x = (1 - self.smoothing_factor) * self.smoothed_velocity_x + self.smoothing_factor * raw_velocity_x
        self.smoothed_velocity_y = (1 - self.smoothing_factor) * self.smoothed_velocity_y + self.smoothing_factor * raw_velocity_y
        
        # Update positions for next frame
        self.last_hand_position = current_position
        self.cached_left_hand_position = current_position
        
        # Ultra-responsive movement calculation
        mouse_dx = 0.0
        mouse_dy = 0.0
        
        # Lower deadzone for more sensitive movement detection
        ultra_deadzone = self.camera_deadzone * 0.5  # Half the deadzone for faster response
        
        if abs(self.smoothed_velocity_x) > ultra_deadzone:
            # Increased sensitivity multiplier for faster response
            mouse_dx = self.smoothed_velocity_x * self.camera_sensitivity_x * 1000  # Increased from 800
            
        if abs(self.smoothed_velocity_y) > ultra_deadzone:
            mouse_dy = self.smoothed_velocity_y * self.camera_sensitivity_y * 1000  # Increased from 800
        
        return mouse_dx, mouse_dy
    
    def set_hand_camera_reference(self):
        """Reset hand camera tracking"""
        self.last_hand_position = None
        self.smoothed_velocity_x = 0.0
        self.smoothed_velocity_y = 0.0
        self.cached_left_hand_position = None
        print("📍 Hand camera tracking reset")
    
    def get_left_hand_performance_stats(self):
        """Get performance statistics for left hand tracking"""
        if not self.left_hand_response_times:
            return "No data yet"
        
        avg_response = sum(self.left_hand_response_times) / len(self.left_hand_response_times)
        min_response = min(self.left_hand_response_times)
        max_response = max(self.left_hand_response_times)
        
        return f"Left Hand FPS: {1/avg_response:.1f} | Avg: {avg_response*1000:.1f}ms | Min: {min_response*1000:.1f}ms | Max: {max_response*1000:.1f}ms"
    
    def toggle_performance_debug(self):
        """Toggle performance debugging display"""
        self.performance_debug = not self.performance_debug
        status = "ON" if self.performance_debug else "OFF"
        print(f"📊 Left hand performance debug: {status}")
        return self.performance_debug
    
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
    
    def add_custom_gesture(self, gesture_name: str, key_mapping: str, hand_type: str = "right"):
        """Add a custom gesture after initial calibration with hand type support"""
        if not self.mlp_trainer:
            print("❌ MLP trainer not available")
            return False
        
        # Temporarily stop threaded camera to avoid conflicts
        camera_was_threaded = False
        if self.use_threaded_camera and self.threaded_camera and self.threaded_camera.running:
            print("⏸️ Temporarily stopping threaded camera for gesture capture")
            self.threaded_camera.stop()
            camera_was_threaded = True
        
        # Provide camera access to MLP trainer
        if self.cap and self.cap.isOpened():
            self.mlp_trainer.external_camera = self.cap
        
        # Add hand type information to the gesture
        success = self.mlp_trainer.add_custom_gesture(gesture_name, key_mapping, hand_type=hand_type)
        
        # If this is a camera control gesture, initialize gesture-based camera control
        if success and key_mapping == "camera_control":
            self.gesture_camera_control = True
            self.camera_control_gesture = gesture_name
            self.camera_control_hand = hand_type
            print(f"✅ Gesture-based camera control enabled for {hand_type} hand gesture: {gesture_name}")
            print("💡 Camera will only move when this gesture is detected")
        
        # Remove camera reference
        if hasattr(self.mlp_trainer, 'external_camera'):
            self.mlp_trainer.external_camera = None
        
        # Restart threaded camera if it was running
        if camera_was_threaded:
            print("▶️ Restarting threaded camera")
            self.threaded_camera.start()
        
        if success:
            self.use_mlp_model = True
            print(f"✅ Custom gesture '{gesture_name}' added!")
        return success
    
    def toggle_gesture_camera_control(self):
        """Toggle gesture-based camera control on/off"""
        if not self.camera_control_gesture:
            print("❌ No camera control gesture trained yet")
            print("💡 Use 'j' to add a left hand camera control gesture")
            return False
        
        self.gesture_camera_control = not self.gesture_camera_control
        
        if self.gesture_camera_control:
            print(f"✅ Gesture camera control ENABLED")
            print(f"🎥 Camera will only move when '{self.camera_control_gesture}' gesture is detected")
            print(f"✋ Use your {self.camera_control_hand} hand")
        else:
            print(f"❌ Gesture camera control DISABLED")
            print("🔄 Reverted to always-on left hand camera movement")
            print("💡 Left hand will now always control camera movement")
        
        return self.gesture_camera_control

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
    
    def start_camera(self):
        """Start the camera system"""
        if self.use_threaded_camera and self.threaded_camera:
            success = self.threaded_camera.start()
            if success:
                print("✅ High-performance threaded camera started")
                return True
            else:
                print("⚠️ Threaded camera failed, falling back to direct capture")
                self.use_threaded_camera = False
        
        # Fallback: initialize direct capture
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return self.cap.isOpened() if self.cap else False
    
    def get_camera_stats(self):
        """Get camera performance statistics"""
        if self.use_threaded_camera and self.threaded_camera:
            return self.threaded_camera.get_stats()
        else:
            return {"mode": "direct_capture", "fps": "unknown"}
    
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
        """Process frame with dual-hand gaming: left=camera, right=gestures"""
        
        # Use threaded camera if available
        if self.use_threaded_camera and self.threaded_camera:
            ret, frame = self.threaded_camera.get_frame()
            if not ret:
                return None, "none", 0.0, 0, 0
        else:
            # Fallback to direct capture
            if not self.cap:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, frame = self.cap.read()
            if not ret:
                return None, "none", 0.0, 0, 0
            
            # Flip frame for mirror effect (threaded camera already does this)
            frame = cv2.flip(frame, 1)
        
        # Frame counting for debugging purposes
        if hasattr(self, '_debug_frame_count'):
            self._debug_frame_count += 1
        else:
            self._debug_frame_count = 1
        
        # Initialize all variables
        final_gesture = "none"
        final_confidence = 0.0
        display_gesture = "none"
        display_confidence = 0.0
        hand_mouse_dx, hand_mouse_dy = 0, 0
        features = {}
        ai_used = False
        
        # TWO-HAND GAMING SYSTEM
        if self.dual_hand_mode and self.left_hand_camera and self.right_hand_gestures:
            # Get both hands separately
            left_hand, right_hand, results = self.get_both_hands(frame)
            
            # Debug: Show dual hand mode status and hand detection
            hand_info = f"DUAL MODE: L={left_hand is not None}, R={right_hand is not None}"
            cv2.putText(frame, hand_info, (10, 350), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show detected hands info
            if hasattr(self, 'detected_hands_debug'):
                cv2.putText(frame, f"Detected: {self.detected_hands_debug}", (10, 370), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Draw both hands if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # LEFT HAND: ULTRA-OPTIMIZED camera movement for immediate response
            if left_hand and self.hand_camera_enabled:
                # Debug: Show that left hand is detected
                cv2.putText(frame, "LEFT HAND: DETECTED", (10, 320), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                # Performance tracking
                import time
                current_time = time.time()
                if self.last_left_hand_time > 0:
                    response_time = current_time - self.last_left_hand_time
                    self.left_hand_response_times.append(response_time)
                    # Keep only last 30 measurements
                    if len(self.left_hand_response_times) > 30:
                        self.left_hand_response_times.pop(0)
                self.last_left_hand_time = current_time
                
                # AGGRESSIVE OPTIMIZATION: Minimize all processing overhead
                self.left_hand_frame_skip += 1
                
                if self.gesture_camera_control and self.camera_control_gesture:
                    # Reduce gesture checking to every 5 frames for maximum performance
                    if not hasattr(self, 'gesture_check_counter'):
                        self.gesture_check_counter = 0
                    
                    self.gesture_check_counter += 1
                    
                    # Only check gesture every 5 frames to eliminate lag
                    if self.gesture_check_counter % 5 == 0:
                        left_gesture, left_confidence = self.classify_gesture_for_hand(left_hand, "left")
                        self.last_camera_gesture = left_gesture
                        self.last_camera_confidence = left_confidence
                    
                    # INVERTED LOGIC: Camera moves by default, stops when cam gesture detected
                    if hasattr(self, 'last_camera_gesture') and self.last_camera_gesture == self.camera_control_gesture and getattr(self, 'last_camera_confidence', 0) > 0.6:
                        # Camera control gesture detected - STOP movement
                        self.current_camera_gesture_active = False
                        hand_mouse_dx, hand_mouse_dy = 0, 0
                        
                        # Visual feedback showing camera is paused
                        cv2.putText(frame, f"CAM: PAUSED ({self.camera_control_gesture})", (10, 300), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else:
                        # No camera gesture - enable normal movement (default behavior)
                        self.current_camera_gesture_active = True
                        hand_mouse_dx, hand_mouse_dy = self.calculate_hand_camera_movement(left_hand, frame.shape)
                else:
                    # Always-on camera mode - MAXIMUM PERFORMANCE path
                    hand_mouse_dx, hand_mouse_dy = self.calculate_hand_camera_movement(left_hand, frame.shape)
                    
                    # Show left hand camera status
                    cv2.putText(frame, "LEFT HAND CAMERA: ACTIVE", (10, 300), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Show when left hand is not detected
                if self.hand_camera_enabled:
                    if not hasattr(self, 'detected_hands_debug') or "Left" not in self.detected_hands_debug:
                        cv2.putText(frame, "LEFT HAND: NOT DETECTED", (10, 320), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.putText(frame, "LEFT HAND CAMERA: WAITING", (10, 300), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # RIGHT HAND: Gesture recognition only (existing trained model)
            if right_hand:
                landmarks = right_hand  # Use right hand for all gesture recognition
            else:
                landmarks = None
                
        else:
            # FALLBACK: Single hand mode (original system)
            # Debug: Show why we're in single hand mode
            cv2.putText(frame, f"SINGLE HAND MODE: dual={self.dual_hand_mode}, left={self.left_hand_camera}, right={self.right_hand_gestures}", 
                       (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            landmarks = self.get_hand_landmarks(frame)
            
            if landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Calculate hand camera movement (if enabled)
                if self.hand_camera_enabled:
                    hand_mouse_dx, hand_mouse_dy = self.calculate_hand_camera_movement(landmarks, frame.shape)
        
        # GESTURE RECOGNITION (for right hand in dual mode, or single hand in single mode)
        if landmarks:
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
            
            # Add hand camera status with performance metrics
            if self.hand_camera_enabled:
                camera_text = f"Hand Camera: ({hand_mouse_dx:.1f}, {hand_mouse_dy:.1f}) | Smoothing: {self.smoothing_factor:.1f}"
                if self.performance_debug and self.left_hand_response_times:
                    camera_text += f" | {self.get_left_hand_performance_stats()}"
                cv2.putText(frame, camera_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
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
    
    def toggle_dual_hand_mode(self):
        """Toggle between single hand and dual hand gaming mode"""
        self.dual_hand_mode = not self.dual_hand_mode
        mode = "DUAL HAND" if self.dual_hand_mode else "SINGLE HAND"
        print(f"🙌 Hand Mode: {mode}")
        
        if self.dual_hand_mode:
            print("💡 Left hand = Camera, Right hand = Gestures")
            print("💡 Your existing trained gestures work on RIGHT hand")
        else:
            print("💡 Single hand for both camera and gestures")
            print("💡 Original system restored")
        
        return self.dual_hand_mode
        
    def get_hand_mode_status(self):
        """Get current hand mode status for display"""
        if self.dual_hand_mode:
            return "Dual Hand (L=Cam, R=Gest)"
        else:
            return "Single Hand"
            
    def release(self):
        """Clean up resources"""
        # Stop threaded camera
        if self.use_threaded_camera and self.threaded_camera:
            self.threaded_camera.stop()
        
        # Release direct capture if used
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("🛑 Camera resources released")