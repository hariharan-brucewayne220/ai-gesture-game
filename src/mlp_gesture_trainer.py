"""
MLP Gesture Trainer - Landmark-based gesture recognition using MLP neural network
Calibrates with 100 pictures per gesture, 10 per 10-second session with angle variations
"""

import numpy as np
import pickle
import os
import time
from typing import Dict, List, Tuple, Optional
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import cv2

class MLPGestureTrainer:
    def __init__(self):
        self.gesture_classes = {}  # gesture_name -> class_index
        self.training_data = []    # List of (landmarks, class_index)
        self.model = None
        self.scaler = StandardScaler()
        self.is_calibrated = False
        
        # Model paths
        self.model_dir = "mlp_models"
        self.model_path = os.path.join(self.model_dir, "gesture_model.pkl").replace("\\", "/")
        self.scaler_path = os.path.join(self.model_dir, "scaler.pkl").replace("\\", "/")
        self.classes_path = os.path.join(self.model_dir, "classes.pkl").replace("\\", "/")
        self.training_data_path = os.path.join(self.model_dir, "training_data.pkl").replace("\\", "/")
        self.model_dir = self.model_dir.replace("\\", "/")
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Default WSAD gestures
        self.default_gestures = {
            "forward": "w",
            "left": "a", 
            "backward": "s",
            "right": "d",
            "jump": "space",
            "attack": "click"
        }
        
        # Training session settings
        self.pics_per_gesture = 50  # Reduced for better balance
        self.pics_per_session = 10
        self.session_duration = 10  # seconds
        
    def extract_landmark_features(self, landmarks) -> np.ndarray:
        """Extract features from MediaPipe landmarks"""
        if not landmarks:
            return np.array([])
            
        # Convert landmarks to numpy array (21 points, x,y coordinates)
        points = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
        
        # Normalize relative to wrist (landmark 0)
        wrist = points[0]
        normalized_points = points - wrist
        
        # Flatten to 1D feature vector (42 features: 21 points * 2 coordinates)
        features = normalized_points.flatten()
        
        # Add additional computed features
        additional_features = []
        
        # Finger tip positions (landmarks 4, 8, 12, 16, 20)
        finger_tips = [4, 8, 12, 16, 20]
        for tip in finger_tips:
            additional_features.extend([points[tip][0], points[tip][1]])
        
        # Distances between key points
        thumb_index_dist = np.linalg.norm(points[4] - points[8])
        index_middle_dist = np.linalg.norm(points[8] - points[12])
        
        # Thumb orientation features (important for thumbs up/down)
        thumb_tip = points[4]
        thumb_base = points[2]  # Thumb MCP joint
        wrist = points[0]
        
        # Thumb direction relative to wrist
        thumb_vector = thumb_tip - wrist
        thumb_vertical = thumb_vector[1]  # Y component (up/down)
        thumb_horizontal = thumb_vector[0]  # X component (left/right)
        
        # Thumb relative to hand center
        hand_center = np.mean(points[5:], axis=0)  # Average of finger bases
        thumb_relative_y = thumb_tip[1] - hand_center[1]
        
        additional_features.extend([
            thumb_index_dist, 
            index_middle_dist,
            thumb_vertical,      # Key for thumbs up/down detection
            thumb_horizontal,
            thumb_relative_y     # Thumb position relative to hand
        ])
        
        # Combine all features
        all_features = np.concatenate([features, additional_features])
        
        return all_features
    
    def start_calibration(self):
        """Start the full calibration process for all default gestures"""
        print("🎯 Starting MLP Gesture Calibration")
        print("=" * 50)
        print(f"We'll capture {self.pics_per_gesture} pictures for each gesture")
        print(f"Each gesture will have {self.pics_per_gesture // self.pics_per_session} sessions of {self.session_duration} seconds")
        print("During each session, slightly change your hand angle for variety")
        print()
        
        # Clear existing training data
        self.training_data = []
        self.gesture_classes = {}
        
        # Calibrate each default gesture
        for gesture_name, key_mapping in self.default_gestures.items():
            print(f"\n🎯 Calibrating gesture: {gesture_name.upper()} → {key_mapping}")
            self.calibrate_gesture(gesture_name, key_mapping)
        
        # Train the model
        print("\nTraining MLP model...")
        success = self.train_model()
        
        if success:
            print("\nCalibration complete! MLP model ready for use.")
            self.is_calibrated = True
            return True
        else:
            print("\nCalibration failed during training.")
            return False
    
    def calibrate_gesture(self, gesture_name: str, key_mapping: str):
        """Calibrate a specific gesture with multiple sessions"""
        # Add to gesture classes
        class_index = len(self.gesture_classes)
        self.gesture_classes[gesture_name] = class_index
        
        total_sessions = self.pics_per_gesture // self.pics_per_session
        
        print(f"\nCalibrating '{gesture_name}' gesture")
        print(f"Will capture {total_sessions} sessions of {self.pics_per_session} pictures each")
        
        for session in range(total_sessions):
            print(f"\nSession {session + 1}/{total_sessions}")
            print(f"Show '{gesture_name}' gesture and hold for {self.session_duration} seconds")
            print("Slightly change angle during the session for variety")
            
            input("Press ENTER when ready...")
            
            # Capture session
            self.capture_session(gesture_name, class_index, session + 1)
        
        print(f"Completed calibration for '{gesture_name}' - {self.pics_per_gesture} pictures captured")
    
    def capture_session(self, gesture_name: str, class_index: int, session_num: int):
        """Capture one session of pictures for a gesture"""
        import cv2
        import mediapipe as mp
        
        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
        
        # Session timing
        start_time = time.time()
        end_time = start_time + self.session_duration
        
        # Capture timing
        capture_interval = self.session_duration / self.pics_per_session
        next_capture_time = start_time + capture_interval
        
        captured_count = 0
        last_frame_time = time.time()
        
        print(f"Recording session {session_num} for '{gesture_name}'...")
        print("Hold your gesture steadily, changing angle slightly")
        
        try:
            while time.time() < end_time and captured_count < self.pics_per_session:
                current_time = time.time()
                
                # Control frame rate to prevent freezing
                if current_time - last_frame_time < 0.033:  # ~30 FPS max
                    time.sleep(0.01)
                    continue
                last_frame_time = current_time
                
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                
                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                # Check if it's time to capture
                if current_time >= next_capture_time:
                    if results.multi_hand_landmarks:
                        landmarks = results.multi_hand_landmarks[0]
                        
                        # Extract features
                        features = self.extract_landmark_features(landmarks)
                        if len(features) > 0:
                            # Add to training data
                            self.training_data.append((features, class_index))
                            captured_count += 1
                            
                            print(f"Captured {captured_count}/{self.pics_per_session}")
                            
                            # Set next capture time
                            next_capture_time = current_time + capture_interval
                    else:
                        print("No hand detected - keep hand visible")
                
                # Add overlay
                remaining_time = end_time - current_time
                cv2.putText(frame, f"Session {session_num}: {gesture_name}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {remaining_time:.1f}s", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Captured: {captured_count}/{self.pics_per_session}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Draw landmarks if detected
                if results.multi_hand_landmarks:
                    mp_draw = mp.solutions.drawing_utils
                    mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], 
                                         mp_hands.HAND_CONNECTIONS)
                
                try:
                    cv2.imshow(f"Calibrating {gesture_name}", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except Exception as e:
                    print(f"Display error: {e}")
                    break
        
        finally:
            try:
                cap.release()
            except:
                pass
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Force window cleanup
            except:
                pass
        
        print(f"Session {session_num} complete - captured {captured_count} samples")
    
    def train_model(self):
        """Train the MLP model on collected data"""
        if not self.training_data:
            print("No training data available")
            return False
        
        print(f"Training MLP on {len(self.training_data)} samples...")
        
        # Prepare data
        X = np.array([sample[0] for sample in self.training_data])
        y = np.array([sample[1] for sample in self.training_data])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data with error handling
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError as e:
            print(f"Error in train/test split: {e}")
            # Fallback without stratification if classes too small
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        
        # Check class distribution
        from collections import Counter
        class_counts = Counter(y)
        print(f"Training samples per gesture: {dict(class_counts)}")
        
        # Ensure minimum samples per class for train/test split
        min_samples = min(class_counts.values())
        if min_samples < 4:
            print(f"Warning: Only {min_samples} samples for some gestures - need at least 4")
            return False
        
        # Create and train MLP with correct number of output classes
        num_classes = len(self.gesture_classes)
        print(f"Creating MLP for {num_classes} gesture classes")
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Train with error handling
        print("Training neural network...")
        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            print(f"Training failed: {e}")
            return False
        
        # Evaluate
        train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.model.predict(X_test))
        
        print("Training complete!")
        print(f"Training accuracy: {train_accuracy:.3f}")
        print(f"Test accuracy: {test_accuracy:.3f}")
        
        # Save model
        self.save_model()
        
        return True
    
    def predict_gesture(self, landmarks) -> Tuple[str, float]:
        """Predict gesture from landmarks"""
        if not self.model or not landmarks:
            return "none", 0.0
        
        try:
            # Extract features
            features = self.extract_landmark_features(landmarks)
            if len(features) == 0:
                return "none", 0.0
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            # Get gesture name
            gesture_name = None
            for name, class_idx in self.gesture_classes.items():
                if class_idx == prediction:
                    gesture_name = name
                    break
            
            return gesture_name or "none", confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "none", 0.0
    
    def add_custom_gesture(self, gesture_name: str, key_mapping: str):
        """Add a custom gesture after initial calibration"""
        print(f"\nAdding custom gesture: {gesture_name} → {key_mapping}")
        
        # First, ensure existing model is loaded
        if not self.model or not self.training_data:
            print("Loading existing model and training data...")
            if not self.load_model():
                print("❌ No existing model found. Please run full calibration first.")
                return False
            
            # If training data is empty but model exists, we need to regenerate it
            if not self.training_data:
                print("❌ No training data available. Please run full calibration first.")
                return False
        
        # Reassign ALL gesture classes to ensure continuous indexing 0,1,2,3...
        old_gesture_classes = self.gesture_classes.copy()
        all_gestures = list(self.gesture_classes.keys()) + [gesture_name]
        self.gesture_classes = {}
        for i, gesture in enumerate(all_gestures):
            self.gesture_classes[gesture] = i
        
        print(f"Reassigned gesture indices: {self.gesture_classes}")
        
        # Update existing training data labels to match new indices
        if self.training_data:
            print("Updating training data labels to match new indices...")
            # Create mapping from old index to new index
            index_mapping = {}
            for gesture, old_index in old_gesture_classes.items():
                new_index = self.gesture_classes[gesture]
                index_mapping[old_index] = new_index
            
            # Update all training data labels
            updated_training_data = []
            for features, old_label in self.training_data:
                new_label = index_mapping.get(old_label, old_label)
                updated_training_data.append((features, new_label))
            self.training_data = updated_training_data
            print(f"Updated {len(self.training_data)} training samples with new labels")
        
        # Add to default gestures for key mapping
        self.default_gestures[gesture_name] = key_mapping
        
        # Store current training data count
        original_samples = len(self.training_data)
        print(f"Existing training data: {original_samples} samples")
        
        # Calibrate the new gesture (this adds to self.training_data)
        self.calibrate_gesture(gesture_name, key_mapping)
        
        # Verify new data was added
        new_samples = len(self.training_data) - original_samples
        print(f"Added {new_samples} new samples for '{gesture_name}'")
        print(f"Total training data: {len(self.training_data)} samples")
        
        # Create NEW model for the new number of classes
        print("Creating new model architecture for updated gesture set...")
        self.model = None  # Force creation of new model with correct output size
        success = self.train_model()
        
        if success:
            print(f"Custom gesture '{gesture_name}' added successfully!")
            return True
        else:
            print(f"Failed to retrain model with '{gesture_name}'")
            return False
    
    def save_model(self):
        """Save the trained model and scaler"""
        try:
            # Ensure directory exists
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save training data for incremental learning
            with open(self.training_data_path, 'wb') as f:
                pickle.dump(self.training_data, f)
            
            # Save gesture classes and mappings
            with open(self.classes_path, 'wb') as f:
                pickle.dump({
                    'gesture_classes': self.gesture_classes,
                    'default_gestures': self.default_gestures
                }, f)
            
            print("Model saved successfully")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            print(f"Trying to save to: {self.model_path}")
            return False
    
    def manual_save(self):
        """Manual save command for current model"""
        if self.model is None:
            print("No model to save")
            return False
        return self.save_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            # Check if all files exist
            if not all(os.path.exists(path) for path in [self.model_path, self.scaler_path, self.classes_path]):
                return False
            
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load training data if available
            if os.path.exists(self.training_data_path):
                with open(self.training_data_path, 'rb') as f:
                    self.training_data = pickle.load(f)
                print(f"Loaded {len(self.training_data)} training samples")
            else:
                self.training_data = []
                print("No training data found - incremental learning disabled")
            
            # Load gesture classes and mappings
            with open(self.classes_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'gesture_classes' in data:
                    self.gesture_classes = data['gesture_classes']
                    self.default_gestures.update(data.get('default_gestures', {}))
                else:
                    # Old format compatibility
                    self.gesture_classes = data
            
            self.is_calibrated = True
            print(f"Model loaded with {len(self.gesture_classes)} gestures")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_gesture_key_mapping(self, gesture_name: str) -> Optional[str]:
        """Get key mapping for a gesture"""
        return self.default_gestures.get(gesture_name)
    
    def clear_all_data(self):
        """Clear all training data and models for fresh start"""
        import shutil
        
        # Clear training data
        self.training_data = []
        self.gesture_classes = {}
        self.model = None
        self.is_calibrated = False
        
        # Reset to default gestures only
        self.default_gestures = {
            "forward": "w",
            "left": "a", 
            "backward": "s",
            "right": "d",
            "jump": "space",
            "attack": "click"
        }
        
        # Remove saved model files
        try:
            if os.path.exists(self.model_dir):
                shutil.rmtree(self.model_dir)
                os.makedirs(self.model_dir, exist_ok=True)
            print("All training data and models cleared!")
        except Exception as e:
            print(f"Error clearing data: {e}")
    
    def list_gestures(self):
        """List all calibrated gestures"""
        if not self.gesture_classes:
            print("No gestures calibrated yet")
            return
        
        print("\nCalibrated Gestures:")
        print("=" * 40)
        for gesture_name, class_idx in self.gesture_classes.items():
            key_mapping = self.get_gesture_key_mapping(gesture_name) or "custom"
            print(f"   {gesture_name} → {key_mapping}")
        print("=" * 40)
        print(f"Total: {len(self.gesture_classes)} gestures")
    
    def get_training_progress(self) -> Dict:
        """Get training progress information"""
        total_expected = len(self.default_gestures) * self.pics_per_gesture
        current_samples = len(self.training_data)
        
        return {
            "total_expected": total_expected,
            "current_samples": current_samples,
            "progress_percent": (current_samples / total_expected) * 100 if total_expected > 0 else 0,
            "gestures_calibrated": len(self.gesture_classes),
            "is_calibrated": self.is_calibrated
        }

# Example usage
if __name__ == "__main__":
    trainer = MLPGestureTrainer()
    
    # Start calibration
    trainer.start_calibration()
    
    print("🎉 MLP Gesture Trainer ready!")