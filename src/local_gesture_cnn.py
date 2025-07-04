"""
Local CNN Model for Hand Gesture Recognition
Optimized for hand gestures using MediaPipe landmarks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import json
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import joblib

class HandGestureDataset(Dataset):
    """Dataset for hand gesture training data"""
    def __init__(self, landmarks_data, labels):
        self.landmarks = landmarks_data
        self.labels = labels
    
    def __len__(self):
        return len(self.landmarks)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.landmarks[idx]), torch.LongTensor([self.labels[idx]])

class HandGestureCNN(nn.Module):
    """
    Lightweight CNN for hand gesture recognition
    Uses MediaPipe landmarks for fast and accurate recognition
    """
    def __init__(self, input_size=42, num_classes=10):
        super(HandGestureCNN, self).__init__()
        
        # Input: 21 landmarks × 2 coordinates = 42 features
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

class LocalGestureTrainer:
    """
    Local training system for hand gesture recognition
    Fast offline training and prediction system
    """
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.gesture_mappings = {}
        self.training_data = {"landmarks": [], "labels": [], "keys": []}
        self.model_path = "personal_gesture_model.pth"
        self.data_path = "gesture_training_data.json"
        
    def add_training_sample(self, landmarks, gesture_name, key_mapping):
        """Add a training sample with landmarks, gesture name, and key mapping"""
        # Normalize landmarks to hand-relative coordinates
        normalized_landmarks = self.normalize_landmarks(landmarks)
        
        self.training_data["landmarks"].append(normalized_landmarks)
        self.training_data["labels"].append(gesture_name)
        self.training_data["keys"].append(key_mapping)
        
        # Update gesture mappings
        self.gesture_mappings[gesture_name] = key_mapping
        
        print(f"📊 Added training sample: {gesture_name} → {key_mapping}")
    
    def normalize_landmarks(self, landmarks):
        """
        Normalize landmarks relative to wrist position
        Makes recognition translation, scale, and rotation invariant
        """
        if len(landmarks) != 21:
            raise ValueError("Expected 21 landmarks")
        
        # Convert to numpy array
        points = np.array([[lm.x, lm.y] for lm in landmarks])
        
        # Normalize relative to wrist (landmark 0)
        wrist = points[0]
        normalized = points - wrist
        
        # Scale normalize by hand size (distance from wrist to middle finger tip)
        middle_tip = normalized[12]
        hand_size = np.linalg.norm(middle_tip)
        if hand_size > 0:
            normalized = normalized / hand_size
        
        # Flatten to 42 features
        return normalized.flatten()
    
    def train_model(self):
        """
        Train the local CNN model
        Fast training with minimal data requirements
        """
        if len(self.training_data["landmarks"]) < 5:
            print("❌ Need at least 5 training samples")
            return False
        
        print("🏋️ Training local model...")
        
        # Prepare data
        X = np.array(self.training_data["landmarks"])
        y = self.label_encoder.fit_transform(self.training_data["labels"])
        
        # Create dataset
        dataset = HandGestureDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Initialize model
        num_classes = len(self.label_encoder.classes_)
        self.model = HandGestureCNN(input_size=42, num_classes=num_classes)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Train for multiple epochs
        self.model.train()
        for epoch in range(100):
            total_loss = 0
            for batch_landmarks, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_landmarks)
                loss = criterion(outputs, batch_labels.squeeze())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Save model and data
        self.save_model()
        print("✅ Model trained successfully!")
        print(f"🎯 Trained on {len(X)} samples, {num_classes} gestures")
        return True
    
    def predict_gesture(self, landmarks):
        """
        Predict gesture from landmarks
        Fast offline inference with high accuracy
        """
        if self.model is None:
            return "none", 0.0
        
        try:
            # Normalize landmarks
            normalized = self.normalize_landmarks(landmarks)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(normalized).unsqueeze(0)
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Convert back to gesture name
                gesture_name = self.label_encoder.inverse_transform([predicted.item()])[0]
                confidence_score = confidence.item()
                
                return gesture_name, confidence_score
        
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "none", 0.0
    
    def get_key_mapping(self, gesture_name):
        """Get the key mapping for a gesture"""
        return self.gesture_mappings.get(gesture_name, None)
    
    def save_model(self):
        """Save the trained model and data"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'label_encoder': self.label_encoder,
                'gesture_mappings': self.gesture_mappings,
                'num_classes': len(self.label_encoder.classes_)
            }, self.model_path)
        
        # Save training data
        with open(self.data_path, 'w') as f:
            json.dump({
                'training_data': self.training_data,
                'gesture_mappings': self.gesture_mappings
            }, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    def load_model(self):
        """Load the trained model and data"""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path)
                
                # Reconstruct model
                num_classes = checkpoint['num_classes']
                self.model = HandGestureCNN(input_size=42, num_classes=num_classes)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                # Load encoders and mappings
                self.label_encoder = checkpoint['label_encoder']
                self.gesture_mappings = checkpoint['gesture_mappings']
                
                print(f"✅ Loaded model with {num_classes} gestures")
                return True
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        
        # Load training data if available
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    self.training_data = data['training_data']
                    self.gesture_mappings = data['gesture_mappings']
                print("📂 Loaded training data")
            except Exception as e:
                print(f"❌ Error loading training data: {e}")
        
        return False
    
    def get_status(self):
        """Get current model status"""
        if self.model is None:
            return "No model trained"
        
        num_gestures = len(self.gesture_mappings)
        num_samples = len(self.training_data["landmarks"])
        return f"Local Model: {num_gestures} gestures, {num_samples} samples"
    
    def list_gestures(self):
        """List all trained gestures and their mappings"""
        if not self.gesture_mappings:
            print("📝 No gestures trained yet")
            return
        
        print("\n🎯 Trained Gestures:")
        print("=" * 40)
        for gesture, key in self.gesture_mappings.items():
            sample_count = self.training_data["labels"].count(gesture)
            print(f"   {gesture} → {key.upper()} ({sample_count} samples)")
        print("=" * 40)