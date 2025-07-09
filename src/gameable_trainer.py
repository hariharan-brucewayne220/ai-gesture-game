"""
GameAble Trainer - Simplified integration for gesture gaming
Based on the original GameAble codebase
"""

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from PIL import Image
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import requests
import base64
import re
import json
import importlib.util

# Create directories
os.makedirs('training_data', exist_ok=True)
os.makedirs('models', exist_ok=True)

class PoseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Get all pose directories
        pose_dirs = [d for d in os.listdir(data_dir) if d.startswith('pose_')]
        pose_dirs.sort(key=lambda x: int(x.split('_')[1]))  # Sort by pose number
        
        for pose_idx, pose_dir in enumerate(pose_dirs):
            pose_path = os.path.join(data_dir, pose_dir)
            if os.path.isdir(pose_path):
                image_files = [f for f in os.listdir(pose_path) if f.endswith('.jpg')]
                if not image_files:
                    print(f"Warning: No images found in {pose_path}")
                    continue
                    
                for img_name in image_files:
                    self.images.append(os.path.join(pose_path, img_name))
                    self.labels.append(pose_idx)
        
        if not self.images:
            raise ValueError("No images found in training data")
            
        # Convert labels to tensor for max operation
        self.labels = torch.tensor(self.labels)
        print(f"📊 Loaded {len(self.images)} images")
        print(f"🎯 Pose distribution: {torch.bincount(self.labels).tolist()}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class GameAbleTrainer:
    def __init__(self, google_ai_key=None):
        self.google_ai_key = google_ai_key
        self.gemini_model = "gemini-2.0-flash"
        self.pose_mappings = {}  # pose_idx -> key_mapping
        self.model = None
        self.SmartCNN = None
        
        # Default transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def add_pose(self, pose_name, key_mapping):
        """Add a pose mapping"""
        pose_idx = len(self.pose_mappings)
        self.pose_mappings[pose_idx] = {
            'name': pose_name,
            'key': key_mapping
        }
        print(f"➕ Added pose {pose_idx}: {pose_name} → {key_mapping}")
    
    def capture_pose_images(self, pose_idx, num_images=30):
        """Capture images for a specific pose"""
        if pose_idx not in self.pose_mappings:
            print(f"❌ Pose {pose_idx} not configured")
            return False
        
        pose_name = self.pose_mappings[pose_idx]['name']
        pose_dir = os.path.join('training_data', f'pose_{pose_idx}')
        os.makedirs(pose_dir, exist_ok=True)
        
        print(f"📸 Starting capture for {pose_name}")
        print(f"🎯 Will capture {num_images} images")
        print("👋 Show your pose and it will auto-capture!")
        print("⏹️ Press 'q' to stop capturing")
        
        cap = cv2.VideoCapture(0)
        count = 0
        
        try:
            while count < num_images and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Save image
                img_path = os.path.join(pose_dir, f'{count}.jpg')
                cv2.imwrite(img_path, frame)
                count += 1
                
                # Add overlay
                cv2.putText(frame, f"Capturing {pose_name}: {count}/{num_images}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to stop", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow(f"Capturing {pose_name}", frame)
                
                # Check for quit
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.1)  # 100ms between captures
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"✅ Captured {count} images for {pose_name}")
        return count > 0
    
    def generate_smart_ai_model(self):
        """Generate Smart AI model using Gemini (GameAble approach)"""
        if not self.google_ai_key:
            print("⚠️ No Google AI key - using default CNN")
            return False
        
        print("🤖 Generating Smart AI model with Gemini...")
        
        # Collect sample images
        sample_images = []
        for pose_idx in self.pose_mappings.keys():
            pose_dir = os.path.join('training_data', f'pose_{pose_idx}')
            if os.path.isdir(pose_dir):
                files = [f for f in os.listdir(pose_dir) if f.endswith('.jpg')]
                if files:
                    img_path = os.path.join(pose_dir, files[0])
                    with open(img_path, 'rb') as img_file:
                        img_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                    sample_images.append(img_b64)
        
        if not sample_images:
            print("❌ No sample images found")
            return False
        
        # Gemini prompt
        num_classes = len(self.pose_mappings)
        prompt = (
            f"Generate a PyTorch CNN model class for image classification. "
            f"The model must take input images of shape (3, 224, 224) and output logits for {num_classes} classes. "
            f"Only output the class definition (no training loop, no extra text). "
            f"The class must be named 'SmartCNN' and inherit from nn.Module. "
            f"The __init__ method must take a num_classes argument. "
            f"Also provide preprocessing pipeline as torchvision.transforms.Compose named 'transform'. "
            f"Recommend hyperparameters as JSON. "
            f"Format: CODE:\\n<model code>\\nPREPROCESSING:\\n<preprocessing>\\nHYPERPARAMETERS:\\n<json>\\nREASON:\\n<reason>"
        )
        
        # Call Gemini
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.google_ai_key}"
            data = {
                "contents": [{
                    "role": "user", 
                    "parts": [
                        {"text": prompt},
                        *[{"inlineData": {"mimeType": "image/jpeg", "data": img_b64}} for img_b64 in sample_images]
                    ]
                }]
            }
            
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            
            # Parse response
            text = ""
            for candidate in result.get('candidates', []):
                parts = candidate.get('content', {}).get('parts', [])
                for part in parts:
                    if 'text' in part:
                        text = part['text']
                        break
                if text:
                    break
            
            if not text:
                print("❌ Gemini returned empty response")
                return False
            
            # Extract components
            code_match = re.search(r'CODE:\s*([\s\S]+?)PREPROCESSING:', text)
            preproc_match = re.search(r'PREPROCESSING:\s*([\s\S]+?)HYPERPARAMETERS:', text)
            hyper_match = re.search(r'HYPERPARAMETERS:\s*([\{\[].*?[\}\]])', text, re.DOTALL)
            reason_match = re.search(r'REASON:\s*([\s\S]+)', text)
            
            # Extract code
            if code_match:
                code = code_match.group(1)
                code = re.sub(r'```+\s*python', '', code, flags=re.IGNORECASE)
                code = re.sub(r'```+', '', code)
                self.smart_code = code.strip()
                
                # Save and import
                with open('smart_cnn.py', 'w') as f:
                    f.write(self.smart_code)
                
                spec = importlib.util.spec_from_file_location("smart_cnn", "smart_cnn.py")
                smart_cnn_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(smart_cnn_module)
                self.SmartCNN = smart_cnn_module.SmartCNN
                print("✅ Smart AI model generated!")
            
            # Extract preprocessing
            if preproc_match:
                preproc = preproc_match.group(1)
                try:
                    namespace = {'transforms': transforms}
                    exec(preproc, namespace)
                    if 'transform' in namespace:
                        self.transform = namespace['transform']
                        print("✅ Updated preprocessing pipeline")
                except Exception as e:
                    print(f"⚠️ Could not apply preprocessing: {e}")
            
            # Extract hyperparameters
            self.smart_hyperparams = {"epochs": 20, "batch_size": 16, "learning_rate": 0.001}
            if hyper_match:
                try:
                    self.smart_hyperparams = json.loads(hyper_match.group(1))
                    print(f"✅ Recommended hyperparameters: {self.smart_hyperparams}")
                except Exception as e:
                    print(f"⚠️ Could not parse hyperparameters: {e}")
            
            # Show reasoning
            if reason_match:
                reason = reason_match.group(1).strip()
                print(f"🧠 Gemini's reasoning: {reason}")
            
            return True
            
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            return False
    
    def train_model(self, epochs=None, batch_size=None, learning_rate=None):
        """Train the model (Smart AI or default CNN)"""
        print("🏋️ Training gesture model...")
        
        # Try Smart AI first
        if self.google_ai_key and not self.SmartCNN:
            print("🤖 Generating Smart AI...")
            self.generate_smart_ai_model()
        
        # Use Smart AI hyperparams if available
        if hasattr(self, 'smart_hyperparams'):
            epochs = epochs or self.smart_hyperparams.get('epochs', 20)
            batch_size = batch_size or self.smart_hyperparams.get('batch_size', 16)  
            learning_rate = learning_rate or self.smart_hyperparams.get('learning_rate', 0.001)
        else:
            epochs = epochs or 20
            batch_size = batch_size or 16
            learning_rate = learning_rate or 0.001
        
        try:
            # Create dataset
            dataset = PoseDataset('training_data', transform=self.transform)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Create model
            num_classes = len(self.pose_mappings)
            if self.SmartCNN:
                self.model = self.SmartCNN(num_classes)
                print("🤖 Using Smart AI CNN!")
            else:
                self.model = CNN(num_classes)
                print("📦 Using default CNN")
            
            # Training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            print(f"🎯 Training: {len(dataset)} images, {num_classes} poses")
            print(f"⚙️ Params: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
            
            self.model.train()
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, (inputs, labels) in enumerate(dataloader):
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                accuracy = 100 * correct / total
                avg_loss = running_loss / len(dataloader)
                
                if epoch % 5 == 0:
                    print(f"📈 Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
            
            # Save model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'pose_mappings': self.pose_mappings,
                'num_classes': num_classes,
                'smart_ai_used': self.SmartCNN is not None
            }, 'models/gameable_model.pth')
            
            print(f"✅ Training complete! Final Accuracy: {accuracy:.2f}%")
            if self.SmartCNN:
                print("🤖 Trained with Smart AI optimization!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def load_model(self):
        """Load trained model"""
        model_path = 'models/gameable_model.pth'
        if not os.path.exists(model_path):
            return False
        
        try:
            checkpoint = torch.load(model_path)
            self.pose_mappings = checkpoint['pose_mappings']
            num_classes = checkpoint['num_classes']
            
            # Load Smart AI if it was used
            if checkpoint.get('smart_ai_used', False) and os.path.exists('smart_cnn.py'):
                try:
                    spec = importlib.util.spec_from_file_location("smart_cnn", "smart_cnn.py")
                    smart_cnn_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(smart_cnn_module)
                    self.SmartCNN = smart_cnn_module.SmartCNN
                    self.model = self.SmartCNN(num_classes)
                    print("🤖 Loaded Smart AI model")
                except:
                    self.model = CNN(num_classes)
                    print("📦 Loaded default CNN model")
            else:
                self.model = CNN(num_classes)
                print("📦 Loaded default CNN model")
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✅ Model loaded with {num_classes} poses")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def predict_pose(self, frame):
        """Predict pose from camera frame"""
        if self.model is None:
            return None, 0.0
        
        try:
            # Preprocess
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tensor = self.transform(img).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted = torch.max(probabilities, 1)
                
                pose_idx = predicted.item()
                confidence_score = confidence.item()
                
                if pose_idx in self.pose_mappings:
                    return self.pose_mappings[pose_idx], confidence_score
                
                return None, 0.0
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, 0.0
    
    def get_pose_key(self, pose_info):
        """Get key mapping for a pose"""
        if pose_info:
            return pose_info.get('key')
        return None
    
    def list_poses(self):
        """List all configured poses"""
        if not self.pose_mappings:
            print("📝 No poses configured")
            return
        
        print("\\n🎯 Configured Poses:")
        print("=" * 40)
        for pose_idx, pose_info in self.pose_mappings.items():
            print(f"   Pose {pose_idx}: {pose_info['name']} → {pose_info['key']}")
        print("=" * 40)

# Example usage
if __name__ == "__main__":
    trainer = GameAbleTrainer()
    
    # Add poses
    trainer.add_pose("c_hand", "c")
    trainer.add_pose("peace_sign", "e")
    
    # Capture images
    trainer.capture_pose_images(0, 30)  # C hand
    trainer.capture_pose_images(1, 30)  # Peace sign
    
    # Train model
    trainer.train_model()