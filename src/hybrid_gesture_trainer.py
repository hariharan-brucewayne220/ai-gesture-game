"""
Hybrid Gesture Trainer - Combines MediaPipe + GameAble CNN approach
Uses MediaPipe for basic gestures, CNN for custom gestures
"""

try:
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
    import json
    import requests
    import base64
    import re
    import importlib.util
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies for hybrid trainer: {e}")
    DEPENDENCIES_AVAILABLE = False

class CustomGestureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Get all gesture directories
        gesture_dirs = [d for d in os.listdir(data_dir) if d.startswith('gesture_')]
        gesture_dirs.sort()
        
        for gesture_idx, gesture_dir in enumerate(gesture_dirs):
            gesture_path = os.path.join(data_dir, gesture_dir)
            if os.path.isdir(gesture_path):
                image_files = [f for f in os.listdir(gesture_path) if f.endswith('.jpg')]
                
                for img_name in image_files:
                    self.images.append(os.path.join(gesture_path, img_name))
                    self.labels.append(gesture_idx)
        
        if not self.images:
            raise ValueError("No images found in training data")
            
        self.labels = torch.tensor(self.labels)
        print(f"📊 Loaded {len(self.images)} images")
        print(f"🎯 Gesture distribution: {torch.bincount(self.labels).tolist()}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class CustomGestureCNN(nn.Module):
    """Lightweight CNN for custom gesture recognition"""
    def __init__(self, num_classes):
        super(CustomGestureCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block  
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Calculate the size after conv layers (224x224 -> 28x28)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

class HybridGestureTrainer:
    """
    Hybrid trainer that combines MediaPipe basic gestures with CNN custom gestures
    """
    
    def __init__(self, google_ai_key=None):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Missing dependencies for hybrid trainer")
        self.custom_gestures = {}  # gesture_name -> key_mapping
        self.training_data_dir = "custom_gesture_data"
        self.model_path = "hybrid_gesture_model.pth"
        self.config_path = "hybrid_gesture_config.json"
        
        # Gemini AI setup
        self.google_ai_key = google_ai_key
        self.gemini_model = "gemini-2.0-flash"
        
        # Create directories
        os.makedirs(self.training_data_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Image preprocessing (default, will be updated by Gemini)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Model
        self.model = None
        self.SmartCNN = None  # Gemini-generated model class
        self.is_capturing = False
        self.current_gesture = None
        
        # Load existing config
        self.load_config()
    
    def add_custom_gesture(self, gesture_name, key_mapping):
        """Add a new custom gesture"""
        self.custom_gestures[gesture_name] = key_mapping
        print(f"➕ Added custom gesture: {gesture_name} → {key_mapping}")
        self.save_config()
    
    def start_capture(self, gesture_name, num_images=50):
        """Start capturing images for a custom gesture"""
        if gesture_name not in self.custom_gestures:
            print(f"❌ Gesture {gesture_name} not configured. Add it first.")
            return False
        
        self.current_gesture = gesture_name
        self.gesture_dir = os.path.join(self.training_data_dir, f"gesture_{gesture_name}")
        os.makedirs(self.gesture_dir, exist_ok=True)
        
        print(f"📸 Starting capture for gesture: {gesture_name}")
        print(f"🎯 Will capture {num_images} images")
        print("👋 Show your gesture and it will auto-capture!")
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
                img_path = os.path.join(self.gesture_dir, f'{gesture_name}_{count:03d}.jpg')
                cv2.imwrite(img_path, frame)
                count += 1
                
                # Add overlay
                cv2.putText(frame, f"Capturing {gesture_name}: {count}/{num_images}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to stop", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow(f"Capturing {gesture_name}", frame)
                
                # Check for quit
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.1)  # 100ms between captures
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"✅ Captured {count} images for {gesture_name}")
        return count > 0
    
    def generate_smart_ai_model(self):
        """Generate custom CNN architecture using Gemini AI (GameAble approach)"""
        if not self.google_ai_key:
            print("⚠️ No Google AI key - using default CNN architecture")
            return False
        
        print("🤖 Generating Smart AI model with Gemini...")
        
        # Collect sample images from each gesture class
        sample_images = []
        for gesture_name in self.custom_gestures.keys():
            gesture_dir = os.path.join(self.training_data_dir, f"gesture_{gesture_name}")
            if os.path.isdir(gesture_dir):
                files = [f for f in os.listdir(gesture_dir) if f.endswith('.jpg')]
                if files:
                    img_path = os.path.join(gesture_dir, files[0])
                    with open(img_path, 'rb') as img_file:
                        img_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                    sample_images.append(img_b64)
        
        if not sample_images:
            print("❌ No sample images found for analysis")
            return False
        
        # Compose Gemini prompt
        num_classes = len(self.custom_gestures)
        prompt = (
            f"Generate a PyTorch CNN model class for image classification. "
            f"The model must take input images of shape (3, 224, 224) and output logits for {num_classes} classes. "
            f"Only output the class definition (no training loop, no extra text). "
            f"The class must be named 'SmartCNN' and inherit from nn.Module. "
            f"The __init__ method must take a num_classes argument. "
            f"Also provide preprocessing pipeline as torchvision.transforms.Compose named 'transform'. "
            f"Recommend hyperparameters as JSON: {{\"epochs\": int, \"batch_size\": int, \"learning_rate\": float}}. "
            f"Provide reasoning for architecture choices. "
            f"Format: CODE:\\n<model code>\\nPREPROCESSING:\\n<preprocessing>\\nHYPERPARAMETERS:\\n<json>\\nREASON:\\n<reason>"
        )
        
        # Call Gemini API
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
            
            # Parse Gemini response
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
            
            # Extract and clean code
            if code_match:
                code = code_match.group(1)
                code = re.sub(r'```+\s*python', '', code, flags=re.IGNORECASE)
                code = re.sub(r'```+', '', code)
                self.smart_code = code.strip()
            
            # Extract preprocessing
            if preproc_match:
                preproc = preproc_match.group(1)
                preproc = re.sub(r'```+\s*python', '', preproc, flags=re.IGNORECASE)
                preproc = re.sub(r'```+', '', preproc)
                
                # Execute preprocessing code to update transform
                try:
                    namespace = {'transforms': transforms, 'nn': nn}
                    exec(preproc, namespace)
                    if 'transform' in namespace:
                        self.transform = namespace['transform']
                        print("✅ Updated preprocessing pipeline with Gemini recommendations")
                except Exception as e:
                    print(f"⚠️ Could not apply preprocessing: {e}")
            
            # Extract hyperparameters
            self.smart_hyperparams = {"epochs": 20, "batch_size": 16, "learning_rate": 0.001}
            if hyper_match:
                try:
                    self.smart_hyperparams = json.loads(hyper_match.group(1))
                    print(f"✅ Gemini recommended hyperparameters: {self.smart_hyperparams}")
                except Exception as e:
                    print(f"⚠️ Could not parse hyperparameters: {e}")
            
            # Show reasoning
            if reason_match:
                reason = reason_match.group(1).strip()
                print(f"🧠 Gemini's reasoning: {reason}")
            
            # Save and import the smart model
            with open('smart_cnn.py', 'w') as f:
                f.write(self.smart_code)
            
            # Dynamic import
            spec = importlib.util.spec_from_file_location("smart_cnn", "smart_cnn.py")
            smart_cnn_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(smart_cnn_module)
            self.SmartCNN = smart_cnn_module.SmartCNN
            
            print("🚀 Smart AI model generated successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Gemini AI generation failed: {e}")
            return False

    def train_model(self, epochs=None, batch_size=None, learning_rate=None, use_smart_ai=True):
        """Train the custom gesture CNN model with optional Smart AI"""
        print("🏋️ Training custom gesture model...")
        
        # Try to generate Smart AI model first
        if use_smart_ai and self.google_ai_key and not self.SmartCNN:
            print("🤖 Generating Smart AI architecture...")
            self.generate_smart_ai_model()
        
        # Use Smart AI hyperparameters if available
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
            dataset = CustomGestureDataset(self.training_data_dir, transform=self.transform)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Create model (Smart AI if available, otherwise default)
            num_classes = len(self.custom_gestures)
            if self.SmartCNN:
                self.model = self.SmartCNN(num_classes)
                print("🤖 Using Gemini-generated Smart CNN!")
            else:
                self.model = CustomGestureCNN(num_classes)
                print("📦 Using default CNN architecture")
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            print(f"🎯 Training on {len(dataset)} images, {num_classes} gestures")
            print(f"⚙️ Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
            
            # Training loop
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
            model_save_path = os.path.join("models", self.model_path)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'custom_gestures': self.custom_gestures,
                'num_classes': num_classes,
                'smart_ai_used': self.SmartCNN is not None
            }, model_save_path)
            
            print(f"✅ Model trained and saved! Final Accuracy: {accuracy:.2f}%")
            if self.SmartCNN:
                print("🤖 Trained with Gemini Smart AI optimization!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def load_model(self):
        """Load the trained model"""
        model_load_path = os.path.join("models", self.model_path)
        
        if not os.path.exists(model_load_path):
            print("⚠️ No trained model found")
            return False
        
        try:
            checkpoint = torch.load(model_load_path)
            num_classes = checkpoint['num_classes']
            
            self.model = CustomGestureCNN(num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.custom_gestures = checkpoint['custom_gestures']
            
            print(f"✅ Loaded model with {num_classes} custom gestures")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def predict_custom_gesture(self, frame):
        """Predict custom gesture from camera frame"""
        if self.model is None:
            return "none", 0.0
        
        try:
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tensor = self.transform(img).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get gesture name
                gesture_names = list(self.custom_gestures.keys())
                if predicted.item() < len(gesture_names):
                    gesture_name = gesture_names[predicted.item()]
                    confidence_score = confidence.item()
                    
                    return gesture_name, confidence_score
            
            return "none", 0.0
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "none", 0.0
    
    def get_gesture_key_mapping(self, gesture_name):
        """Get the key mapping for a gesture"""
        return self.custom_gestures.get(gesture_name, None)
    
    def save_config(self):
        """Save configuration"""
        config = {
            'custom_gestures': self.custom_gestures
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self):
        """Load configuration"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.custom_gestures = config.get('custom_gestures', {})
                print(f"📂 Loaded {len(self.custom_gestures)} custom gestures")
            except Exception as e:
                print(f"⚠️ Failed to load config: {e}")
    
    def list_custom_gestures(self):
        """List all custom gestures"""
        if not self.custom_gestures:
            print("📝 No custom gestures configured")
            return
        
        print("\n🎯 Custom Gestures:")
        print("=" * 40)
        for gesture, key in self.custom_gestures.items():
            print(f"   {gesture} → {key}")
        print("=" * 40)

# Example usage
if __name__ == "__main__":
    trainer = HybridGestureTrainer()
    
    # Add C gesture for crouching
    trainer.add_custom_gesture("c_shape", "c")
    
    # Capture training data
    print("Starting capture for C gesture...")
    trainer.start_capture("c_shape", num_images=30)
    
    # Train model
    print("Training model...")
    trainer.train_model(epochs=15)
    
    print("Custom gesture trainer ready!")