"""
AI Model Generator using Gemini
Generates optimized CNN architectures for hand gesture recognition
"""

import google.generativeai as genai
import torch
import torch.nn as nn
import numpy as np
import json
import ast
import re
from typing import Dict, List, Tuple

class GeminiModelGenerator:
    """
    Generates custom CNN architectures using Gemini AI
    Specialized for hand gesture recognition using MediaPipe landmarks
    """
    
    def __init__(self, google_ai_key: str):
        self.api_key = google_ai_key
        genai.configure(api_key=google_ai_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def analyze_gesture_complexity(self, training_data: Dict) -> str:
        """Analyze gesture complexity to determine optimal architecture"""
        num_gestures = len(set(training_data["labels"]))
        num_samples = len(training_data["landmarks"])
        
        # Calculate feature variance (more complex gestures = higher variance)
        landmarks_array = np.array(training_data["landmarks"])
        feature_variance = np.mean(np.var(landmarks_array, axis=0))
        
        if num_gestures <= 5 and feature_variance < 0.01:
            return "simple"
        elif num_gestures <= 10 and feature_variance < 0.05:
            return "medium"
        else:
            return "complex"
    
    def generate_custom_architecture(self, training_data: Dict, gesture_mappings: Dict) -> str:
        """
        Generate custom CNN architecture using Gemini
        Optimized for hand gesture recognition patterns
        """
        complexity = self.analyze_gesture_complexity(training_data)
        num_gestures = len(set(training_data["labels"]))
        num_samples = len(training_data["landmarks"])
        
        # Analyze gesture types for architecture hints
        gesture_types = list(gesture_mappings.keys())
        gesture_descriptions = self.analyze_gesture_types(gesture_types)
        
        prompt = f"""
You are an expert in designing CNN architectures for hand gesture recognition using MediaPipe landmarks.

CONTEXT:
- Input: 42 features (21 hand landmarks × 2 coordinates)
- Output: {num_gestures} gesture classes
- Training samples: {num_samples}
- Complexity level: {complexity}
- Gesture types: {gesture_types}
- Gesture patterns: {gesture_descriptions}

REQUIREMENTS:
- Design a PyTorch CNN model class
- Optimize for hand gesture landmarks (not images)
- Use appropriate architecture for {complexity} complexity
- Include proper regularization for small dataset
- Target 99%+ accuracy with minimal overfitting

CONSTRAINTS:
- Input size: 42 features
- Output size: {num_gestures} classes
- Fast inference required (< 2ms)
- Memory efficient for real-time use

Generate a complete PyTorch model class with:
1. __init__ method with proper layer definitions
2. forward method
3. Optimized for hand gesture landmark patterns
4. Comments explaining design choices

Return ONLY the Python class code, no explanations.
"""
        
        try:
            response = self.model.generate_content(prompt)
            generated_code = response.text.strip()
            
            # Extract the class code
            class_code = self.extract_class_code(generated_code)
            
            if class_code:
                print("🤖 Generated custom CNN architecture using Gemini AI")
                return class_code
            else:
                print("⚠️ Falling back to default architecture")
                return self.get_default_architecture(num_gestures)
                
        except Exception as e:
            print(f"❌ Gemini generation failed: {e}")
            return self.get_default_architecture(num_gestures)
    
    def analyze_gesture_types(self, gesture_types: List[str]) -> str:
        """Analyze gesture types to provide architecture hints"""
        movement_gestures = ["point_left", "point_right", "point_up", "closed_fist", "open_palm"]
        action_gestures = ["rock_sign", "peace_sign", "thumbs_up"]
        shape_gestures = ["c_hand", "l_shape", "ok_sign"]
        
        movement_count = sum(1 for g in gesture_types if any(mg in g for mg in movement_gestures))
        action_count = sum(1 for g in gesture_types if any(ag in g for ag in action_gestures))
        shape_count = sum(1 for g in gesture_types if any(sg in g for sg in shape_gestures))
        
        if movement_count > action_count + shape_count:
            return "Movement-focused gestures requiring directional feature analysis"
        elif action_count > movement_count + shape_count:
            return "Action-focused gestures requiring finger pattern recognition"
        elif shape_count > movement_count + action_count:
            return "Shape-focused gestures requiring geometric pattern analysis"
        else:
            return "Mixed gesture types requiring comprehensive feature extraction"
    
    def extract_class_code(self, generated_text: str) -> str:
        """Extract the CNN class code from Gemini's response"""
        # Remove markdown code blocks
        generated_text = re.sub(r'```python\n?', '', generated_text)
        generated_text = re.sub(r'```', '', generated_text)
        
        # Find class definition
        lines = generated_text.split('\n')
        class_start = None
        class_lines = []
        
        for i, line in enumerate(lines):
            if line.strip().startswith('class ') and 'nn.Module' in line:
                class_start = i
                break
        
        if class_start is not None:
            # Extract the entire class
            for line in lines[class_start:]:
                class_lines.append(line)
                # Stop at next class or function at root level
                if line and not line.startswith(' ') and not line.startswith('\t') and class_lines and len(class_lines) > 1:
                    class_lines.pop()  # Remove the last line that starts new definition
                    break
            
            return '\n'.join(class_lines)
        
        return None
    
    def get_default_architecture(self, num_classes: int) -> str:
        """Fallback default architecture"""
        return f'''
class CustomHandGestureCNN(nn.Module):
    def __init__(self, input_size=42, num_classes={num_classes}):
        super(CustomHandGestureCNN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
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
'''
    
    def generate_training_hyperparameters(self, training_data: Dict) -> Dict:
        """Generate optimal hyperparameters using Gemini"""
        num_samples = len(training_data["landmarks"])
        num_gestures = len(set(training_data["labels"]))
        complexity = self.analyze_gesture_complexity(training_data)
        
        prompt = f"""
You are an expert in deep learning hyperparameter optimization for hand gesture recognition.

DATASET INFO:
- Training samples: {num_samples}
- Number of gestures: {num_gestures}
- Complexity: {complexity}
- Input: MediaPipe hand landmarks (42 features)

Generate optimal hyperparameters for training a CNN on this small dataset.
Consider overfitting prevention and fast convergence.

Return a JSON object with these fields:
- learning_rate
- batch_size
- epochs
- weight_decay
- dropout_rate
- optimizer (adam/sgd)

Return ONLY the JSON, no explanations.
"""
        
        try:
            response = self.model.generate_content(prompt)
            hyperparams_text = response.text.strip()
            
            # Extract JSON
            hyperparams_text = re.sub(r'```json\n?', '', hyperparams_text)
            hyperparams_text = re.sub(r'```', '', hyperparams_text)
            
            hyperparams = json.loads(hyperparams_text)
            print("🎯 Generated optimal hyperparameters using Gemini AI")
            return hyperparams
            
        except Exception as e:
            print(f"⚠️ Using default hyperparameters: {e}")
            return {
                "learning_rate": 0.001,
                "batch_size": 8,
                "epochs": 100,
                "weight_decay": 0.0001,
                "dropout_rate": 0.3,
                "optimizer": "adam"
            }
    
    def create_model_from_code(self, model_code: str, num_classes: int):
        """Dynamically create model from generated code"""
        try:
            # Create a namespace for execution
            namespace = {
                'nn': nn,
                'torch': torch,
                'np': np
            }
            
            # Execute the generated code
            exec(model_code, namespace)
            
            # Find the model class (should be the last class defined)
            model_classes = [obj for name, obj in namespace.items() 
                           if isinstance(obj, type) and issubclass(obj, nn.Module) and obj != nn.Module]
            
            if model_classes:
                ModelClass = model_classes[-1]  # Take the last defined class
                return ModelClass(input_size=42, num_classes=num_classes)
            else:
                raise ValueError("No valid model class found in generated code")
                
        except Exception as e:
            print(f"❌ Error creating model from generated code: {e}")
            # Fallback to default
            from local_gesture_cnn import HandGestureCNN
            return HandGestureCNN(input_size=42, num_classes=num_classes)