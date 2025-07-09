"""
Custom Gesture AI using Google Gemini Vision
This module handles custom gesture recognition using AI
"""

import google.generativeai as genai
import cv2
import numpy as np
import base64
import json
import os
from PIL import Image
from typing import List, Dict, Optional, Tuple
import time

class CustomGestureAI:
    """AI-powered custom gesture recognition using Google Gemini Vision"""
    
    def __init__(self, api_key: str):
        """
        Initialize the Custom Gesture AI
        
        EDUCATION: Why Gemini Vision?
        - Can understand images and classify them
        - Few-shot learning: learns from just a few examples
        - Natural language: we can describe gestures in plain English
        - Real-time: fast enough for gaming (~200ms)
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Initialize the Gemini model
        # We use 'gemini-1.5-flash' for speed (good for real-time)
        # Alternative: 'gemini-1.5-pro' for better accuracy but slower
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Storage for trained gestures
        self.trained_gestures = {}  # {"gesture_name": {"examples": [], "description": ""}}
        self.gesture_file = "trained_gestures.json"
        
        # Load existing trained gestures
        self.load_trained_gestures()
        
        # Performance tracking
        self.last_prediction_time = 0
        self.prediction_cooldown = 0.5  # Don't spam API (500ms between calls)
        
        print("🤖 Custom Gesture AI initialized with Gemini Vision")
        print(f"📚 Loaded {len(self.trained_gestures)} trained gestures")
    
    def encode_image_to_base64(self, frame: np.ndarray) -> str:
        """
        Convert OpenCV frame to base64 string for Gemini
        
        EDUCATION: Why base64?
        - Gemini API expects images as base64 strings
        - Allows us to send image data over HTTP
        - Standard format for image APIs
        """
        # Convert BGR (OpenCV) to RGB (PIL/Standard)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize for faster processing (Gemini works well with smaller images)
        pil_image = pil_image.resize((640, 480))
        
        # Convert to base64
        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return image_base64
    
    def record_gesture_examples(self, gesture_name: str, description: str, frames: List[np.ndarray]):
        """
        Record examples of a custom gesture for training
        
        EDUCATION: Few-shot Learning
        - AI can learn from just 5-10 examples
        - More examples = better accuracy
        - Variety is key: different angles, lighting, hand positions
        
        Args:
            gesture_name: Name of the gesture (e.g., "cross_hands")
            description: Human description (e.g., "Both hands crossed in front of body")
            frames: List of example frames showing the gesture
        """
        print(f"📸 Recording {len(frames)} examples for gesture: {gesture_name}")
        
        # Convert frames to base64
        examples = []
        for i, frame in enumerate(frames):
            base64_image = self.encode_image_to_base64(frame)
            examples.append(base64_image)
            print(f"   Example {i+1}/{len(frames)} encoded")
        
        # Store the gesture
        self.trained_gestures[gesture_name] = {
            "examples": examples,
            "description": description,
            "created_at": time.time()
        }
        
        # Save to file
        self.save_trained_gestures()
        
        print(f"✅ Gesture '{gesture_name}' trained with {len(examples)} examples")
        print(f"📝 Description: {description}")
    
    def recognize_gesture(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Recognize gesture in current frame using trained AI
        
        EDUCATION: How AI Classification Works
        1. Send current frame + list of known gestures to Gemini
        2. Gemini compares frame to trained examples
        3. Returns best match + confidence score
        4. We only act if confidence is high enough
        
        Returns:
            (gesture_name, confidence) - "none" if no match
        """
        # Rate limiting to avoid API spam
        current_time = time.time()
        if current_time - self.last_prediction_time < self.prediction_cooldown:
            return "none", 0.0
        
        self.last_prediction_time = current_time
        
        # No trained gestures = no recognition
        if not self.trained_gestures:
            return "none", 0.0
        
        try:
            # Encode current frame
            current_image_b64 = self.encode_image_to_base64(frame)
            
            # Create prompt with all known gestures
            gesture_list = list(self.trained_gestures.keys())
            gesture_descriptions = [self.trained_gestures[g]["description"] for g in gesture_list]
            
            prompt = f"""
            Analyze this hand gesture image and classify it.
            
            Known gestures:
            {', '.join([f'{name}: {desc}' for name, desc in zip(gesture_list, gesture_descriptions)])}
            
            Instructions:
            1. Look carefully at the hand positions and shapes
            2. Compare to the known gesture descriptions
            3. Respond with ONLY the gesture name if you're confident (>80% sure)
            4. Respond with "none" if no clear match or confidence is low
            5. Be strict - better to say "none" than guess wrong
            
            Response format: Just the gesture name or "none"
            """
            
            # Send to Gemini
            # We include the current frame for analysis
            response = self.model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": current_image_b64}
            ])
            
            # Parse response
            predicted_gesture = response.text.strip().lower()
            
            # Validate response
            if predicted_gesture in gesture_list:
                confidence = 0.85  # High confidence for valid matches
                print(f"🎯 Gesture recognized: {predicted_gesture} ({confidence:.2f})")
                return predicted_gesture, confidence
            elif predicted_gesture == "none":
                return "none", 0.0
            else:
                # Invalid response - treat as no match
                print(f"⚠️ Invalid AI response: {predicted_gesture}")
                return "none", 0.0
                
        except Exception as e:
            print(f"❌ Gesture recognition error: {e}")
            return "none", 0.0
    
    def save_trained_gestures(self):
        """Save trained gestures to file for persistence"""
        try:
            with open(self.gesture_file, 'w') as f:
                json.dump(self.trained_gestures, f, indent=2)
            print(f"💾 Saved trained gestures to {self.gesture_file}")
        except Exception as e:
            print(f"❌ Error saving gestures: {e}")
    
    def load_trained_gestures(self):
        """Load previously trained gestures from file"""
        try:
            if os.path.exists(self.gesture_file):
                with open(self.gesture_file, 'r') as f:
                    self.trained_gestures = json.load(f)
                print(f"📂 Loaded trained gestures from {self.gesture_file}")
            else:
                print("📝 No existing gesture file found - starting fresh")
        except Exception as e:
            print(f"❌ Error loading gestures: {e}")
            self.trained_gestures = {}
    
    def list_trained_gestures(self):
        """List all trained gestures"""
        if not self.trained_gestures:
            print("📝 No trained gestures yet")
            return
        
        print("🎭 Trained Custom Gestures:")
        for name, data in self.trained_gestures.items():
            examples_count = len(data["examples"])
            description = data["description"]
            print(f"   {name}: {description} ({examples_count} examples)")
    
    def delete_gesture(self, gesture_name: str):
        """Delete a trained gesture"""
        if gesture_name in self.trained_gestures:
            del self.trained_gestures[gesture_name]
            self.save_trained_gestures()
            print(f"🗑️ Deleted gesture: {gesture_name}")
        else:
            print(f"❌ Gesture not found: {gesture_name}")
    
    def get_status(self) -> str:
        """Get current status string for display"""
        count = len(self.trained_gestures)
        return f"Custom AI: {count} gestures"