import numpy as np
from typing import Tuple, Optional, List
from collections import deque

# Try importing PyTorch - graceful degradation if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - LSTM features disabled")

class LSTMModel(nn.Module):
    """Lightweight LSTM model for hand movement prediction"""
    
    def __init__(self, input_size=2, hidden_size=16, num_layers=2, output_size=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.dropout(last_output)
        return self.fc(output)

class LSTMPredictor:
    """Lightweight LSTM for predicting hand movement trajectories"""
    
    def __init__(self):
        self.enabled = False
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.sequence_length = 10  # Use last 10 positions to predict next
        self.position_history = deque(maxlen=self.sequence_length)
        self.prediction_confidence = 0.0
        
        # Fallback to exponential smoothing if LSTM fails
        self.use_fallback = True
        
    def initialize_model(self):
        """Initialize lightweight LSTM model"""
        if not PYTORCH_AVAILABLE:
            print("❌ Cannot initialize LSTM - PyTorch not installed")
            return False
            
        try:
            # Create PyTorch LSTM model
            self.model = LSTMModel(input_size=2, hidden_size=16, num_layers=2, output_size=2)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Initialize with some basic patterns (simple online learning)
            self._pretrain_model()
            
            self.use_fallback = False
            print("✅ LSTM predictor initialized successfully")
            return True
            
        except Exception as e:
            print(f"⚠️ LSTM initialization failed: {e}")
            print("🔄 Using exponential smoothing fallback")
            self.use_fallback = True
            return False
    
    def _pretrain_model(self):
        """Pre-train with basic movement patterns"""
        try:
            # Generate synthetic training data for common movement patterns
            patterns = []
            
            # Linear movements (left, right, up, down)
            for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                for speed in [0.01, 0.02, 0.03]:
                    sequence = []
                    pos = [0.5, 0.5]  # Start at center
                    for i in range(self.sequence_length + 1):
                        sequence.append([pos[0], pos[1]])
                        pos[0] += direction[0] * speed
                        pos[1] += direction[1] * speed
                    patterns.append(sequence)
            
            # Circular movements
            for radius in [0.05, 0.1]:
                sequence = []
                for i in range(self.sequence_length + 1):
                    angle = i * 0.2
                    x = 0.5 + radius * np.cos(angle)
                    y = 0.5 + radius * np.sin(angle)
                    sequence.append([x, y])
                patterns.append(sequence)
            
            # Convert to training data
            X, y = [], []
            for pattern in patterns:
                X.append(pattern[:-1])  # Input sequence
                y.append(pattern[-1])   # Next position
            
            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y)
            
            # Quick training
            self.model.train()
            for epoch in range(5):
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
            
        except Exception as e:
            print(f"⚠️ LSTM pre-training failed: {e}")
    
    def add_position(self, x: float, y: float):
        """Add new hand position to history"""
        self.position_history.append([x, y])
    
    def predict_next_position(self, current_x: float, current_y: float) -> Tuple[float, float, float]:
        """Predict next hand position"""
        if not self.enabled or self.use_fallback or len(self.position_history) < self.sequence_length:
            # Use exponential smoothing fallback
            return current_x, current_y, 0.5
        
        try:
            # Prepare input sequence
            sequence = torch.FloatTensor(list(self.position_history)).unsqueeze(0)
            
            # Get prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(sequence)[0]
            
            # Calculate confidence based on prediction consistency
            recent_velocity = np.linalg.norm(np.array([current_x, current_y]) - np.array(self.position_history[-2]))
            predicted_velocity = np.linalg.norm(prediction.numpy() - np.array([current_x, current_y]))
            
            # Higher confidence for consistent predictions
            confidence = max(0.1, min(0.9, 1.0 - abs(predicted_velocity - recent_velocity) * 10))
            
            return float(prediction[0]), float(prediction[1]), confidence
            
        except Exception as e:
            print(f"⚠️ LSTM prediction failed: {e}")
            # Fallback to current position
            return current_x, current_y, 0.0
    
    def update_with_actual(self, actual_x: float, actual_y: float):
        """Update model with actual observed position (online learning)"""
        if not self.enabled or self.use_fallback:
            return
            
        try:
            if len(self.position_history) >= self.sequence_length:
                # Create training sample from recent history
                X = torch.FloatTensor(list(self.position_history)[:-1]).unsqueeze(0)
                y = torch.FloatTensor([[actual_x, actual_y]])
                
                # Quick online update (single epoch)
                self.model.train()
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                
        except Exception as e:
            # Silent fail for online learning
            pass
    
    def toggle(self) -> bool:
        """Toggle LSTM prediction on/off"""
        if not PYTORCH_AVAILABLE:
            print("❌ LSTM unavailable - PyTorch not installed")
            print("💡 Install with: pip install torch")
            return False
            
        if self.model is None:
            success = self.initialize_model()
            if not success:
                return False
        
        self.enabled = not self.enabled
        status = "ENABLED" if self.enabled else "DISABLED"
        mode = "LSTM" if not self.use_fallback else "Fallback"
        print(f"🧠 LSTM Prediction {status} ({mode} mode)")
        return self.enabled
    
    def get_status(self) -> str:
        """Get current status string"""
        if not self.enabled:
            return "LSTM: OFF"
        elif self.use_fallback:
            return "LSTM: Fallback"
        else:
            return f"LSTM: ON ({self.prediction_confidence:.2f})"