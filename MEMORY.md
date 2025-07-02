# AI Gesture Gaming System - Development Memory

## Project Overview
Built a comprehensive AI-first multimodal gaming control system that combines gesture recognition, head tracking, voice commands, and hand camera control for immersive gaming experiences.

## Key Features Implemented

### 1. Gesture Recognition System (`gesture_detector.py`)
- **Hand gesture detection** using MediaPipe
- **Directional gestures** for movement:
  - Open palm → Jump (Space)
  - Closed fist → Backward (S)
  - Index point left → Strafe left (A)
  - Index point right → Strafe right (D) 
  - Index point up → Forward (W)
  - Rock sign (index + pinky) → Attack (Left Click)
- **Smart stabilization** for action vs movement gestures
- **Hand camera control** using velocity-based tracking (wrist position)

### 2. Head Tracking System (`head_tracker.py`)
- **MediaPipe Face Mesh** for head pose estimation
- **Camera control** via head movement (disabled by default to avoid neck pain)
- **Reference position** calibration system
- **Sensitivity controls** and deadzone configuration
- **pydirectinput integration** for AAA game compatibility (replaced pynput)

### 3. Voice Control System (`voice_controller.py`)
- **Real-time speech recognition** using Google Web Speech API
- **Fuzzy matching** for intent recognition (70% similarity threshold)
- **Last of Us control mappings** hardcoded for testing:
  - "flashlight" → T key
  - "listen" → Q key (Listen Mode)
  - "heal" → 7 key (Health Kit)
  - "crouch" → C key
  - "sprint" → Shift key
  - "shield"/"defend" → E key
  - "escape"/"flee" → F key
  - Plus 15+ more voice commands
- **Audio feedback** with text-to-speech confirmations
- **High-frequency listening** optimized for gaming

### 4. Input Controller (`input_controller.py`)
- **Unified input management** for all control methods
- **pydirectinput integration** for better AAA game compatibility
- **Voice command execution** with key mapping
- **Camera movement handling** for both head and hand tracking
- **Cooldown systems** to prevent input spam

### 5. Main System (`main.py`)
- **Multimodal integration** - gestures + voice + head + hand camera
- **Real-time processing** of all input streams
- **Comprehensive keyboard controls**:
  - 'g' - Toggle hand camera control
  - 'h' - Toggle head tracking
  - 'v' - Toggle voice recognition
  - 'x' - Reset hand camera reference
  - '+/-' - Adjust hand camera sensitivity
  - 'c' - List voice commands
  - 'b' - Toggle audio feedback
  - 'p' - Pause/Resume
  - 'q' - Quit

## Technical Decisions Made

### 1. AAA Game Compatibility
- **Replaced pynput with pydirectinput** for mouse movement to bypass game input protection
- **Optimized for performance** during intensive gameplay
- **Addressed Last of Us performance issues** (120GB games vs 70GB games)

### 2. User Comfort & Safety
- **Head tracking OFF by default** to prevent neck strain during long gaming sessions
- **Hand camera control** as comfortable alternative
- **Velocity-based hand tracking** to prevent runaway camera movement
- **Adjustable sensitivity** for personal preference

### 3. Voice Recognition Strategy
- **Chose speech_recognition over Whisper** for real-time gaming (lower latency)
- **Hardcoded Last of Us controls** instead of dynamic config files for MVP testing
- **Fuzzy matching** for natural language flexibility
- **High-frequency listening loop** for responsive gaming commands

### 4. Camera Control Evolution
- **Started with head tracking** (MediaPipe Face Mesh)
- **Added hand camera control** as neck-pain-free alternative
- **Implemented velocity-based tracking** to fix continuous movement issue
- **Combined both systems** for user choice

## Key Technical Fixes

### 1. Voice Recognition Frequency Issue
**Problem**: Voice commands only recognized occasionally
**Solution**: 
- Reduced recognition timeout to 0.1s
- Optimized listening loop for continuous operation
- Added debug output for troubleshooting

### 2. Hand Camera Sensitivity Issue
**Problem**: Slight hand movements caused huge camera movements
**Solution**:
- Reduced sensitivity from 3.0x to 0.5x initially
- Implemented velocity-based tracking instead of position-based
- Added real-time sensitivity adjustment (+/- keys)

### 3. Continuous Camera Movement Issue
**Problem**: Camera kept moving when hand held in position
**Solution**:
- **Switched from position-based to velocity-based tracking**
- Camera now only moves when hand is actively moving
- Stops immediately when hand stops moving

### 4. AAA Game Input Issues
**Problem**: Mouse movement not working in games like Last of Us
**Solution**:
- Replaced pynput with pydirectinput for better game compatibility
- Configured DirectInput for lower-level access
- Added proper error handling and cooldowns

## File Structure
```
src/
├── main.py                 # Main system integration
├── gesture_detector.py     # Hand gesture + hand camera control
├── head_tracker.py         # Head tracking for camera movement  
├── voice_controller.py     # Speech recognition + voice commands
├── input_controller.py     # Unified input management
└── __init__.py

requirements.txt            # All dependencies
test_voice.py              # Voice recognition testing utility
MEMORY.md                  # This development log
```

## Dependencies Added
```
opencv-python==4.8.1.78
mediapipe==0.10.7
numpy==1.24.3
pynput==1.7.6
pyautogui==0.9.54
Pillow==10.0.1
pydirectinput==1.0.4
speechrecognition==3.10.0
pyaudio==0.2.11
pyttsx3==2.90
fuzzywuzzy==0.18.0
```

## Future AI Integration Possibilities Discussed
1. **Natural Language Processing** - Replace hardcoded intents with LLM understanding
2. **Computer Vision AI** - YOLOv8 for game state analysis
3. **Local AI Integration** - Ollama for offline LLM processing
4. **Predictive Gaming AI** - Learn user patterns and adapt controls
5. **Multimodal AI Fusion** - Intelligent combination of all input methods

## Current System Status
- ✅ **Gesture control** - Working, optimized for movement and actions
- ✅ **Voice commands** - Working with Last of Us control set
- ✅ **Head tracking** - Working but disabled by default (neck pain prevention)
- ✅ **Hand camera control** - Working with velocity-based tracking
- ✅ **AAA game compatibility** - Enhanced with pydirectinput
- ✅ **Multimodal integration** - All systems working together
- 🔄 **Performance optimization** - Ongoing for heavy games

## Testing Status
- Voice recognition frequency: ✅ Fixed
- Hand camera sensitivity: ✅ Fixed  
- Continuous movement: ✅ Fixed
- AAA game input: ✅ Enhanced
- Integration testing: ✅ Complete

## Next Steps Identified
1. Add push-to-talk mode for noisy gaming environments
2. Implement offline voice recognition (Whisper/Vosk) for privacy
3. Add game-specific configuration system
4. Integrate LLM for natural language command understanding
5. Add performance monitoring for resource-intensive games