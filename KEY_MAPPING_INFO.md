# 🔧 Key Mapping Information

## 📍 WHERE KEY MAPPING HAPPENS

### **1. Gesture Detection** 
📁 **File**: `src/gesture_detector.py`  
📍 **Function**: `classify_gesture()` (lines 83-132)
- Detects hand gestures from camera feed
- Returns gesture names like "one_finger", "two_fingers", etc.

### **2. Gesture → Action Mapping**
📁 **File**: `src/input_controller.py`  
📍 **Dictionary**: `self.gesture_map` (lines 16-22)
```python
self.gesture_map = {
    "open_palm": "w",      # Open Palm → Forward (W)
    "closed_fist": "s",    # Closed Fist → Backward (S)  
    "peace_sign": "a",     # Peace Sign → Strafe Left (A)
    "rock_sign": "d",      # Rock Sign → Strafe Right (D)
    "thumbs_up": Key.space, # Thumbs Up → Jump (Space)
    "index_point": "attack" # Index Point → Attack (Mouse Click)
}
```

### **3. Action Execution**
📁 **File**: `src/input_controller.py`  
📍 **Function**: `send_action()` (lines 34-58)
- Maps gestures to keyboard/mouse actions
- **Line 40**: `action = self.gesture_map.get(gesture)` ← KEY MAPPING LINE
- **Line 56**: `self.press_key(action, gesture)` ← ACTUAL KEY PRESS
- **Line 71**: `self.keyboard_controller.press(key)` ← PHYSICAL KEY PRESS

---

## 🖐 MediaPipe Hand Gestures Available

### **Built-in MediaPipe Hand Landmarks**
MediaPipe provides 21 hand landmarks but NO gesture recognition. We build our own:

```
Landmark Points (0-20):
0: WRIST
1-4: THUMB (base to tip)
5-8: INDEX (base to tip)  
9-12: MIDDLE (base to tip)
13-16: RING (base to tip)
17-20: PINKY (base to tip)
```

### **Our Custom Gesture System**
📁 **File**: `src/gesture_detector.py` (lines 97-130)

We detect:
- **Extended fingers count** (how many fingers are up)
- **Thumb position** (extended or not)
- **Specific finger combinations**

### **Current Gesture Set:**
1. **"open_palm"** - All fingers extended (4+ fingers up)
2. **"closed_fist"** - No fingers up, small distance to wrist
3. **"thumbs_up"** - Only thumb extended
4. **"index_point"** - Only index finger extended
5. **"peace_sign"** - Index + middle fingers extended
6. **"rock_sign"** - Index + pinky fingers extended

### **Unused Gestures We Could Add:**
- **"ok_sign"** - Thumb + index in circle
- **"finger_counting"** - Numerical finger counting (1-5)
- **"pointing"** - Directional pointing detection
- **"middle_finger"** - Only middle finger up

---

## 🎯 Flow Summary

```
Camera → MediaPipe → Feature Extraction → Gesture Classification → Action Mapping → Key Press
   ↓         ↓              ↓                    ↓                     ↓            ↓
Webcam → Hand Detection → Finger Pattern → "rock_sign" → "d" key → Physical D Press
```

### **Key Line of Code:**
📍 **input_controller.py:40** 
```python
action = self.gesture_map.get(gesture)  # ← THIS IS WHERE GESTURE MAPS TO KEY
```

📍 **input_controller.py:71**
```python
self.keyboard_controller.press(key)     # ← THIS IS WHERE KEY ACTUALLY GETS PRESSED
```

---

## 🔧 To Modify Mappings:

**Change the dictionary in lines 16-22 of `input_controller.py`:**
```python
"open_palm": "w",      # Change "w" to any key
"rock_sign": "d",      # Change "d" to any key  
# etc.
```