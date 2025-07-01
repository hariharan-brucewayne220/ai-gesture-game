# 🚀 Installation Guide - AI Gesture Gaming Controller

## 📋 Quick Start Instructions

### **Step 1: Install Python Dependencies**

You need to install these packages. Run ONE of these methods:

#### **Method A: Using pip (Recommended)**
```bash
pip install opencv-python mediapipe numpy pynput pyautogui Pillow
```

#### **Method B: Using conda (if you have Anaconda/Miniconda)**
```bash
conda install -c conda-forge opencv mediapipe numpy
pip install pynput pyautogui Pillow
```

#### **Method C: Using WSL/Linux package manager**
```bash
# Install Python packages
sudo apt update
sudo apt install python3-pip python3-opencv
pip3 install mediapipe numpy pynput pyautogui Pillow
```

### **Step 2: Test Installation**

Create a simple test file to verify packages work:

```python
# test_installation.py
try:
    import cv2
    print("✅ OpenCV installed")
except ImportError:
    print("❌ OpenCV missing - run: pip install opencv-python")

try:
    import mediapipe
    print("✅ MediaPipe installed")
except ImportError:
    print("❌ MediaPipe missing - run: pip install mediapipe")

try:
    import numpy
    print("✅ NumPy installed")
except ImportError:
    print("❌ NumPy missing - run: pip install numpy")

try:
    import pynput
    print("✅ PyInput installed")
except ImportError:
    print("❌ PyInput missing - run: pip install pynput")

print("\n🎮 If all packages show ✅, you're ready to run the gesture controller!")
```

Save this as `test_installation.py` and run:
```bash
python test_installation.py
```

### **Step 3: Test Camera**

Run this to verify your camera works:
```bash
cd ai-gesture-gaming
python tests/test_camera.py
```

### **Step 4: Run Gesture Controller**

Once packages are installed:
```bash
cd ai-gesture-gaming/src
python main.py
```

## 🔧 **Troubleshooting**

### **Issue: "pip command not found"**
```bash
# Try these alternatives:
python -m pip install [package]
python3 -m pip install [package]
py -m pip install [package]  # Windows
```

### **Issue: "Permission denied"**
```bash
# Add --user flag:
pip install --user opencv-python mediapipe numpy pynput pyautogui
```

### **Issue: "Camera not found"**
- Check if other apps are using camera
- Try different camera indices (0, 1, 2)
- Restart computer if needed

### **Issue: "ModuleNotFoundError"**
```bash
# Install missing packages one by one:
pip install opencv-python
pip install mediapipe  
pip install numpy
pip install pynput
pip install pyautogui
```

## 🎯 **Quick Browser Game Test**

1. **Install packages** (above)
2. **Open browser game**: https://krunker.io or https://slither.io
3. **Run gesture controller**: `python src/main.py`
4. **Test gestures**:
   - 🖐 Open palm = W (forward)
   - ✊ Fist = S (backward)  
   - ✌️ Peace = A (left)
   - 🤟 Rock = D (right)

---

**Follow these steps and let me know which step you get stuck on!** 🚀