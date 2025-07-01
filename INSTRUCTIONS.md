# 🎮 **EXACT INSTRUCTIONS TO RUN YOUR GESTURE CONTROLLER**

## 🚨 **You Need to Install Python Packages First**

Your system has Python but not pip (package manager). Here's how to fix it:

### **Option 1: Install via apt (Recommended for WSL)**

Open your terminal and run these commands **one by one**:

```bash
# Install pip and required system packages
sudo apt update
sudo apt install python3-pip python3-dev python3-opencv

# Install Python packages
pip3 install mediapipe numpy pynput pyautogui Pillow

# Verify installation
python3 test_installation.py
```

### **Option 2: Use Windows Python (Easier)**

If you have Python installed on Windows:

1. **Open Windows Command Prompt** (not WSL)
2. **Navigate to project folder**:
   ```cmd
   cd D:\claude-projects\ai-gesture-gaming
   ```
3. **Install packages**:
   ```cmd
   pip install opencv-python mediapipe numpy pynput pyautogui Pillow
   ```
4. **Run the gesture controller**:
   ```cmd
   python src/main.py
   ```

### **Option 3: Download Anaconda (Most Reliable)**

1. **Download Anaconda**: https://www.anaconda.com/download
2. **Install it** (includes Python + pip + common packages)
3. **Open Anaconda Prompt**
4. **Install packages**:
   ```bash
   conda install -c conda-forge opencv mediapipe numpy
   pip install pynput pyautogui
   ```

## 🎯 **Quick Test Sequence (Once Packages Are Installed)**

### **Step 1: Test Installation**
```bash
python test_installation.py
# Should show all ✅ green checkmarks
```

### **Step 2: Test Camera**
```bash
python tests/test_camera.py
# Should open camera window - press 'q' to quit
```

### **Step 3: Run Gesture Controller**
```bash
cd src
python main.py
# Should open gesture detection window
```

### **Step 4: Test with Browser Game**
1. **Open browser** → Go to https://slither.io or https://krunker.io
2. **Run gesture controller**: `python src/main.py`
3. **Make gestures** to control the game:
   - 🖐 **Open Palm** = Move forward
   - ✊ **Fist** = Move backward
   - ✌️ **Peace Sign** = Turn left
   - 🤟 **Rock Sign** = Turn right

## 🔧 **Troubleshooting**

### **"sudo: a password is required"**
- Enter your WSL/Linux password
- If you don't know it, use Windows Python instead (Option 2)

### **"Permission denied"**
```bash
# Add --user flag to install for current user only
pip3 install --user mediapipe numpy pynput pyautogui Pillow
```

### **"Camera not found"**
- Make sure no other apps are using your camera
- Try restarting your computer
- Check Windows Camera app works first

### **"ModuleNotFoundError"**
- Packages didn't install correctly
- Try installing one by one:
  ```bash
  pip install numpy
  pip install opencv-python
  pip install mediapipe
  # etc.
  ```

## 🎮 **Expected Results**

When working correctly:
1. **Camera window opens** showing your hand
2. **Green skeleton** drawn on your hand  
3. **Gesture name** appears on screen (e.g., "open_palm")
4. **Game responds** to your gestures

## 📞 **What to Tell Me**

Let me know:
1. **Which option you chose** (apt/Windows Python/Anaconda)
2. **Any error messages** you see
3. **Which step you got stuck on**
4. **What happens when you run** `python test_installation.py`

---

## 🚀 **Quick Start (If You Have Windows Python)**

If you have Python on Windows, this is the fastest path:

1. **Open Windows Command Prompt**
2. **Run these commands**:
   ```cmd
   cd D:\claude-projects\ai-gesture-gaming
   pip install opencv-python mediapipe numpy pynput pyautogui Pillow
   python test_installation.py
   python src/main.py
   ```

**That's it!** Your gesture controller should be running! 🎮🤖