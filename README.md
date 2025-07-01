# 🎮🤖 AI Gesture Gaming Controller

Transform your webcam into an AI-powered game controller!  
Control any PC game using natural hand gestures and voice commands.

---

## ✨ Features

- 🔍 **Real-time gesture recognition** using MediaPipe AI  
- 🖐 **6 distinct hand gestures** mapped to game inputs  
- 🎮 **Universal game compatibility** — works with any PC game  
- ⚡ **Low latency input simulation** (<100ms)  
- 🧩 **Modular design** for easy customization  

---

## 🖐 Gesture Controls

| Gesture         | Action         | Game Input  |
|----------------|----------------|-------------|
| 🖐 Open Palm    | Forward        | `W`         |
| ✊ Closed Fist  | Backward       | `S`         |
| ✌️ Peace Sign   | Strafe Left    | `A`         |
| 🤟 Rock Sign    | Strafe Right   | `D`         |
| 👍 Thumbs Up    | Jump           | `Space`     |
| ☝️ Index Point | Attack         | `Left Click`|

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Camera

```bash
python tests/test_camera.py
```

### 3. Run Gesture Controller

```bash
cd src
python main.py
```

### 4. Test with Game

- Launch any PC game (or Notepad for basic testing)  
- Position your hand in front of the webcam  
- Use gestures to control the game!

---

## 🔧 Runtime Controls

- Press **`p`** → Pause/Resume the system  
- Press **`q`** → Quit the application  
- Press **`ESC`** → Emergency stop (release all keys)

---

## 🖥 System Requirements

- Python **3.8+**  
- A functional **webcam** (USB or built-in)  
- Compatible with **Windows / Mac / Linux**  
- Minimum **4 GB RAM**  
- Modern **CPU** for real-time performance

---

## 📁 Project Structure

```
ai-gesture-gaming/
├── src/
│   ├── main.py              # Main app
│   ├── gesture_detector.py  # Gesture recognition logic
│   └── input_controller.py  # Keyboard/mouse event simulator
├── config/
│   └── settings.json        # Custom settings
├── tests/
│   └── test_camera.py       # Camera check utility
└── docs/                    # Project documentation
```

---

## ✅ Game Compatibility

Tested and working with:

- **Witcher 3** — Combat + movement  
- **Minecraft** — Building, movement, interactions  
- **Notepad/Text editors** — Input emulation testing  
- **Browser games** — Universal gesture control

---

## 🔄 Coming Soon (Day 2+)

- 🎤 Voice command integration  
- 🎯 Game-specific control profiles  
- 📊 Performance statistics  
- 🎨 Visual configuration GUI  
- 🧠 Custom gesture training module

---

## 📝 License

MIT License — Free to use, modify, and share.

---

## 🤝 Contributing

Contributions welcome!  
This is a **portfolio project** showcasing:

- Computer Vision & AI  
- Real-time input simulation  
- Game & input device hacking  
- Human-computer interaction research

---

**Built with ❤️ for the future of gaming**
