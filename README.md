# 🎮🤖 AI Gesture Gaming Controller

Transform your webcam into an AI-powered game controller! Control any PC game using natural hand gestures and voice commands.

---

## ✨ Features

- 🖐 Real-time **gesture recognition** using MediaPipe CNN models
- 🧠 Integrated **voice commands** via OpenAI Whisper (coming soon)
- 🎮 **6 distinct hand gestures** for in-game movement and actions
- 🖥️ **Universal compatibility** with any PC game (no in-game modding required)
- ⚡ **Low-latency input simulation** (<100ms)
- 🧩 **Modular design** for easy customization and extension

---

## 🖐 Gesture Controls

| Gesture | Action | Game Input |
|--------|--------|------------|
| 🖐 Open Palm | Jump | Space |
| ✊ Closed Fist | Move Backward | S |
| ☝️ Index Point Up | Move Forward | W |
| ☝️ Index Point Left | Strafe Left | A |
| ☝️ Index Point Right | Strafe Right | D |
| 🤟 Rock Sign (Index + Pinky) | Attack | Left Click |

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

### 4. Launch a Game
1. Open any PC game (or Notepad for testing)
2. Place your hand in front of your webcam
3. Perform gestures to control the game!

---

## ⌨️ Hotkeys

- **P** - Pause/Resume system
- **Q** - Quit application
- **ESC** - Emergency stop (releases all held keys immediately)

---

## 🎯 System Requirements

- Python 3.8+
- Webcam (any USB or built-in)
- Windows, Mac, or Linux
- 4GB RAM minimum
- Modern CPU for real-time performance

---

## 📁 Project Structure

```
ai-gesture-gaming/
├── src/
│   ├── main.py              # Main entry point
│   ├── gesture_detector.py  # Hand detection logic
│   └── input_controller.py  # Keyboard and mouse input module
├── config/
│   └── settings.json        # Customizable config
├── tests/
│   └── test_camera.py       # Test your webcam
└── docs/                    # Documentation
```

---

## ✅ Tested Games

- 🧙‍♂️ The Witcher 3 – Movement & combat
- ⛏️ Minecraft – Navigation & interaction
- 🧾 Notepad – Basic testing for inputs
- 🌐 Browser Games – Plug-and-play control

---

## 🔮 Coming Soon 

- 🎤 Voice command integration via Whisper + LLMs
- 🎮 Game-specific control profiles (custom mappings)
- 📊 In-app performance analytics dashboard
- 🎛️ GUI-based control panel for gesture remapping
- ✋ Custom gesture training module (add your own signs!)

---

## 🛠 License

MIT License — Free to use, modify, and share.

---

## 🤝 Contributing

Pull requests welcome! This project showcases:

- 🧠 Computer Vision & AI (MediaPipe, Whisper)
- ⚙️ Real-time input processing
- 🎮 Game development interfaces
- 🧍‍♂️ Human-computer interaction

---

**Built with ❤️ to redefine the way we play.**
