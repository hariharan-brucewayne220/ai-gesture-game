# AI Gesture Gaming Controller

Transform your webcam into an AI-powered game controller! Control any PC game using natural hand gestures, voice commands, and get intelligent game assistance through our advanced RAG system.

## Core Features

### Gesture Control
- Real-time hand tracking using MediaPipe neural networks
- 6 distinct gestures for movement and actions
- Low-latency input (<100ms response time)
- Universal game compatibility (no modding required)

### Voice Commands  
- Advanced voice recognition with multiple engine support
- Custom game profiles with dynamic command mapping
- Noise suppression and ambient adjustment
- Gaming-optimized responses for fast-paced action

### RAG Game Assistant
- Interactive AI guide for game strategies and tips
- PDF guide integration - Upload any game guide PDF
- 20+ pre-loaded games with curated content
- Semantic search for precise answers
- Game-specific filtering for targeted assistance

## Complete Control System

### Gesture Controls
| Gesture | Action | Game Input |
|--------|--------|------------|
| Open Palm | Jump | Space |
| Closed Fist | Move Backward | S |
| Index Point Up | Move Forward | W |
| Index Point Left | Strafe Left | A |
| Index Point Right | Strafe Right | D |
| Rock Sign | Attack | Left Click |

You can also train your custom gestures too by pressing j key once your main.py starts running.

NOTE : every key press operation needs to be done by clicking on the camera feed window only then the custom key events such as adding gestures,voice would work

### Voice Commands
- "fire" → Left Click (attack)
- "jump" → Space (jump)
- "reload" → R key
- "inventory" → Tab key
- Custom game profiles with 50+ commands per game

### AI Game Assistant
- "How do I defeat Sigrun?" → Detailed boss strategies
- "What's the best armor?" → Equipment recommendations
- "Combat tips" → Advanced gameplay strategies
- Support for any game with PDF guide upload

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/ai-gesture-gaming.git
cd ai-gesture-gaming

# Create main environment for gesture/voice control
python -m venv mp_env
mp_env/Scripts/activate  # Windows
pip install -r requirements.txt

# Create separate RAG environment (prevents conflicts)
python -m venv rag_env
rag_env/Scripts/activate  # Windows
pip install chromadb>=0.4.0 langchain>=0.1.0 openai>=1.0.0 sentence-transformers>=2.2.0 PyPDF2>=3.0.0 pycryptodome>=3.0.0
```

### 2. Launch Gesture & Voice Control
```bash
mp_env/Scripts/python.exe src/main.py
```

### 3. Launch AI Game Assistant
```bash
rag_env/Scripts/python.exe query_rag.py
```

### 4. Test the System
1. **Gesture Control**: Place hand in front of webcam, perform gestures
2. **Voice Control**: Say "fire" or "jump" clearly into microphone  
3. **AI Assistant**: Ask "How do I defeat [boss name]?" for strategy help

## RAG Game Assistant

Our advanced RAG (Retrieval-Augmented Generation) system provides intelligent game assistance:

### Features
- PDF Guide Integration: Upload any game guide PDF for instant access
- Game-Specific Search: Filter by game for precise answers
- Query History: Track your previous questions
- System Statistics: View database contents and performance
- Semantic Search: Natural language query understanding

### Supported Games (Pre-loaded)
- **God of War** (with PDF guide) - Complete strategies and boss guides
- **Elden Ring** - Character builds and area guides
- **Dark Souls 3** - Boss strategies and combat tips
- **The Witcher 3** - Quest guides and combat strategies
- **Cyberpunk 2077** - Build guides and story choices
- **And 15+ more games** with curated content

### Example Queries
```
"How do I defeat Sigrun in God of War?"
"What's the best starting class in Elden Ring?"
"Where can I find the Master Sword in Zelda?"
"Combat strategies for Dark Souls bosses"
"Best armor combinations for late game"
```

## Project Structure

```
ai-gesture-gaming/
├── src/                          # Core system
│   ├── main.py                   # Main gesture/voice controller
│   ├── gesture_detector.py       # Hand tracking and recognition
│   ├── voice_controller.py       # Voice command processing
│   ├── input_controller.py       # Keyboard/mouse automation
│   └── game_controller_mapper.py # Game-specific mappings
├── game_rag/                     # RAG system
│   ├── query_engine.py          # Main RAG engine
│   ├── vector_store.py          # ChromaDB vector database
│   ├── document_loader.py       # PDF processing
│   └── chroma_db/               # Vector database storage
├── guides/                       # PDF game guides
│   └── god_of_war_guide.pdf     # Example integrated guide
├── config/                       # Configuration
│   └── settings.json            # System settings
├── custom_controller/            # Game profiles
│   ├── god_of_war_controls.json
│   └── last_of_us_2_controls.json
├── query_rag.py                 # Interactive RAG interface
└── requirements.txt             # Dependencies
```

## System Requirements

### Minimum Requirements
- Python 3.8+ (3.10+ recommended)
- 4GB RAM (8GB+ for RAG system)
- Webcam (any USB or built-in)
- Microphone (for voice commands)
- Windows 10+ (Mac/Linux compatible)

### Recommended Specifications
- 8GB+ RAM for smooth RAG operations
- SSD storage for faster database queries
- Good lighting for optimal gesture recognition
- Quality microphone for voice command accuracy

## Advanced Configuration

### Voice Command Customization
Create custom game profiles in `custom_controller/`:
```json
{
  "game_name": "Your Game",
  "voice_commands": {
    "attack": ["fire", "shoot", "attack"],
    "defend": ["block", "shield", "defend"],
    "special": ["ultimate", "special", "ability"]
  }
}
```

### RAG System Configuration
- **Add PDF guides**: Place PDFs in `guides/` folder
- **Custom embeddings**: Modify vector store settings
- **OpenAI integration**: Add API key for enhanced responses

### Gesture Training
Train custom gestures using the MLP trainer:
```bash
mp_env/Scripts/python.exe src/mlp_gesture_trainer.py
```

## Tested Games & Compatibility

### Fully Tested (Gesture + Voice + RAG)
- **God of War** - Complete integration with PDF guide
- **Elden Ring** - Combat and exploration
- **The Witcher 3** - Movement and combat
- **Dark Souls 3** - Precise combat control
- **Minecraft** - Building and navigation

### Voice Command Compatible
- **Call of Duty** series
- **Forza Horizon** series  
- **Skyrim/Fallout** series
- **Most FPS/RPG games**

### RAG Assistant Available
- **20+ games** with pre-loaded strategies
- **Any game** with uploaded PDF guide
- **General gaming** tips and strategies

## Troubleshooting

### Common Issues

**Gesture Recognition Issues:**
```bash
# Test camera
mp_env/Scripts/python.exe tests/test_camera.py

# Adjust lighting and hand position
# Ensure hand is 1-2 feet from camera
```

**Voice Command Issues:**
```bash
# Test microphone
mp_env/Scripts/python.exe debug_audio.py

# Fix microphone permissions
mp_env/Scripts/python.exe fix_microphone.py
```

**RAG System Issues:**
```bash
# Test RAG system
rag_env/Scripts/python.exe demo_rag_usage.py

# Check database status in query_rag.py → System Stats
```

### Performance Optimization
- Lower camera resolution for slower systems
- Adjust gesture sensitivity in settings
- Use separate environments to prevent conflicts
- Close unnecessary applications while gaming

## Recent Updates (2025)

### v2.0 - RAG Integration
- Complete RAG system with PDF guide support
- Interactive CLI interface for game assistance
- Vector database with 1,297+ documents
- Semantic search for natural language queries

### v1.5 - Voice Control
- Advanced voice recognition with multiple engines
- Game-specific profiles with custom commands
- Noise suppression and ambient adjustment
- Unicode compatibility fixes for Windows

### v1.0 - Gesture Control
- Real-time hand tracking with MediaPipe
- 6 distinct gestures with high accuracy
- Universal game compatibility
- Low-latency input system

## Future Roadmap

### Short Term (2025)
- GUI interface for easier configuration
- Mobile app for remote control
- Console support (Xbox/PlayStation)
- Auto-calibration for different lighting

### Long Term
- Custom AI models for gesture recognition
- Cloud RAG system with shared knowledge
- VR/AR integration for immersive control
- Real-time strategy suggestions during gameplay

## Performance Stats

- **Gesture Response**: <100ms average latency
- **Voice Recognition**: 95%+ accuracy in quiet environments  
- **RAG Queries**: <2 seconds average response time
- **Database Size**: 1,297 documents across 20+ games
- **Memory Usage**: ~500MB for full system

## Contributing

We welcome contributions! This project showcases:
- Computer Vision & MediaPipe integration
- Voice Recognition & NLP processing  
- RAG Systems & Vector Databases
- Real-time Input & Game Automation
- Human-Computer Interaction research

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

## License

MIT License - Free to use, modify, and distribute.

## Acknowledgments

- **MediaPipe** team for hand tracking technology
- **ChromaDB** for vector database capabilities
- **LangChain** for RAG system framework
- **OpenAI** for language model integration
- **Gaming community** for testing and feedback

---

**Built to revolutionize gaming control and assistance**

*Transform your gaming experience with AI-powered gesture control, voice commands, and intelligent game assistance. Welcome to the future of gaming interaction.*

## Note on Large Files

The following folders are not tracked by git and are excluded via .gitignore due to their size:
- vosk_models/
- whisper.cpp/

If you need these models, please download them separately as per the project instructions.

## Virtual Environment Setup

This project requires **two separate virtual environments**:

### 1. Main Environment (for running main.py)
```bash
# Create and activate the main environment
python -m venv gesture-env
gesture-env\Scripts\activate  # Windows
# or
source gesture-env/bin/activate  # Linux/Mac

# Install main dependencies
pip install -r requirements.txt
```

### 2. RAG Environment (for running query_rag.py)
```bash
# Create and activate the RAG environment
python -m venv rag_env
rag_env\Scripts\activate  # Windows
# or
source rag_env/bin/activate  # Linux/Mac

# Install RAG dependencies
pip install -r game_rag/requirements.txt
```

**Important**: Each environment has different dependencies and should be used separately:
- Use `gesture-env` for running `main.py` and gesture/voice control features
- Use `rag_env` for running `query_rag.py` and RAG (Retrieval-Augmented Generation) features
