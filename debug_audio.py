#!/usr/bin/env python3
"""
Debug script to test audio/microphone setup
"""

import sys
import traceback

print("🔍 Testing Audio/Voice Recognition Setup...")
print("=" * 50)

# Test 1: Check if speech_recognition is installed
try:
    import speech_recognition as sr
    print(f"✅ speech_recognition installed: v{sr.__version__}")
except ImportError as e:
    print(f"❌ speech_recognition not installed: {e}")
    sys.exit(1)

# Test 2: Check if pyaudio is available
try:
    import pyaudio
    print(f"✅ pyaudio available")
except ImportError as e:
    print(f"⚠️ pyaudio not available: {e}")
    print("💡 Try: pip install pyaudio")

# Test 3: List available microphones
try:
    print("\n🎤 Available Microphones:")
    mic_list = sr.Microphone.list_microphone_names()
    if not mic_list:
        print("❌ No microphones found!")
    else:
        for i, name in enumerate(mic_list):
            print(f"   {i}: {name}")
            
    # Test default microphone
    print(f"\n🎯 Default microphone index: {sr.Microphone().device_index}")
    
except Exception as e:
    print(f"❌ Error listing microphones: {e}")
    traceback.print_exc()

# Test 4: Test microphone access
try:
    print("\n🔊 Testing microphone access...")
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    print("📡 Adjusting for ambient noise...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print(f"✅ Energy threshold set to: {recognizer.energy_threshold}")
    
except Exception as e:
    print(f"❌ Microphone access failed: {e}")
    traceback.print_exc()

# Test 5: Quick recognition test
try:
    print("\n🎙️ Quick speech test (say something for 2 seconds)...")
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        print("Listening...")
        audio = recognizer.listen(source, timeout=2, phrase_time_limit=2)
        print("Processing...")
        
    # Try Google Web Speech
    try:
        text = recognizer.recognize_google(audio)
        print(f"✅ Google Speech recognized: '{text}'")
    except sr.UnknownValueError:
        print("⚠️ Google Speech could not understand audio")
    except sr.RequestError as e:
        print(f"❌ Google Speech error: {e}")
        
except Exception as e:
    print(f"❌ Speech test failed: {e}")
    traceback.print_exc()

# Test 6: Check system dependencies
print("\n🔧 System Dependencies:")
try:
    import pyttsx3
    print("✅ pyttsx3 (text-to-speech) available")
except ImportError:
    print("⚠️ pyttsx3 not available")

try:
    import fuzzywuzzy
    print("✅ fuzzywuzzy (fuzzy matching) available")
except ImportError:
    print("⚠️ fuzzywuzzy not available")

print("\n" + "=" * 50)
print("🏁 Audio debug complete!")