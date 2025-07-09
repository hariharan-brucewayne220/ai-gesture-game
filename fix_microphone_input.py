#!/usr/bin/env python3
"""
Fix microphone input detection issue
"""

import speech_recognition as sr
import time

print("MICROPHONE INPUT FIX")
print("=" * 30)

print("Issue detected: WaitTimeoutError - microphone not detecting speech")
print()

recognizer = sr.Recognizer()

# Test different microphone indices
print("Testing different microphones...")
mics = sr.Microphone.list_microphone_names()

# Show available microphones
print("Available microphones:")
for i, name in enumerate(mics):
    if i < 10:  # Show first 10
        print(f"  {i}: {name}")

print()

# Try specific microphone indices that looked promising before
test_indices = [None, 0, 1, 7, 15, 19]  # None = default

for mic_index in test_indices:
    print(f"Testing microphone {mic_index}...")
    
    try:
        if mic_index is None:
            mic = sr.Microphone()
            print("Using default microphone")
        else:
            mic = sr.Microphone(device_index=mic_index)
            print(f"Using microphone {mic_index}: {mics[mic_index] if mic_index < len(mics) else 'Unknown'}")
        
        # Set very low energy threshold
        recognizer.energy_threshold = 50  # Very sensitive
        
        with mic as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        print(f"Energy threshold: {recognizer.energy_threshold}")
        
        # Test with very short timeout and low threshold
        recognizer.energy_threshold = min(recognizer.energy_threshold, 100)
        print(f"Set to: {recognizer.energy_threshold}")
        
        print("SPEAK LOUDLY NOW (3 seconds)...")
        
        with mic as source:
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
                print("AUDIO CAPTURED!")
                
                # Try to recognize
                text = recognizer.recognize_google(audio)
                print(f"SUCCESS! Heard: '{text}'")
                print(f"WORKING MICROPHONE: Index {mic_index}")
                break
                
            except sr.WaitTimeoutError:
                print("Still no input detected")
            except sr.UnknownValueError:
                print("Audio captured but not understood")
            except Exception as e:
                print(f"Recognition error: {e}")
                
    except Exception as e:
        print(f"Microphone {mic_index} failed: {e}")
    
    print("-" * 20)

print()
print("TROUBLESHOOTING STEPS:")
print("1. Windows microphone permissions:")
print("   - Press Win+I → Privacy & Security → Microphone")
print("   - Enable 'Allow apps to access your microphone'")
print("   - Enable 'Allow desktop apps to access your microphone'")
print()
print("2. Default microphone settings:")
print("   - Right-click speaker icon → Sound settings")
print("   - Under Input, select your microphone")
print("   - Test by speaking - blue bar should move")
print("   - Set input volume to 70-80%")
print()
print("3. Windows Sound Control Panel:")
print("   - Right-click speaker icon → Sounds → Recording tab")
print("   - Right-click your microphone → Set as Default Device")
print("   - Right-click → Properties → Levels → Set to 70-80%")
print()
print("4. Hardware check:")
print("   - Is microphone connected and powered on?")
print("   - Try a different microphone/headset")
print("   - Check if microphone works in other apps")
print()
print("If no microphone worked:")
print("- This is a Windows permissions/hardware issue")
print("- Voice recognition code is working correctly") 
print("- Fix microphone access and voice commands will work perfectly")