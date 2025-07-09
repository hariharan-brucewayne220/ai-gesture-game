#!/usr/bin/env python3
"""
Microphone fix and diagnostic tool
"""

import speech_recognition as sr
import time

print("MICROPHONE DIAGNOSTIC AND FIX")
print("=" * 40)

# Step 1: Try different microphones
recognizer = sr.Recognizer()

# Lower the energy threshold significantly
recognizer.energy_threshold = 50  # Much lower than default 250
print(f"Set energy threshold to: {recognizer.energy_threshold}")

# Try multiple microphone indices
mic_indices_to_try = [None, 0, 1, 7, 15, 19]  # None = default, then specific indices

for mic_index in mic_indices_to_try:
    print(f"\nTesting microphone index: {mic_index}")
    
    try:
        if mic_index is None:
            mic = sr.Microphone()
            print("Using default microphone")
        else:
            mic = sr.Microphone(device_index=mic_index)
            print(f"Using microphone {mic_index}")
        
        # Quick ambient noise adjustment
        with mic as source:
            print("Adjusting for ambient noise (1 second)...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print(f"Energy threshold after adjustment: {recognizer.energy_threshold}")
        
        # Try to listen with very low thresholds
        recognizer.energy_threshold = min(recognizer.energy_threshold, 100)
        print(f"Final energy threshold: {recognizer.energy_threshold}")
        
        print("SPEAK NOW for 5 seconds (say anything loudly)...")
        
        with mic as source:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                print("Audio captured! Trying to recognize...")
                
                # Try recognition
                text = recognizer.recognize_google(audio)
                print(f"SUCCESS! Recognized: '{text}'")
                print(f"WORKING MICROPHONE INDEX: {mic_index}")
                break
                
            except sr.WaitTimeoutError:
                print("TIMEOUT: No speech detected")
            except sr.UnknownValueError:
                print("UNCLEAR: Audio detected but not understood")
            except Exception as e:
                print(f"Recognition error: {e}")
                
    except Exception as e:
        print(f"Microphone {mic_index} failed: {e}")

print("\n" + "=" * 40)
print("SOLUTIONS TO TRY:")
print("1. Check Windows microphone permissions:")
print("   Settings > Privacy > Microphone > Allow desktop apps")
print("2. Set default microphone:")
print("   Right-click speaker icon > Sound settings > Input")
print("3. Test microphone in Windows:")
print("   Settings > System > Sound > Test your microphone")
print("4. Try speaking MUCH louder")
print("5. Move closer to microphone")
print("6. Use a USB headset if available")

# Final test with manual threshold setting
print("\nFINAL TEST - Manual low threshold:")
recognizer.energy_threshold = 30  # Very low
try:
    mic = sr.Microphone()
    print("Speak VERY LOUDLY for 3 seconds...")
    with mic as source:
        audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
    text = recognizer.recognize_google(audio)
    print(f"FINAL TEST SUCCESS: '{text}'")
except Exception as e:
    print(f"Final test failed: {e}")