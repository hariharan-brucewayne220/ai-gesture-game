#!/usr/bin/env python3
"""
Clean startup script bypassing all unicode issues
"""

import sys
import os

# Completely replace print to avoid any unicode issues
def clean_print(*args, **kwargs):
    try:
        # Convert all args to ASCII-safe strings
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                # Remove all non-ASCII characters
                safe_str = ''.join(c if ord(c) < 128 else '?' for c in str(arg))
                safe_args.append(safe_str)
            else:
                safe_args.append(str(arg))
        
        # Use simple print
        import builtins
        builtins.__dict__['original_print'](*safe_args, **kwargs)
    except:
        # Last resort - just print basic info
        import builtins
        builtins.__dict__['original_print']("Audio system running...")

# Save original print and replace
import builtins
builtins.__dict__['original_print'] = builtins.print
builtins.print = clean_print

# Now import the system
sys.path.append('src')

try:
    print("=== AI GESTURE GAMING SYSTEM ===")
    print("Starting voice recognition...")
    
    from voice_controller import VoiceController
    from input_controller import InputController
    
    # Create voice controller
    voice = VoiceController()
    controller = InputController()
    
    print("Voice system initialized!")
    print("Available commands: fire, jump, run, crouch, heal, etc.")
    print("Press Ctrl+C to stop")
    
    # Create a simple callback to handle voice commands
    def handle_voice(spoken_text):
        print(f"You said: '{spoken_text}'")
        result = voice.execute_voice_command(spoken_text)
        if result:
            print(f"Action: {result['name']} -> {result['key']}")
            # Here you would normally send to input controller
            # controller.send_voice_action(result)
        else:
            print("No matching command found")
    
    # Set the callback
    voice._on_speech_recognized = handle_voice
    
    # Start listening
    print("Starting voice recognition...")
    voice.start_listening()
    
    print("Voice recognition is now ACTIVE!")
    print("Try saying: 'fire', 'jump', 'run', 'heal', 'flashlight'")
    
    # Keep running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        voice.stop_listening()
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()