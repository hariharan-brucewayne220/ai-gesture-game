#!/usr/bin/env python3
"""
Run the original system bypassing encoding issues
"""

import sys
import os

# Fix encoding issues by monkey patching print
original_print = print

def safe_print(*args, **kwargs):
    try:
        original_print(*args, **kwargs)
    except UnicodeEncodeError:
        # Replace problematic characters
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                # Replace emojis with simple text
                safe_arg = arg.replace('🤖', 'AI').replace('🎮', 'Game').replace('🎤', 'Voice').replace('✅', 'OK').replace('❌', 'ERROR').replace('⚠️', 'WARNING').replace('💡', 'TIP').replace('🔧', 'TOOL').replace('🎯', 'TARGET').replace('🙌', 'HANDS').replace('👋', 'HAND').replace('🎧', 'AUDIO').replace('🔊', 'SPEAKER').replace('🔇', 'MUTE').replace('📋', 'LIST').replace('⚡', 'FAST').replace('🚀', 'READY').replace('🛑', 'STOP').replace('🔍', 'SEARCH').replace('📸', 'CAPTURE').replace('🌊', 'SMOOTH').replace('📹', 'CAMERA').replace('👁️', 'EYE').replace('🏃', 'RUN').replace('🚶', 'WALK').replace('🤟', 'ROCK').replace('✊', 'FIST').replace('🖐', 'PALM').replace('☝️', 'POINT').replace('🤚', 'HAND')
                # Remove any remaining emojis
                safe_arg = ''.join(c for c in safe_arg if ord(c) < 65536)
                safe_args.append(safe_arg)
            else:
                safe_args.append(arg)
        original_print(*safe_args, **kwargs)

# Override print globally
import builtins
builtins.print = safe_print

# Now run the original system
sys.path.append('src')

try:
    from main import GestureGamingSystem
    
    print("Starting AI Gesture Gaming System...")
    print("Bypassing API key prompts to test voice...")
    
    # Create system without API keys to avoid prompts
    system = GestureGamingSystem(groq_api_key=None, google_ai_key=None, openai_api_key=None)
    
    print("System initialized!")
    print("Testing voice controller...")
    
    # Test if voice controller is working
    if system.voice_controller.enabled:
        print("Voice controller is ENABLED")
        
        # Test a simple voice command recognition
        test_commands = ["test", "hello", "fire", "jump"]
        for cmd in test_commands:
            result = system.voice_controller.execute_voice_command(cmd)
            if result:
                print(f"Command '{cmd}' -> {result['name']} ({result['key']})")
            else:
                print(f"Command '{cmd}' -> No match")
    else:
        print("Voice controller is DISABLED")
    
    print("\nVoice system status:")
    print(f"- Real-time mode: {system.voice_controller.real_time_mode}")
    print(f"- Gaming mode: {system.voice_controller.gaming_mode}")
    print(f"- Noise suppression: {system.voice_controller.noise_suppression}")
    print(f"- Energy threshold: {system.voice_controller.recognizer.energy_threshold}")
    
    print("\nTo test with camera, run: mp_env/Scripts/python.exe run_original.py --full")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("Starting full system with camera...")
        system.start()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()