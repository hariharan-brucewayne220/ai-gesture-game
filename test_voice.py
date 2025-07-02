import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from voice_controller import VoiceController
import time

def test_voice_recognition():
    print("🎤 Voice Recognition Test")
    print("=" * 40)
    
    # Initialize voice controller
    vc = VoiceController()
    
    # Override the speech recognition handler to just print
    def test_handler(spoken_text):
        print(f"✅ HEARD: '{spoken_text}'")
        
        # Test the command matching
        result = vc.execute_voice_command(spoken_text)
        if result:
            print(f"🎯 MATCHED: {result['name']} -> {result['key']}")
        else:
            print("❌ NO MATCH FOUND")
        print("-" * 40)
    
    # Connect our test handler
    vc._on_speech_recognized = test_handler
    
    print("🎧 Starting voice recognition...")
    print("Say 'flashlight', 'heal', 'listen', etc.")
    print("Press Ctrl+C to stop")
    print("-" * 40)
    
    # Start listening
    vc.start_listening()
    
    try:
        # Keep the test running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping test...")
        vc.cleanup()
        print("✅ Test complete!")

if __name__ == "__main__":
    test_voice_recognition()