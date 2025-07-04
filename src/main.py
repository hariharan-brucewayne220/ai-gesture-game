import cv2
import time
import threading
from gesture_detector import GestureDetector
from input_controller import InputController
from head_tracker import HeadTracker
from voice_controller import VoiceController

class GestureGamingSystem:
    def __init__(self, groq_api_key=None, google_ai_key=None):
        self.detector = GestureDetector(google_ai_key=google_ai_key)
        self.controller = InputController()
        self.head_tracker = HeadTracker(enabled=False)  # OFF by default to avoid neck pain
        self.voice_controller = VoiceController(groq_api_key=groq_api_key)
        self.running = False
        self.paused = False
        
        # Connect components
        self.controller.set_gesture_detector(self.detector)  # For voice hand camera control
        self.detector.parent_controller = self.controller  # For dynamic gesture mapping
        self.voice_controller._on_speech_recognized = self.handle_voice_command
        
    def start(self):
        """Start the gesture gaming system"""
        print("AI Gesture Gaming System Starting...")
        print("📋 Gesture Controls:")
        print("   🖐 Open Palm → Jump (Space)")
        print("   ✊ Closed Fist → Backward (S)")
        print("   ☝️ Index Point Left → Strafe Left (A)")
        print("   ☝️ Index Point Right → Strafe Right (D)")
        print("   ☝️ Index Point Up → Forward (W)")
        print("   🤟 Rock Sign (Index + Pinky) → Attack (Left Click)")
        print("\n🎯 Camera Controls:")
        print("   Say 'flame on/off' to control hand camera")
        print("   '+' - Increase hand camera sensitivity")
        print("   '-' - Decrease hand camera sensitivity")
        print("\nMLP Gesture System:")
        print("   'k' - Start full calibration (WSAD + default gestures)")
        print("   'j' - Add custom gesture")
        print("   'l' - List calibrated gestures")
        print("   'z' - Clear all data and start fresh")
        print("\n🎤 Voice Commands:")
        print("   Say 'flashlight', 'heal', 'listen', etc.")
        print("   'v' - Toggle voice recognition")
        print("\n🔧 Controls:")
        print("   'p' - Pause/Resume")
        print("   'q' - Quit")
        print("\n🚀 System Ready! Show your hand to the camera...")
        
        
        # Start voice recognition
        self.voice_controller.start_listening()
        
        self.running = True
        
        try:
            while self.running:
                # Process frame (now returns hand camera movement too)
                frame, gesture, confidence, hand_mouse_dx, hand_mouse_dy = self.detector.process_frame()
                
                # Process head tracking
                head_mouse_dx, head_mouse_dy = 0, 0
                if frame is not None:
                    frame, head_mouse_dx, head_mouse_dy = self.head_tracker.process_head_movement(frame)
                
                # Combine camera movements (head tracking + hand camera)
                total_mouse_dx = head_mouse_dx + hand_mouse_dx
                total_mouse_dy = head_mouse_dy + hand_mouse_dy
                
                if frame is not None:
                    # Add system status to frame
                    status = "PAUSED" if self.paused else "ACTIVE"
                    color = (0, 255, 255) if self.paused else (0, 255, 0)
                    cv2.putText(frame, f"Status: {status}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Add instructions
                    cv2.putText(frame, "Press 'p' to pause, 'q' to quit", (10, frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Show frame
                    cv2.imshow("AI Gesture Gaming Controller", frame)
                    
                    # Send input if not paused
                    if not self.paused:
                        self.controller.send_action(gesture, confidence)
                        # Send camera movement (combined head tracking + hand camera)
                        if total_mouse_dx != 0 or total_mouse_dy != 0:
                            self.controller.send_camera_movement(total_mouse_dx, total_mouse_dy)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.paused = not self.paused
                    if self.paused:
                        self.controller.release_all_keys()
                        print("⏸️ System PAUSED")
                    else:
                        print("▶️ System RESUMED")
                elif key == ord('v'):
                    status = "ENABLED" if self.voice_controller.toggle() else "DISABLED"
                    print(f"🎤 Voice recognition {status}")
                elif key == ord('+') or key == ord('='):  # Both + and = keys
                    self.detector.adjust_camera_sensitivity(increase=True)
                elif key == ord('-'):
                    self.detector.adjust_camera_sensitivity(increase=False)
                elif key == ord('k'):
                    self.start_calibration()
                elif key == ord('j'):
                    self.add_custom_gesture()
                elif key == ord('l'):
                    if self.detector.mlp_trainer:
                        self.detector.mlp_trainer.list_gestures()
                    else:
                        print("No MLP trainer available")
                elif key == ord('z'):
                    if self.detector.mlp_trainer:
                        confirm = input("\nClear ALL training data? (y/N): ").strip().lower()
                        if confirm == 'y':
                            self.detector.mlp_trainer.clear_all_data()
                            self.detector.use_mlp_model = False
                        else:
                            print("Clear cancelled")
                    else:
                        print("No MLP trainer available")
                    
        except KeyboardInterrupt:
            print("\n🛑 System interrupted by user")
            
        finally:
            self.cleanup()
    
    
    
    def handle_voice_command(self, spoken_text: str):
        """Handle voice commands from the voice controller"""
        if self.paused:
            return  # Don't process voice commands when paused
        
        voice_command = self.voice_controller.execute_voice_command(spoken_text)
        if voice_command:
            self.controller.send_voice_action(voice_command)
    
    
    
    def start_calibration(self):
        """Start MLP calibration for all default gestures"""
        print("\nMLP Gesture Calibration")
        print("=" * 50)
        print("This will calibrate all default gestures (WSAD + Jump + Attack)")
        print("You'll capture 50 pictures per gesture in 10-second sessions")
        print("Get ready to hold each gesture with slight angle variations!")
        
        input("Press ENTER to start calibration...")
        
        success = self.detector.start_calibration()
        if success:
            print("MLP calibration completed successfully!")
        else:
            print("MLP calibration failed")
    
    def add_custom_gesture(self):
        """Add a custom gesture to the MLP system"""
        print("\nAdd Custom Gesture")
        print("=" * 30)
        
        gesture_name = input("Enter gesture name (e.g., 'crouch'): ").strip()
        if not gesture_name:
            print("Invalid gesture name")
            return
        
        key_mapping = input("Enter key to map (e.g., 'c'): ").strip().lower()
        if not key_mapping:
            print("Invalid key mapping")
            return
        
        print(f"\nAdding custom gesture: {gesture_name} → {key_mapping}")
        print("You'll capture 50 pictures in 10-second sessions")
        input("Press ENTER when ready...")
        
        success = self.detector.add_custom_gesture(gesture_name, key_mapping)
        if success:
            print("Custom gesture added successfully!")
        else:
            print("Custom gesture addition failed")
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        self.controller.release_all_keys()
        self.voice_controller.cleanup()
        self.detector.release()
        print("✅ Cleanup complete!")

def get_groq_api_key():
    """Get Groq API key from user (optional)"""
    print("\n🤖 AI Enhancement Available!")
    print("Want to enable AI-powered voice parsing? (Optional)")
    print("Get free API key at: https://console.groq.com/keys")
    
    while True:
        api_key = input("\nEnter Groq API key (or press Enter to skip): ").strip()
        
        if not api_key:
            print("⚡ Continuing with standard voice recognition...")
            return None
            
        # Basic validation
        if len(api_key) < 10:
            print("❌ API key too short. Try again or press Enter to skip.")
            continue
            
        # Test API key
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            # Quick test call
            response = requests.get("https://api.groq.com/openai/v1/models", 
                                  headers=headers, timeout=10)
            if response.status_code == 200:
                print("✅ API key validated! AI mode will be available.")
                return api_key
            else:
                print(f"❌ API validation failed (status: {response.status_code}). Try again or press Enter to skip.")
                print("💡 Check your API key at: https://console.groq.com/keys")
        except Exception as e:
            print(f"❌ Could not validate API key: {e}")
            print("💡 Check internet connection or try again. Press Enter to skip.")

def get_google_ai_key():
    """Get Google AI Studio API key from user (optional)"""
    print("\n🤖 Custom Gesture AI Available!")
    print("Want to enable custom gesture training? (Optional)")
    print("Get free API key at: https://aistudio.google.com/app/apikey")
    
    while True:
        api_key = input("\nEnter Google AI Studio API key (or press Enter to skip): ").strip()
        
        if not api_key:
            print("⚡ Continuing without custom gesture AI...")
            return None
            
        # Basic validation
        if len(api_key) < 20:
            print("❌ API key too short. Try again or press Enter to skip.")
            continue
            
        if not api_key.startswith('AIza'):
            print("⚠️ API key doesn't start with 'AIza' - might still work")
        
        print("✅ Google AI key accepted! Custom gesture training will be available.")
        return api_key

def main():
    """Main entry point"""
    print("🤖 AI Gesture Gaming Controller v1.0")
    print("=" * 50)
    
    # Get optional API keys
    groq_key = get_groq_api_key()
    google_key = get_google_ai_key()
    
    # Initialize system
    system = GestureGamingSystem(groq_api_key=groq_key, google_ai_key=google_key)
    
    try:
        system.start()
    except Exception as e:
        print(f"❌ Error: {e}")
        system.cleanup()

if __name__ == "__main__":
    main()