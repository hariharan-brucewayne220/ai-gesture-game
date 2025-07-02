import cv2
import time
import threading
from gesture_detector import GestureDetector
from input_controller import InputController
from head_tracker import HeadTracker
from voice_controller import VoiceController

class GestureGamingSystem:
    def __init__(self):
        self.detector = GestureDetector()
        self.controller = InputController()
        self.head_tracker = HeadTracker(enabled=False)  # OFF by default to avoid neck pain
        self.voice_controller = VoiceController()
        self.running = False
        self.paused = False
        
        # Connect voice controller to main system
        self.voice_controller._on_speech_recognized = self.handle_voice_command
        
    def start(self):
        """Start the gesture gaming system"""
        print("🎮 AI Gesture Gaming System Starting...")
        print("📋 Gesture Controls:")
        print("   🖐 Open Palm → Jump (Space)")
        print("   ✊ Closed Fist → Backward (S)")
        print("   ☝️ Index Point Left → Strafe Left (A)")
        print("   ☝️ Index Point Right → Strafe Right (D)")
        print("   ☝️ Index Point Up → Forward (W)")
        print("   🤟 Rock Sign (Index + Pinky) → Attack (Left Click)")
        print("\n🎯 Camera Controls:")
        print("   Head tracking: OFF by default (avoid neck pain)")
        print("   'h' - Toggle head tracking")
        print("   'r' - Reset head reference")
        print("   'f' - Toggle face mesh visibility")
        print("   'g' - Toggle hand camera control")
        print("   'x' - Reset hand camera reference")
        print("   '+' - Increase hand camera sensitivity")
        print("   '-' - Decrease hand camera sensitivity")
        print("\n🎤 Voice Commands:")
        print("   Say 'flashlight', 'heal', 'listen', etc.")
        print("   'v' - Toggle voice recognition")
        print("   'c' - List all voice commands")
        print("   'b' - Toggle audio feedback")
        print("\n🔧 Controls:")
        print("   'p' - Pause/Resume")
        print("   'q' - Quit")
        print("   ESC - Emergency Stop")
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
                elif key == 27:  # ESC key
                    self.controller.emergency_stop()
                elif key == ord('h'):
                    status = "ENABLED" if self.head_tracker.toggle() else "DISABLED"
                    print(f"🎯 Head tracking {status}")
                elif key == ord('r'):
                    self.head_tracker.reset_reference()
                elif key == ord('f'):
                    status = "SHOWN" if self.head_tracker.toggle_face_mesh() else "HIDDEN"
                    print(f"👤 Face mesh {status}")
                elif key == ord('v'):
                    status = "ENABLED" if self.voice_controller.toggle() else "DISABLED"
                    print(f"🎤 Voice recognition {status}")
                elif key == ord('c'):
                    self.voice_controller.list_available_commands()
                elif key == ord('b'):
                    current_feedback = self.voice_controller.audio_feedback
                    self.voice_controller.set_audio_feedback(not current_feedback)
                elif key == ord('g'):
                    status = "ENABLED" if self.detector.toggle_hand_camera() else "DISABLED"
                    print(f"👋 Hand camera control {status}")
                elif key == ord('x'):
                    self.detector.set_hand_camera_reference()
                elif key == ord('+') or key == ord('='):  # Both + and = keys
                    self.detector.adjust_camera_sensitivity(increase=True)
                elif key == ord('-'):
                    self.detector.adjust_camera_sensitivity(increase=False)
                    
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
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        self.running = False
        self.controller.release_all_keys()
        self.voice_controller.cleanup()
        self.detector.release()
        print("✅ Cleanup complete!")

def main():
    """Main entry point"""
    print("🤖 AI Gesture Gaming Controller v1.0")
    print("=" * 50)
    
    # Initialize system
    system = GestureGamingSystem()
    
    try:
        system.start()
    except Exception as e:
        print(f"❌ Error: {e}")
        system.cleanup()

if __name__ == "__main__":
    main()