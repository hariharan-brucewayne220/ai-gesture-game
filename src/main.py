import cv2
import time
import threading
from gesture_detector import GestureDetector
from input_controller import InputController
from head_tracker import HeadTracker

class GestureGamingSystem:
    def __init__(self):
        self.detector = GestureDetector()
        self.controller = InputController()
        self.head_tracker = HeadTracker(enabled=True)  # Set to False to disable
        self.running = False
        self.paused = False
        
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
        print("\n🎯 Head Tracking:")
        print("   Camera movement via head pose")
        print("   'r' - Reset head reference")
        print("   'h' - Toggle head tracking")
        print("   'f' - Toggle face mesh visibility")
        print("\n🔧 Controls:")
        print("   'p' - Pause/Resume")
        print("   'q' - Quit")
        print("   ESC - Emergency Stop")
        print("\n🚀 System Ready! Show your hand to the camera...")
        
        self.running = True
        
        try:
            while self.running:
                # Process frame
                frame, gesture, confidence = self.detector.process_frame()
                
                # Process head tracking
                if frame is not None:
                    frame, mouse_dx, mouse_dy = self.head_tracker.process_head_movement(frame)
                
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
                    
        except KeyboardInterrupt:
            print("\n🛑 System interrupted by user")
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        self.running = False
        self.controller.release_all_keys()
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