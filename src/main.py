import cv2
import time
import threading
from gesture_detector import GestureDetector
from input_controller import InputController
from head_tracker import HeadTracker
from voice_controller import VoiceController
from game_controller_mapper import GameControllerMapper
from face_detector import FaceDetector

class GestureGamingSystem:
    def __init__(self, groq_api_key=None, google_ai_key=None, openai_api_key=None, deepgram_api_key=None):
        self.detector = GestureDetector(google_ai_key=google_ai_key)
        self.controller = InputController()
        self.head_tracker = HeadTracker(enabled=False)  # OFF by default to avoid neck pain
        self.voice_controller = VoiceController(groq_api_key=groq_api_key, openai_api_key=openai_api_key, deepgram_api_key=deepgram_api_key)
        self.game_mapper = GameControllerMapper(groq_api_key) if groq_api_key else None
        self.face_detector = FaceDetector()
        self.running = False
        self.paused = False
        
        # Store groq key for LLM parsing
        self.groq_api_key = groq_api_key
        
        # Connect components
        self.controller.set_gesture_detector(self.detector)  # For voice hand camera control
        self.controller.groq_api_key = groq_api_key  # Give input controller access to LLM
        self.detector.parent_controller = self.controller  # For dynamic gesture mapping
        self.voice_controller._on_speech_recognized = self.handle_voice_command
        
    def setup_game_controls(self):
        """Setup game-specific controls using LLM analysis"""
        if not self.game_mapper:
            print("Warning: No Groq API key - using default controls")
            return
        
        print("\nGame Control Setup")
        print("=" * 30)
        game_name = input("Enter game name (e.g., 'last of us 2'): ").strip()
        
        if not game_name:
            print("No game specified - using default controls")
            return
        
        # Load game controls
        game_controls = self.game_mapper.load_game_controls(game_name)
        if not game_controls:
            print(f"No controls found for '{game_name}' - using default controls")
            return
        
        print(f"Loaded controls for '{game_name}'")
        
        # Get current gesture training state
        current_gesture_count = 0
        if self.detector.mlp_trainer and self.detector.mlp_trainer.gesture_classes:
            current_gesture_count = len(self.detector.mlp_trainer.gesture_classes)
        
        print(f"Current trained gestures: {current_gesture_count}/8")
        
        # Analyze with LLM
        print("Analyzing optimal control mappings with LLM...")
        mappings = self.game_mapper.analyze_game_controls(game_controls, current_gesture_count, self.detector.mlp_trainer)
        
        # Apply mappings (CLEAR previous voice commands and apply new ones)
        if mappings:
            self.game_mapper.apply_mappings(mappings, self.detector.mlp_trainer, self.controller, self.voice_controller)
            print("\nGame controls optimized!")
        else:
            print("Failed to generate control mappings")

    def start(self):
        """Start the gesture gaming system"""
        print("AI Gesture Gaming System Starting...")
        
        # Setup game-specific controls first
        self.setup_game_controls()
        
        print("\nDual Hand Gaming System:")
        print("   🎮 Left Hand = Camera Control (smooth movement)")
        print("   Right Hand = All Gestures (WASD, jump, attack)")
        print("   No retraining needed - use existing gestures on RIGHT hand!")
        print("\nGesture Controls (RIGHT HAND):")
        print("   Open Palm → Jump (Space)")
        print("   Closed Fist → Backward (S)")
        print("   Index Point Left → Strafe Left (A)")
        print("   Index Point Right → Strafe Right (D)")
        print("   Index Point Up → Forward (W)")
        print("   Rock Sign (Index + Pinky) → Attack (Left Click)")
        print("\nCamera Controls (LEFT HAND):")
        print("   Move left hand to control camera smoothly")
        print("   Say 'flame on/off' to control hand camera")
        print("   '+' - Increase hand camera sensitivity")
        print("   '-' - Decrease hand camera sensitivity")
        print("\n👁️ Wink Controls:")
        print("   Wink (close one eye) to aim - holds Right Mouse Button")
        print("   Open eye to stop aiming")
        print("   'w' - Toggle wink detection debug mode")
        print("   ',' - Decrease aim smoothing (more responsive)")
        print("   '.' - Increase aim smoothing (less jittery)")
        print("   '/' - Toggle left hand performance debug")
        print("   '\\' - Show left hand performance stats")
        print("   ';' - Toggle camera movement debug")
        print("   \"'\" - Toggle dual hand mode on/off")
        print("\nMLP Gesture System:")
        print("   'k' - Start full calibration (WSAD + default gestures)")
        print("   'j' - Add custom gesture (right/left hand, key/camera control)")
        print("   'l' - List calibrated gestures")
        print("   'z' - Clear all data and start fresh")
        print("   'o' - Toggle gesture-based camera control (enable/disable/revert)")
        print("   📷 Camera only moves when you gesture 'cam' with LEFT hand")
        print("\n🎤 Voice Commands:")
        print("   Say 'flashlight', 'heal', 'listen', etc.")
        print("   'v' - Toggle voice recognition")
        print("   'n' - Add new voice command (say phrase → press key/combo)")
        print("   'm' - Remove voice command (by voice phrase)")
        print("   'b' - List all voice commands")
        print("   't' - Toggle voice recognition API (Google ↔ Deepgram)")
        print("   'u' - Clear and reload voice commands from JSON")
        print("   'g' - Toggle voice debug mode (filter random speech)")
        print("   'c' - Recalibrate microphone (if voice not working well)")
        print("   'h' - Toggle gaming mode (optimized for game audio)")
        print("   'x' - Toggle noise suppression (filter game sounds)")
        print("   'y' - Toggle dual hand mode (L=camera, R=gestures)")
        print("   'F6' - Toggle real-time voice mode (always listening)")
        print("\nAttack Controls:")
        print("   '[' - Decrease attack hold duration (for faster games)")
        print("   ']' - Increase attack hold duration (for precise shooting)")
        print("\nPerformance Controls:")
        print("   '`' - Toggle adaptive FPS mode (auto-reduce during high CPU)")
        print("   '\\' - Show performance statistics (FPS, CPU usage)")
        print("\nAudio Isolation Controls:")
        print("   'i' - Toggle game audio filtering (ON/OFF)")
        print("   ',' - More filtering (less sensitive to game audio)")
        print("   '.' - Less filtering (more sensitive to voice)")
        print("   '/' - 🚨 EMERGENCY MODE (disable ALL filtering for debugging)")
        print("\nSystem Controls:")
        print("   'p' - Pause/Resume")
        print("   'q' - Quit")
        print("   'r' - Emergency release all keys")
        print("\nSystem Ready! Show your hand to the camera...")
        
        
        # Start camera system
        print("🎥 Starting threaded camera system...")
        camera_started = self.detector.start_camera()
        if not camera_started:
            print("❌ Failed to start camera - exiting")
            return
        
        # Display performance info
        if self.detector.threaded_camera:
            stats = self.detector.threaded_camera.get_stats()
            print(f"   Target FPS: {stats['target_fps']}")
            print(f"   Adaptive mode: {stats['adaptive_mode']}")
            print(f"   CPU threshold: {self.detector.threaded_camera.cpu_threshold}%")
        
        # Start voice recognition
        self.voice_controller.start_listening()
        
        # Detect and set aim key for wink mapping from current voice commands
        self.controller.detect_and_set_aim_key(self.voice_controller.action_mappings)
        
        self.running = True
        
        try:
            while self.running:
                # Process frame (now returns hand camera movement too)
                frame, gesture, confidence, hand_mouse_dx, hand_mouse_dy = self.detector.process_frame()
                
                # Process wink detection for aiming
                is_aiming, wink_type, wink_confidence = False, "none", 0.0
                if frame is not None:
                    is_aiming, wink_type, wink_confidence = self.face_detector.detect_wink(frame)
                    # Add wink debug info to frame
                    frame = self.face_detector.draw_debug_info(frame, is_aiming, wink_type, wink_confidence)
                
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
                    
                    # Add voice API status with real-time indicator
                    voice_api = "Google"
                    api_color = (255, 255, 255)  # White for free
                    # Add Deepgram fallback indicator
                    if self.voice_controller.use_deepgram_fallback:
                        voice_api += "+DG"
                    
                    # Add real-time indicator
                    if self.voice_controller.real_time_mode:
                        voice_api += " (RT)"
                        api_color = (0, 255, 0)  # Green for real-time
                    
                    cv2.putText(frame, f"Voice: {voice_api}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, api_color, 2)
                    
                    # Add hand mode status
                    hand_mode = self.detector.get_hand_mode_status()
                    hand_color = (255, 0, 255)  # Magenta for hand mode
                    cv2.putText(frame, f"Hands: {hand_mode}", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
                    
                    # Add performance stats if threaded camera is available
                    if self.detector.threaded_camera:
                        stats = self.detector.threaded_camera.get_stats()
                        fps_color = (0, 255, 0) if stats['current_fps'] >= stats['target_fps'] * 0.8 else (0, 255, 255)
                        cv2.putText(frame, f"FPS: {stats['current_fps']:.1f}/{stats['target_fps']}", 
                                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
                    
                    # Add instructions
                    cv2.putText(frame, "Press 'p' to pause, 'q' to quit", (10, frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Resize frame for better performance and smaller window
                    display_frame = cv2.resize(frame, (480, 360))  # Reduced from original size
                    
                    # Show smaller frame
                    cv2.imshow("AI Gesture Gaming Controller", display_frame)
                    
                    # Send input if not paused
                    if not self.paused:
                        self.controller.send_action(gesture, confidence)
                        # Send camera movement (combined head tracking + hand camera)
                        if total_mouse_dx != 0 or total_mouse_dy != 0:
                            self.controller.send_camera_movement(total_mouse_dx, total_mouse_dy)
                        # Send wink aiming action (hold/release right mouse button)
                        self.controller.send_wink_action(is_aiming, wink_type, wink_confidence)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.paused = not self.paused
                    if self.paused:
                        self.controller.release_all_keys()
                        print("System PAUSED")
                    else:
                        print("System RESUMED")
                elif key == ord('v'):
                    status = "ENABLED" if self.voice_controller.toggle() else "DISABLED"
                    print(f"🎤 Voice recognition {status}")
                elif key == ord('+') or key == ord('='):  # Both + and = keys
                    # Adjust camera sensitivity based on current mode
                    if hasattr(self.controller, 'adjust_camera_sensitivity'):
                        self.controller.adjust_camera_sensitivity(increase=True)
                    else:
                        self.detector.adjust_camera_sensitivity(increase=True)
                elif key == ord('-'):
                    # Adjust camera sensitivity based on current mode
                    if hasattr(self.controller, 'adjust_camera_sensitivity'):
                        self.controller.adjust_camera_sensitivity(increase=False)
                    else:
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
                elif key == ord('r'):
                    # Emergency release all keys
                    print("Emergency: Releasing ALL keys (including voice-controlled)")
                    try:
                        from pynput.keyboard import Key
                        from pynput import mouse
                        self.controller.keyboard_controller.release(Key.shift)
                        self.controller.keyboard_controller.release(Key.ctrl)
                        self.controller.keyboard_controller.release(Key.alt)
                        self.controller.mouse_controller.release(mouse.Button.right)
                        print("Released: Shift, Ctrl, Alt, Right Mouse")
                    except:
                        pass
                    self.controller.pressed_keys.clear()
                    self.controller.is_aiming_with_wink = False
                    # Clean up voice-held keys
                    self.controller.cleanup_voice_held_keys()
                    print("Cleared pressed_keys tracking and voice-held keys")
                elif key == ord('w'):
                    # Toggle wink detection debug mode
                    debug_status = self.face_detector.toggle_debug()
                    print(f"👁️ Wink detection debug: {'ON' if debug_status else 'OFF'}")
                elif key == ord('n'):
                    # Add new voice command
                    self.add_voice_command_interactive()
                elif key == ord('m'):
                    # Remove voice command
                    self.remove_voice_command_interactive()
                elif key == ord('b'):
                    # List all voice commands
                    self.voice_controller.list_available_commands()
                elif key == ord('t'):
                    # Toggle voice recognition API
                    self.toggle_voice_api()
                elif key == ord('g'):
                    # Toggle voice debug mode
                    self.voice_controller.toggle_debug_mode()
                elif key == ord('u'):
                    # Clear and reload voice commands from JSON
                    self.voice_controller.clear_and_reload_commands()
                    # Detect and set aim key for wink mapping
                    self.controller.detect_and_set_aim_key(self.voice_controller.action_mappings)
                elif key == ord('c'):
                    # Recalibrate microphone
                    self.voice_controller.recalibrate_microphone()
                elif key == ord('h'):
                    # Toggle gaming mode
                    self.voice_controller.toggle_gaming_mode()
                elif key == ord('x'):
                    # Toggle noise suppression
                    self.voice_controller.toggle_noise_suppression()
                elif key == ord('y'):
                    # Toggle dual hand mode
                    self.detector.toggle_dual_hand_mode()
                elif key == 0xFF & 0x43:  # F6 key code
                    # Toggle real-time voice mode
                    self.voice_controller.toggle_real_time_mode()
                elif key == ord('['):
                    # Decrease attack duration
                    self.controller.adjust_attack_duration(increase=False)
                elif key == ord(']'):
                    # Increase attack duration
                    self.controller.adjust_attack_duration(increase=True)
                elif key == ord(','):
                    # Decrease aim smoothing (more responsive)
                    self.controller.adjust_aim_smoothing(increase_delay=False)
                elif key == ord('.'):
                    # Increase aim smoothing (less jittery)
                    self.controller.adjust_aim_smoothing(increase_delay=True)
                elif key == ord('/'):
                    # Toggle left hand performance debugging
                    debug_status = self.detector.toggle_performance_debug()
                    print(f"📊 Left hand performance debug: {'ON' if debug_status else 'OFF'}")
                elif key == ord('\\'):
                    # Print left hand performance stats
                    if hasattr(self.detector, 'get_left_hand_performance_stats'):
                        stats = self.detector.get_left_hand_performance_stats()
                        print(f"📊 {stats}")
                    else:
                        print("Performance stats not available")
                elif key == ord(';'):
                    # Toggle camera movement debugging
                    debug_status = self.controller.toggle_camera_movement_debug()
                    print(f"🎥 Camera movement debug: {'ON' if debug_status else 'OFF'}")
                elif key == ord('\''):
                    # Toggle dual hand mode debugging
                    self.detector.dual_hand_mode = not self.detector.dual_hand_mode
                    status = "ON" if self.detector.dual_hand_mode else "OFF"
                    print(f"👐 Dual hand mode: {status}")
                elif key == ord('|'):
                    # Toggle camera movement debugging (Shift+\)
                    debug_status = self.controller.toggle_camera_movement_debug()
                    print(f"🎥 Camera movement debug: {'ON' if debug_status else 'OFF'}")
                elif key == ord('`'):
                    # Toggle adaptive FPS mode (backtick key)
                    if self.detector.threaded_camera:
                        current_mode = self.detector.threaded_camera.adaptive_mode
                        self.detector.threaded_camera.set_adaptive_mode(not current_mode)
                    else:
                        print("⚠️ Threaded camera not available")
                elif key == ord('\\'):
                    # Show performance stats (backslash key)
                    if self.detector.threaded_camera:
                        stats = self.detector.threaded_camera.get_stats()
                        print(f"\n📊 Performance Stats:")
                        print(f"   Current FPS: {stats['current_fps']:.1f}")
                        print(f"   Target FPS: {stats['target_fps']}")
                        print(f"   Queue size: {stats['queue_size']}/{self.detector.threaded_camera.buffer_size}")
                        print(f"   Adaptive mode: {'ON' if stats['adaptive_mode'] else 'OFF'}")
                        print(f"   CPU threshold: {self.detector.threaded_camera.cpu_threshold}%")
                        try:
                            import psutil
                            print(f"   Current CPU: {psutil.cpu_percent():.1f}%")
                        except ImportError:
                            print("   Current CPU: N/A (psutil not available)")
                    else:
                        print("⚠️ Threaded camera not available - using direct capture")
                elif key == ord('i'):
                    # Toggle game audio filtering
                    status = self.voice_controller.toggle_game_audio_filter()
                    filter_status = "ON" if status else "OFF"
                    print(f"🎮 Game audio filtering: {filter_status}")
                elif key == ord(','):
                    # Decrease audio sensitivity (more filtering)
                    self.voice_controller.adjust_audio_sensitivity(increase=False)
                elif key == ord('.'):
                    # Increase audio sensitivity (less filtering)
                    self.voice_controller.adjust_audio_sensitivity(increase=True)
                elif key == ord('/'):
                    # Toggle emergency mode - disable ALL audio filtering for debugging
                    status = self.voice_controller.toggle_emergency_mode()
                    mode_status = "ENABLED" if status else "DISABLED"
                    print(f"🚨 Emergency mode: {mode_status} (disables ALL audio filtering)")
                elif key == ord('o'):
                    # Toggle gesture-based camera control
                    if hasattr(self.detector, 'toggle_gesture_camera_control'):
                        status = self.detector.toggle_gesture_camera_control()
                        mode_status = "ENABLED" if status else "DISABLED"
                        print(f"🎥 Gesture camera control: {mode_status}")
                        if not status:
                            print("🔄 Reverted to always-on left hand camera control")
                    else:
                        print("⚠️ No gesture camera control system available")
                    
        except KeyboardInterrupt:
            print("\nSystem interrupted by user")
            
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
        """Add a custom gesture with hand and type selection"""
        print("\nAdd Custom Gesture")
        print("=" * 30)
        
        # Step 1: Choose hand
        print("Which hand do you want to use?")
        print("1. Right Hand (for actions/keys)")
        print("2. Left Hand (for camera control or actions)")
        
        hand_choice = input("Enter choice (1 or 2): ").strip()
        
        if hand_choice == "2":
            hand_type = "left"
            print("\n📱 Left Hand Selected")
        else:
            hand_type = "right" 
            print("\n🎮 Right Hand Selected")
        
        # Step 2: Choose gesture type
        print(f"\nWhat type of {hand_type} hand gesture?")
        print("1. Key-based (press a specific key when gesture is made)")
        print("2. Camera control (enable/disable camera movement)")
        
        gesture_type = input("Enter choice (1 or 2): ").strip()
        
        # Step 3: Get gesture details
        gesture_name = input(f"\nEnter gesture name (e.g., 'fist', 'palm', 'point'): ").strip()
        if not gesture_name:
            print("Invalid gesture name")
            return
        
        if gesture_type == "2":
            # Camera control gesture
            key_mapping = "camera_control"
            print(f"\n🎥 Adding {hand_type} hand CAMERA CONTROL gesture: {gesture_name}")
            print("💡 Camera will only move when this gesture is detected")
            print("🔄 You can disable this later if it doesn't work well")
        else:
            # Key-based gesture
            key_mapping = input("Enter key to press (e.g., 'c', 'r', 'shift'): ").strip().lower()
            if not key_mapping:
                print("Invalid key mapping")
                return
            print(f"\n🎯 Adding {hand_type} hand KEY gesture: {gesture_name} → {key_mapping}")
        
        print(f"✋ Use your {hand_type.upper()} hand for this gesture")
        print("📸 You'll capture 50 pictures in 10-second sessions")
        input("Press ENTER when ready...")
        
        # Add gesture with hand type
        success = self.detector.add_custom_gesture(gesture_name, key_mapping, hand_type=hand_type)
        if success:
            if gesture_type == "2":
                print("✅ Camera control gesture added successfully!")
                print("🎮 Camera will now respond to your left hand gesture")
                print("⚙️  You can disable this with 'o' key if needed")
            else:
                print("✅ Key gesture added successfully!")
                print(f"🎮 Gesture '{gesture_name}' will now press '{key_mapping}'")
        else:
            print("❌ Gesture addition failed")
    
    def add_voice_command_interactive(self):
        """Interactive method to add voice commands"""
        print("\nAdd Voice Command")
        print("=" * 30)
        print("Create a custom voice command that will press a key/combo when you say it.")
        print("Examples:")
        print("  Voice: 'heal' → Key: 'h'")
        print("  Voice: 'inventory' → Key: 'i'") 
        print("  Voice: 'dodge roll' → Key: 'w + space'")
        print("  Voice: 'sprint jump' → Key: 'shift + space'")
        print("  Voice: 'heavy attack' → Key: 'right_click'")
        print("  Voice: 'grenade' → Key: 'ctrl + g'")
        print()
        
        # Ask if this is a hold key
        is_hold_key = input("Is this a HOLD key? (y/N): ").strip().lower() == 'y'
        
        if is_hold_key:
            # Hold key flow
            print("\nHold Key Setup:")
            print("You'll need to create two commands:")
            print("1. Command to HOLD the key")
            print("2. Command to RELEASE the key")
            print()
            
            # Get hold action phrase
            hold_phrase = input("What do you want to SAY to HOLD the key? (e.g., 'start running', 'aim'): ").strip().lower()
            if not hold_phrase:
                print("Cancelled")
                return
            
            # Get key to hold
            print("Key examples: 'shift', 'ctrl', 'left_click', 'right_click'")
            key_to_hold = input(f"What KEY should be HELD when you say '{hold_phrase}'? ").strip().lower()
            if not key_to_hold:
                print("Cancelled")
                return
            
            # Get release action phrase
            release_phrase = input(f"What do you want to SAY to RELEASE the key? (e.g., 'stop running', 'stop aim'): ").strip().lower()
            if not release_phrase:
                print("Cancelled")
                return
            
            # Create action names
            hold_action_name = hold_phrase.replace(" ", "_").replace("'", "")
            release_action_name = release_phrase.replace(" ", "_").replace("'", "")
            
            # Show what will be created
            print(f"\nCreating HOLD commands:")
            print(f"  Say '{hold_phrase}' → HOLD key: {key_to_hold}")
            print(f"  Say '{release_phrase}' → RELEASE key: {key_to_hold}")
            
            confirm = input("Confirm? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Cancelled")
                return
            
            # Create both voice commands
            hold_success = self.voice_controller.add_custom_voice_command(hold_action_name, hold_phrase, f"hold_{key_to_hold}")
            release_success = self.voice_controller.add_custom_voice_command(release_action_name, release_phrase, f"release_{key_to_hold}")
            
            if hold_success and release_success:
                print(f"✅ Hold commands added!")
                print(f"  Say '{hold_phrase}' to hold: {key_to_hold}")
                print(f"  Say '{release_phrase}' to release: {key_to_hold}")
                # Save to file
                self.voice_controller.save_voice_config()
            else:
                print("❌ Failed to add hold commands")
        else:
            # Regular key flow (existing)
            # Get voice command phrase
            voice_phrase = input("What do you want to SAY? (e.g., 'heal', 'dodge roll'): ").strip().lower()
            if not voice_phrase:
                print("Cancelled")
                return
            
            # Get key to press with guided selection
            print("\nSelect what type of action:")
            print("1. Single key (e.g., 'h', 'f', 'space')")
            print("2. Single mouse button (e.g., left click, right click)")
            print("3. Key combination (e.g., Shift + Space, Ctrl + F)")
            print("4. Advanced/Custom (enter raw format)")
            
            action_type = input("Choose option (1-4): ").strip()
            
            if action_type == "1":
                # Single key
                key_to_press = self._get_single_key()
            elif action_type == "2":
                # Single mouse button
                key_to_press = self._get_mouse_button()
            elif action_type == "3":
                # Key combination
                key_to_press = self._get_key_combo()
            elif action_type == "4":
                # Advanced/Custom
                print("Advanced format examples: 'Key.mouse.middle', 'shift + right_click'")
                key_to_press = input("Enter custom key format: ").strip()
            else:
                print("Invalid option. Cancelled.")
                return
            
            if not key_to_press:
                print("Cancelled")
                return
            
            # Create action name from voice phrase (for internal use)
            action_name = voice_phrase.replace(" ", "_").replace("'", "")
            
            # Show what will be created
            if "+" in key_to_press:
                print(f"\nCreating voice command: Say '{voice_phrase}' → Press combo: {key_to_press}")
            else:
                print(f"\nCreating voice command: Say '{voice_phrase}' → Press key: {key_to_press}")
                
            confirm = input("Confirm? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Cancelled")
                return
            
            # Create the voice command using the improved interface
            success = self.voice_controller.add_custom_voice_command(action_name, voice_phrase, key_to_press)
            if success:
                if "+" in key_to_press:
                    print(f"✅ Voice command added! Say '{voice_phrase}' to press combo: {key_to_press}")
                else:
                    print(f"✅ Voice command added! Say '{voice_phrase}' to press: {key_to_press}")
                # Save to file
                self.voice_controller.save_voice_config()
            else:
                print("❌ Failed to add voice command")
    
    def _get_single_key(self):
        """Get a single key from user in user-friendly way"""
        print("\nCommon keys:")
        print("Letters: a, b, c, ... z")
        print("Numbers: 1, 2, 3, ... 0")
        print("Special: space, enter, shift, ctrl, alt, tab, esc")
        print("F-keys: f1, f2, f3, ... f12")
        print("Actions: none/stop (releases all keys)")
        
        key = input("Enter the key: ").strip().lower()
        return key if key else None
    
    def _get_mouse_button(self):
        """Get mouse button selection from user"""
        print("\nMouse buttons:")
        print("1. Left click")
        print("2. Right click") 
        print("3. Middle click (wheel button)")
        print("4. Mouse wheel up")
        print("5. Mouse wheel down")
        
        choice = input("Choose mouse button (1-5): ").strip()
        
        mouse_map = {
            "1": "left_click",
            "2": "right_click", 
            "3": "middle_click",
            "4": "mouse_wheel_up",
            "5": "mouse_wheel_down"
        }
        
        return mouse_map.get(choice)
    
    def _get_key_combo(self):
        """Get key combination from user in user-friendly way"""
        print("\nKey Combination Builder")
        print("We'll build your combo step by step...")
        
        keys = []
        
        # Get modifier keys first
        print("\nModifier keys (optional):")
        print("1. Shift")
        print("2. Ctrl") 
        print("3. Alt")
        print("4. None/Skip")
        
        modifier = input("Choose modifier (1-4, or press Enter to skip): ").strip()
        modifier_map = {"1": "shift", "2": "ctrl", "3": "alt"}
        
        if modifier in modifier_map:
            keys.append(modifier_map[modifier])
            print(f"Added: {modifier_map[modifier]}")
        
        # Get main action
        print("\nMain action:")
        print("1. Letter key (a-z)")
        print("2. Number key (0-9)")
        print("3. Special key (space, enter, etc.)")
        print("4. Mouse button")
        
        action_type = input("Choose main action type (1-4): ").strip()
        
        if action_type == "1":
            letter = input("Enter letter (a-z): ").strip().lower()
            if len(letter) == 1 and letter.isalpha():
                keys.append(letter)
        elif action_type == "2":
            number = input("Enter number (0-9): ").strip()
            if len(number) == 1 and number.isdigit():
                keys.append(number)
        elif action_type == "3":
            print("Special keys: space, enter, tab, esc, f1-f12")
            special = input("Enter special key: ").strip().lower()
            keys.append(special)
        elif action_type == "4":
            mouse = self._get_mouse_button()
            if mouse:
                keys.append(mouse)
        
        if len(keys) == 0:
            return None
        elif len(keys) == 1:
            return keys[0]
        else:
            combo = " + ".join(keys)
            print(f"\nCreated combo: {combo}")
            return combo
    
    def remove_voice_command_interactive(self):
        """Interactive method to remove voice commands"""
        print("\nRemove Voice Command")
        print("=" * 30)
        
        # Show all current voice commands
        print("Current voice commands:")
        for i, (action_id, config) in enumerate(self.voice_controller.action_mappings.items(), 1):
            intents_str = ", ".join(config['intents'][:3])  # Show first 3
            if len(config['intents']) > 3:
                intents_str += f" (+{len(config['intents'])-3} more)"
            print(f"  {i:2}. {config['name']} ({config['key']}) - Say: {intents_str}")
        
        print()
        voice_phrase = input("Which voice phrase do you want to REMOVE? (e.g., 'heal', 'inventory'): ").strip().lower()
        if not voice_phrase:
            print("Cancelled")
            return
        
        # Find which action contains this voice phrase
        found_action = None
        for action_id, config in self.voice_controller.action_mappings.items():
            if voice_phrase in config['intents']:
                found_action = action_id
                break
        
        if found_action:
            config = self.voice_controller.action_mappings[found_action]
            print(f"\nFound '{voice_phrase}' in {config['name']} → {config['key']}")
            confirm = input(f"Remove '{voice_phrase}'? (y/N): ").strip().lower()
            
            if confirm == 'y':
                success = self.voice_controller.remove_voice_command(found_action, voice_phrase)
                if success:
                    print(f"✅ Removed '{voice_phrase}'")
                    # Save to file
                    self.voice_controller.save_voice_config()
                else:
                    print("❌ Failed to remove command")
            else:
                print("Cancelled")
        else:
            print(f"❌ Voice phrase '{voice_phrase}' not found")
    
    def toggle_voice_api(self):
        """Toggle voice API between Google and Deepgram"""
        if self.voice_controller.use_deepgram:
            # Deepgram → Google
            self.voice_controller.use_deepgram = False
            self.voice_controller.use_deepgram_fallback = False
            print("Switched to Google Web Speech (Free)")
        else:
            # Google → Deepgram (if available)
            if self.voice_controller.deepgram_api_key:
                self.voice_controller.use_deepgram = True
                self.voice_controller.use_deepgram_fallback = True
                print("Switched to Deepgram API")
            else:
                print("Deepgram API key required")
                print("Get key at: https://console.deepgram.com/")
                api_key = input("Enter Deepgram API key (or press Enter to skip): ").strip()
                
                if api_key and len(api_key) > 20:
                    self.voice_controller.deepgram_api_key = api_key
                    self.voice_controller.use_deepgram = True
                    self.voice_controller.use_deepgram_fallback = True
                    print("Deepgram API enabled!")
                else:
                    print("Staying with Google Web Speech (Free)")

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        self.controller.release_all_keys()
        # Clean up voice-held keys
        self.controller.cleanup_voice_held_keys()
        self.voice_controller.cleanup()
        self.detector.release()
        # Clean up wink aiming
        self.controller.cleanup_wink_aiming()
        print("Cleanup complete!")

def get_groq_api_key():
    """Get Groq API key from user (optional)"""
    print("\nAI Enhancement Available!")
    print("Want to enable AI-powered voice parsing? (Optional)")
    print("Get free API key at: https://console.groq.com/keys")
    
    while True:
        api_key = input("\nEnter Groq API key (or press Enter to skip): ").strip()
        
        if not api_key:
            print("Continuing with standard voice recognition...")
            return None
            
        # Basic validation
        if len(api_key) < 10:
            print("API key too short. Try again or press Enter to skip.")
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
                print("API key validated! AI mode will be available.")
                return api_key
            else:
                print(f"API validation failed (status: {response.status_code}). Try again or press Enter to skip.")
                print("Check your API key at: https://console.groq.com/keys")
        except Exception as e:
            print(f"Could not validate API key: {e}")
            print("Check internet connection or try again. Press Enter to skip.")

def get_google_ai_key():
    """Get Google AI Studio API key from user (optional)"""
    print("\nCustom Gesture AI Available!")
    print("Want to enable custom gesture training? (Optional)")
    print("Get free API key at: https://aistudio.google.com/app/apikey")
    
    while True:
        api_key = input("\nEnter Google AI Studio API key (or press Enter to skip): ").strip()
        
        if not api_key:
            print("Continuing without custom gesture AI...")
            return None
            
        # Basic validation
        if len(api_key) < 20:
            print("API key too short. Try again or press Enter to skip.")
            continue
            
        if not api_key.startswith('AIza'):
            print("Warning: API key doesn't start with 'AIza' - might still work")
        
        print("Google AI key accepted! Custom gesture training will be available.")
        return api_key

def get_deepgram_api_key():
    """Get Deepgram API key for fallback speech recognition (optional)"""
    print("\nEnhanced Voice Recognition Fallback Available!")
    print("Want Deepgram as backup when Google fails? (Premium accuracy)")
    print("Get API key at: https://console.deepgram.com/")
    
    while True:
        api_key = input("\nEnter Deepgram API key (or press Enter to skip): ").strip()
        
        if not api_key:
            print("Continuing with Google only...")
            return None
            
        # Basic validation
        if len(api_key) < 20:
            print("API key too short. Try again or press Enter to skip.")
            continue
        
        print("Deepgram key accepted! Will be used as fallback when Google fails.")
        return api_key

def get_openai_api_key():
    """Get OpenAI API key for Whisper speech recognition (optional)"""
    print("\nEnhanced Voice Recognition Available!")
    print("Want super accurate speech recognition? (Costs ~1¢ per minute)")
    print("Get API key at: https://platform.openai.com/api-keys")
    
    while True:
        api_key = input("\nEnter OpenAI API key (or press Enter to skip): ").strip()
        
        if not api_key:
            print("Continuing with free Google speech recognition...")
            return None
            
        # Basic validation
        if len(api_key) < 20:
            print("API key too short. Try again or press Enter to skip.")
            continue
            
        if not api_key.startswith('sk-'):
            print("Warning: API key doesn't start with 'sk-' - might still work")
        
        print("OpenAI key accepted! Whisper API will be used for speech recognition.")
        return api_key

def main():
    """Main entry point"""
    # Set console encoding for Windows
    import sys
    if sys.platform == "win32":
        import os
        os.system("chcp 65001 > nul")  # Set to UTF-8
    
    print("AI Gesture Gaming Controller v1.0")
    print("=" * 50)
    
    # Get optional API keys
    groq_key = get_groq_api_key()
    google_key = get_google_ai_key()
    deepgram_key = get_deepgram_api_key()
    
    # Initialize system (OpenAI key will be added via toggle)
    system = GestureGamingSystem(groq_api_key=groq_key, google_ai_key=google_key, openai_api_key=None, deepgram_api_key=deepgram_key)
    
    try:
        system.start()
    except Exception as e:
        print(f"Error: {e}")
        system.cleanup()

if __name__ == "__main__":
    main()