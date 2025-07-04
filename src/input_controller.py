import time
from pynput import keyboard, mouse
from pynput.keyboard import Key, Listener
import threading
from typing import Dict, Set
import pydirectinput

class InputController:
    def __init__(self):
        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = mouse.Controller()
        
        # Currently pressed keys (to avoid spam)
        self.pressed_keys: Set[str] = set()
        
        # DIRECTIONAL POINTING GESTURE MAPPINGS
        self.gesture_map = {
            "open_palm": Key.space,  # Jump
            "closed_fist": "s",      # Backward  
            "point_left": "a",       # Strafe Left (Index pointing left)
            "point_right": "d",      # Strafe Right (Index pointing right)
            "point_up": "w",         # Forward (Index pointing up)
            "rock_sign": "attack",   # Attack (Index + pinky)
            "c_shape": "c"           # Crouch (C shape with thumb)
        }
        
        
        # Key press timing
        self.last_action_time = {}
        self.action_cooldown = 0.5  # 500ms cooldown between actions (prevent spam)
        
        # Camera movement timing
        self.last_camera_move = time.time()
        self.camera_cooldown = 0.01  # 10ms between camera moves
        
        # Configure pydirectinput for camera movement
        pydirectinput.FAILSAFE = False
        pydirectinput.PAUSE = 0
        
        # Reference to gesture detector for hybrid support
        self.gesture_detector = None
        
    def send_action(self, gesture: str, confidence: float):
        """Send input action based on gesture"""
        if gesture == "none" or confidence < 0.5:  # Lower threshold to prevent quick releases
            self.release_all_keys()
            return
        
        # Handle MLP gestures (format: "MLP:gesture_name")
        if gesture.startswith("MLP:"):
            actual_gesture = gesture.split(":", 1)[1]
            # Map MLP gestures to their key mappings
            if self.gesture_detector and self.gesture_detector.mlp_trainer:
                key_mapping = self.gesture_detector.mlp_trainer.get_gesture_key_mapping(actual_gesture)
                if key_mapping:
                    # Handle different key types with proper frequency control
                    if key_mapping == "space":
                        # Low frequency - single press with cooldown
                        current_time = time.time()
                        if f"MLP:{actual_gesture}" in self.last_action_time:
                            if current_time - self.last_action_time[f"MLP:{actual_gesture}"] < self.action_cooldown:
                                return
                        self.last_action_time[f"MLP:{actual_gesture}"] = current_time
                        self.jump()
                    elif key_mapping == "click":
                        # Low frequency - single press with cooldown
                        current_time = time.time()
                        if f"MLP:{actual_gesture}" in self.last_action_time:
                            if current_time - self.last_action_time[f"MLP:{actual_gesture}"] < self.action_cooldown:
                                return
                        self.last_action_time[f"MLP:{actual_gesture}"] = current_time
                        self.attack()
                    elif key_mapping in ["w", "a", "s", "d"]:
                        # High frequency - continuous movement (no cooldown)
                        self.press_key(key_mapping, f"MLP:{actual_gesture}")
                    else:
                        self._send_key_action(key_mapping, confidence)
                    return
            
        action = self.gesture_map.get(gesture)
        if not action:
            return
            
        # Movement keys (W,A,S,D) - continuous hold, no cooldown needed
        movement_keys = ["w", "a", "s", "d"]
        
        # Check movement keys first (continuous hold)
        if isinstance(action, str) and action in movement_keys:
            # Execute continuous movement (no cooldown)
            self.press_key(action, gesture)
        # Action keys (Jump, Attack) - single press with cooldown
        elif action == "attack" or action == Key.space:
            current_time = time.time()
            if gesture in self.last_action_time:
                if current_time - self.last_action_time[gesture] < self.action_cooldown:
                    return
            self.last_action_time[gesture] = current_time
            
            # Execute single-press actions
            if action == "attack":
                self.attack()
            elif action == Key.space:
                self.jump()
    
    def press_key(self, key: str, gesture: str):
        """Press and hold a movement key"""
        # Release other movement keys first
        movement_keys = ["w", "a", "s", "d"]
        for mk in movement_keys:
            if mk != key and mk in self.pressed_keys:
                self.keyboard_controller.release(mk)
                self.pressed_keys.discard(mk)
        
        # Press new key if not already pressed
        if key not in self.pressed_keys:
            self.keyboard_controller.press(key)
            self.pressed_keys.add(key)
            print(f"🎮 Action: {gesture} -> {key.upper()}")
        else:
            # Key already pressed, but show continuous feedback (less frequent)
            import time
            current_time = time.time()
            feedback_key = f"feedback_{key}"
            if feedback_key not in self.last_action_time or current_time - self.last_action_time[feedback_key] > 1.0:
                print(f"🎮 Holding: {gesture} -> {key.upper()}")
                self.last_action_time[feedback_key] = current_time
    
    def jump(self):
        """Perform jump action"""
        self.keyboard_controller.press(Key.space)
        time.sleep(0.05)  # Brief press
        self.keyboard_controller.release(Key.space)
        print("🎮 Action: JUMP!")
    
    def attack(self):
        """Perform attack action"""
        self.mouse_controller.click(mouse.Button.left, 1)
        print("🎮 Action: ATTACK!")
    
    def release_all_keys(self):
        """Release all currently pressed keys"""
        for key in list(self.pressed_keys):
            try:
                self.keyboard_controller.release(key)
                print(f"🎮 Released: {key.upper()}")
            except:
                pass
        self.pressed_keys.clear()
    
    def send_voice_action(self, voice_command: Dict):
        """Send input action based on voice command"""
        if not voice_command:
            return
        
        key = voice_command['key']
        name = voice_command['name']
        
        print(f"🎤 Executing voice command: {name} -> {key}")
        
        # Handle different key types
        if key == 'left_click':
            self.mouse_controller.click(mouse.Button.left, 1)
        elif key == 'right_click':
            self.mouse_controller.click(mouse.Button.right, 1)
        elif key == 'shift':
            self.press_and_release_key(Key.shift)
        elif key == 'ctrl':
            self.press_and_release_key(Key.ctrl)
        elif key == 'alt':
            self.press_and_release_key(Key.alt)
        elif key == 'tab':
            self.press_and_release_key(Key.tab)
        elif key == 'esc':
            self.press_and_release_key(Key.esc)
        elif key == 'space':
            self.press_and_release_key(Key.space)
        elif key == 'hand_camera_on':
            # Handle hand camera enable via voice
            self._handle_hand_camera_command(True)
        elif key == 'hand_camera_off':
            # Handle hand camera disable via voice
            self._handle_hand_camera_command(False)
        elif len(key) == 1:  # Single character keys
            self.press_and_release_key(key)
        else:
            print(f"⚠️ Unknown key mapping: {key}")
    
    def press_and_release_key(self, key):
        """Press and release a key (for single actions)"""
        try:
            self.keyboard_controller.press(key)
            time.sleep(0.05)  # Brief press
            self.keyboard_controller.release(key)
        except Exception as e:
            print(f"⚠️ Error pressing key {key}: {e}")
    
    def send_camera_movement(self, mouse_dx: float, mouse_dy: float):
        """Send camera movement - simple and responsive"""
        if abs(mouse_dx) < 0.5 and abs(mouse_dy) < 0.5:
            return
        
        try:
            # Simple conversion to integers - responsive and direct
            int_dx = int(round(mouse_dx))
            int_dy = int(round(mouse_dy))
            
            # Send movement directly
            if int_dx != 0 or int_dy != 0:
                pydirectinput.moveRel(int_dx, int_dy)
                
        except Exception as e:
            print(f"⚠️ Camera movement error: {e}")
    
    def emergency_stop(self):
        """Emergency stop - release everything"""
        self.release_all_keys()
        print("🛑 EMERGENCY STOP - All keys released!")
    
    def set_gesture_detector(self, detector):
        """Set reference to gesture detector for hand camera control"""
        self.gesture_detector = detector
    
    def _handle_hand_camera_command(self, enable: bool):
        """Handle hand camera on/off voice commands"""
        if self.gesture_detector:
            self.gesture_detector.hand_camera_enabled = enable
            status = "ENABLED" if enable else "DISABLED"
            print(f"🎤👋 Hand camera {status} via voice command")
        else:
            print("❌ Could not control hand camera - no detector reference")
    
    def _send_key_action(self, key_mapping: str, confidence: float):
        """Send key action for hybrid gestures"""
        try:
            # Simple key press for hybrid gestures
            if key_mapping not in self.pressed_keys:
                self.keyboard_controller.press(key_mapping)
                self.pressed_keys.add(key_mapping)
                print(f"🎮 Hybrid Action: {key_mapping.upper()}")
        except Exception as e:
            print(f"⚠️ Hybrid key action error: {e}")
    
    def set_gesture_detector(self, gesture_detector):
        """Set reference to gesture detector for hybrid support"""
        self.gesture_detector = gesture_detector

    def add_gesture_mapping(self, gesture_name: str, key: str):
        """Dynamically add a new gesture to key mapping"""
        self.gesture_map[gesture_name] = key
        print(f"🎮 Added mapping: {gesture_name} → {key.upper()}")
    
    def list_gesture_mappings(self):
        """List all current gesture mappings"""
        print("\n🎮 Current Gesture Mappings:")
        print("=" * 40)
        for gesture, key in self.gesture_map.items():
            key_display = key.name if hasattr(key, 'name') else str(key).upper()
            print(f"   {gesture} → {key_display}")
        print("=" * 40)