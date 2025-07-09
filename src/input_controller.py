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
        
        # Voice-controlled held keys (separate from gesture keys)
        self.voice_held_keys: Set[str] = set()
        
        # Game-specific camera control mode
        self.current_game = "default"  # Set by game controller mapper
        self.camera_mode = "mouse"     # "mouse" or "keyboard"
        
        # Mouse sensitivity settings for normal mode
        self.mouse_sensitivity_x = 1.0  # Horizontal mouse sensitivity multiplier
        self.mouse_sensitivity_y = 1.0  # Vertical mouse sensitivity multiplier
        
        # Game-specific wink/blink aiming mappings
        self.wink_mappings = {
            "god of war": Key.ctrl_l,      # Left Ctrl for God of War aiming
            "default": mouse.Button.right  # Right click for other games
        }
        
        # Dynamic aim key (detected from voice commands JSON)
        self.dynamic_aim_key = None
        
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
        
        # Wink aiming state
        self.is_aiming_with_wink = False
        self.last_aim_change_time = 0
        self.aim_release_delay = 0.3  # 300ms delay before releasing aim key
        self.aim_confidence_history = []
        self.max_confidence_history = 5  # Track confidence over 5 frames
        
        # Attack/shooting settings
        self.attack_hold_duration = 0.15  # Duration to hold left click for shooting (seconds)
        
        # Camera control settings
        self.camera_keys_pressed = set()  # Track camera keys for keyboard mode
        self.camera_key_deadzone = 8.0    # Minimum movement to trigger key press (higher = less sensitive)
        self.camera_key_sensitivity = 4.0 # How fast to send key presses (lower = less sensitive)
        self.camera_key_hold_time = 0.1   # How long to hold each key press
        
        # Debug settings
        self._debug_camera_movement = False
        
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
                    elif key_mapping in ["none", "stop", "release_all", "release"]:
                        # Special action: release all movement keys
                        print(f"🛑 MLP Gesture: {actual_gesture} → STOP (releasing all keys)")
                        self.release_all_keys()
                    else:
                        self._handle_special_key(key_mapping, actual_gesture)
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
        """Perform attack action with longer hold for shooting"""
        self.mouse_controller.press(mouse.Button.left)
        time.sleep(self.attack_hold_duration)  # Hold for proper shooting duration
        self.mouse_controller.release(mouse.Button.left)
        print(f"🎮 Action: ATTACK! (Held {self.attack_hold_duration*1000:.0f}ms)")
    
    def release_all_keys(self):
        """Release all currently pressed keys (except voice-controlled ones)"""
        # Don't release voice-controlled sprint
        keys_to_release = [key for key in self.pressed_keys if key != "shift_held"]
        
        for key in keys_to_release:
            try:
                self.keyboard_controller.release(key)
                print(f"🎮 Released: {key.upper()}")
                self.pressed_keys.discard(key)
            except:
                self.pressed_keys.discard(key)
        
        # Only clear non-voice keys
        if "shift_held" in self.pressed_keys:
            print(f"🎤 Keeping voice-controlled sprint active")
    
    def cleanup_voice_held_keys(self):
        """Release all voice-controlled held keys"""
        if not self.voice_held_keys:
            return
        
        print(f"🎤 Releasing {len(self.voice_held_keys)} voice-held keys...")
        
        # Create a copy to avoid modification during iteration
        keys_to_release = list(self.voice_held_keys)
        
        for voice_key_id in keys_to_release:
            # Extract the actual key name (remove 'voice_' prefix)
            key_name = voice_key_id[6:]  # Remove 'voice_' prefix
            print(f"🎤 Emergency release: {key_name}")
            self._handle_voice_hold_key(key_name, hold=False)
        
        print("🎤 All voice-held keys released")
    
    def _handle_voice_hold_key(self, key_name: str, hold: bool):
        """Handle voice command hold/release actions"""
        # Parse the key name to get the actual key
        parsed_key = self._parse_user_friendly_key(key_name) or self._parse_god_of_war_key(key_name)
        
        if not parsed_key:
            # Try direct key name for single characters
            if len(key_name) == 1:
                parsed_key = key_name.lower()
            else:
                print(f"❌ Could not parse hold/release key: {key_name}")
                return
        
        voice_key_id = f"voice_{key_name}"
        
        if hold:
            # Hold the key
            if voice_key_id not in self.voice_held_keys:
                try:
                    if parsed_key == 'left_click':
                        self.mouse_controller.press(mouse.Button.left)
                        print(f"🎤 HOLDING left click")
                    elif parsed_key == 'right_click':
                        self.mouse_controller.press(mouse.Button.right)
                        print(f"🎤 HOLDING right click")
                    elif parsed_key == 'middle_click':
                        self.mouse_controller.press(mouse.Button.middle)
                        print(f"🎤 HOLDING middle click")
                    elif parsed_key == 'shift':
                        self.keyboard_controller.press(Key.shift)
                        print(f"🎤 HOLDING shift")
                    elif parsed_key == 'ctrl':
                        self.keyboard_controller.press(Key.ctrl)
                        print(f"🎤 HOLDING ctrl")
                    elif parsed_key == 'alt':
                        self.keyboard_controller.press(Key.alt)
                        print(f"🎤 HOLDING alt")
                    elif isinstance(parsed_key, str) and len(parsed_key) == 1:
                        self.keyboard_controller.press(parsed_key)
                        print(f"🎤 HOLDING {parsed_key}")
                    else:
                        self.keyboard_controller.press(parsed_key)
                        print(f"🎤 HOLDING {parsed_key}")
                    
                    self.voice_held_keys.add(voice_key_id)
                    print(f"Voice held keys: {self.voice_held_keys}")
                except Exception as e:
                    print(f"❌ Error holding key {key_name}: {e}")
            else:
                print(f"🎤 Key {key_name} already held")
        else:
            # Release the key
            if voice_key_id in self.voice_held_keys:
                try:
                    if parsed_key == 'left_click':
                        self.mouse_controller.release(mouse.Button.left)
                        print(f"🎤 RELEASED left click")
                    elif parsed_key == 'right_click':
                        self.mouse_controller.release(mouse.Button.right)
                        print(f"🎤 RELEASED right click")
                    elif parsed_key == 'middle_click':
                        self.mouse_controller.release(mouse.Button.middle)
                        print(f"🎤 RELEASED middle click")
                    elif parsed_key == 'shift':
                        self.keyboard_controller.release(Key.shift)
                        print(f"🎤 RELEASED shift")
                    elif parsed_key == 'ctrl':
                        self.keyboard_controller.release(Key.ctrl)
                        print(f"🎤 RELEASED ctrl")
                    elif parsed_key == 'alt':
                        self.keyboard_controller.release(Key.alt)
                        print(f"🎤 RELEASED alt")
                    elif isinstance(parsed_key, str) and len(parsed_key) == 1:
                        self.keyboard_controller.release(parsed_key)
                        print(f"🎤 RELEASED {parsed_key}")
                    else:
                        self.keyboard_controller.release(parsed_key)
                        print(f"🎤 RELEASED {parsed_key}")
                    
                    self.voice_held_keys.remove(voice_key_id)
                    print(f"Voice held keys: {self.voice_held_keys}")
                except Exception as e:
                    print(f"❌ Error releasing key {key_name}: {e}")
            else:
                print(f"🎤 Key {key_name} was not held")
    
    def send_voice_action(self, voice_command: Dict):
        """Send input action based on voice command"""
        if not voice_command:
            return
        
        key = voice_command['key']
        name = voice_command['name']
        
        print(f"Executing voice command: {name} -> {key}")
        
        # Check for hold/release prefixes
        if key.startswith('hold_'):
            actual_key = key[5:]  # Remove 'hold_' prefix
            print(f"HOLD command detected: {actual_key}")
            self._handle_voice_hold_key(actual_key, hold=True)
            return
        elif key.startswith('release_'):
            actual_key = key[8:]  # Remove 'release_' prefix
            print(f"RELEASE command detected: {actual_key}")
            self._handle_voice_hold_key(actual_key, hold=False)
            return
        
        # Use existing parsing functions to handle JSON format keys
        # Try parsing the key first
        parsed_key = self._parse_user_friendly_key(key) or self._parse_god_of_war_key(key)
        
        if parsed_key:
            print(f"Parsed key: {key} -> {parsed_key}")
            
            # Handle parsed key results
            if parsed_key == 'left_click':
                # Use longer hold for shooting (same as gesture attack)
                self.mouse_controller.press(mouse.Button.left)
                time.sleep(self.attack_hold_duration)  # Hold for proper shooting duration
                self.mouse_controller.release(mouse.Button.left)
            elif parsed_key == 'right_click':
                self.mouse_controller.click(mouse.Button.right, 1)
            elif parsed_key == 'middle_click':
                self.mouse_controller.click(mouse.Button.middle, 1)
            elif parsed_key == 'mouse_wheel_up':
                self.mouse_controller.scroll(0, 1)  # Scroll up
            elif parsed_key == 'mouse_wheel_down':
                self.mouse_controller.scroll(0, -1)  # Scroll down
            elif parsed_key == 'mouse_wheel':
                # Default to scroll up for weapon switching
                self.mouse_controller.scroll(0, 1)
            elif parsed_key == 'hand_camera_on':
                # Handle hand camera enable via voice
                self._handle_hand_camera_command(True)
            elif parsed_key == 'hand_camera_off':
                # Handle hand camera disable via voice
                self._handle_hand_camera_command(False)
            elif isinstance(parsed_key, str):
                # Single character keys or special string keys
                if parsed_key == 'shift':
                    # Handle sprint toggle - hold/release based on current state
                    self._handle_sprint_toggle(name)
                elif len(parsed_key) == 1:  # Single character keys
                    self.press_and_release_key(parsed_key)
                else:
                    # Handle as combo
                    self._handle_combo_key(parsed_key, name)
            else:
                # pynput Key object (Key.space, Key.ctrl, etc.)
                self.press_and_release_key(parsed_key)
        else:
            # Fallback: Check if it's a combo key with "+"
            if "+" in key:
                print(f"Handling as combo key: {key}")
                self._handle_combo_key(key, name)
            else:
                print(f"⚠️ Could not parse key: {key}")
                # Last resort: try direct execution for single character keys
                if len(key) == 1:
                    self.press_and_release_key(key.lower())
                else:
                    print(f"❌ Failed to execute voice command: {name} -> {key}")
    
    def _handle_combo_key(self, key_combo: str, action_name: str):
        """Handle combo keys for both God of War and custom voice commands"""
        print(f"Processing combo key: '{key_combo}' for action: {action_name}")
        
        # Direct parsing for combo format (both God of War and user input)
        if "+" in key_combo:
            # Handle combo keys like "w + space", "shift + ctrl + r", "Q + Middle Mouse Button"
            parts = [part.strip() for part in key_combo.split("+")]
            keys_to_press = []
            
            for part in parts:
                # Try user-friendly parsing first, then God of War format
                key = self._parse_user_friendly_key(part) or self._parse_god_of_war_key(part)
                if key:
                    keys_to_press.append(key)
                else:
                    print(f"⚠️ Cannot parse key part: {part}")
            
            if keys_to_press:
                print(f"Executing combo: {parts} -> {[str(k) for k in keys_to_press]}")
                self._press_keys_simultaneously(keys_to_press)
            else:
                print(f"❌ Failed to parse combo: {key_combo}")
        else:
            # Single key - try user-friendly first, then God of War format
            key = self._parse_user_friendly_key(key_combo) or self._parse_god_of_war_key(key_combo)
            if key:
                print(f"Executing key: {key_combo} -> {key}")
                if key == "left_click":
                    self.mouse_controller.click(mouse.Button.left, 1)
                elif key == "right_click":
                    self.mouse_controller.click(mouse.Button.right, 1)
                elif key == "middle_click":
                    self.mouse_controller.click(mouse.Button.middle, 1)
                else:
                    self.press_and_release_key(key)
            else:
                print(f"❌ Cannot parse key: {key_combo}")
    
    def _execute_dodge_roll(self):
        """Execute dodge/roll by pressing spacebar + detecting current movement"""
        print("Executing context-aware dodge/roll...")
        
        # Check what movement keys are currently pressed or use default
        current_movement = self._detect_current_movement()
        
        # Press spacebar (dodge) + movement simultaneously
        if current_movement:
            print(f"Dodge rolling in direction: {current_movement}")
            self._press_keys_simultaneously([Key.space, current_movement])
        else:
            # No movement detected - just dodge in place or use forward as default
            print("No movement detected - dodge rolling forward")
            self._press_keys_simultaneously([Key.space, 'w'])
    
    def _detect_current_movement(self):
        """Detect if any movement keys are currently being pressed"""
        # Check if WASD keys are in our pressed_keys set
        movement_keys = {'w': 'w', 'a': 'a', 's': 's', 'd': 'd'}
        
        for key_name, key_value in movement_keys.items():
            if key_name in self.pressed_keys or key_value in self.pressed_keys:
                return key_value
        
        # No movement detected - could add gesture detection here
        return None
    
    def _press_keys_simultaneously(self, keys):
        """Press multiple keys simultaneously then release them - handles mouse buttons"""
        pressed_keys = []
        
        try:
            # Press all keys
            for key in keys:
                if key == "left_click":
                    self.mouse_controller.press(mouse.Button.left)
                    pressed_keys.append(("mouse", mouse.Button.left))
                elif key == "right_click":
                    self.mouse_controller.press(mouse.Button.right) 
                    pressed_keys.append(("mouse", mouse.Button.right))
                elif key == "middle_click":
                    self.mouse_controller.press(mouse.Button.middle)
                    pressed_keys.append(("mouse", mouse.Button.middle))
                elif isinstance(key, str):
                    self.keyboard_controller.press(key)
                    pressed_keys.append(("keyboard", key))
                else:
                    self.keyboard_controller.press(key)
                    pressed_keys.append(("keyboard", key))
                time.sleep(0.01)  # Small delay between presses
            
            # Hold for combo duration
            time.sleep(0.15)  # Hold combo for 150ms
            
            # Release in reverse order
            for device_type, key in reversed(pressed_keys):
                if device_type == "mouse":
                    self.mouse_controller.release(key)
                else:
                    self.keyboard_controller.release(key)
                time.sleep(0.01)
                
            print(f"God of War combo executed: {[str(k) for k in keys]}")
            
        except Exception as e:
            print(f"Error executing God of War combo: {e}")
            # Ensure all keys are released
            for device_type, key in pressed_keys:
                try:
                    if device_type == "mouse":
                        self.mouse_controller.release(key)
                    else:
                        self.keyboard_controller.release(key)
                except:
                    pass
    
    def _parse_key_name(self, key_name: str) -> any:
        """Parse key name to pynput key object"""
        key_name = key_name.lower().strip()
        
        # Special key mappings
        key_mappings = {
            'spacebar': Key.space,
            'space': Key.space,
            'enter': Key.enter,
            'return': Key.enter,
            'shift': Key.shift,
            'ctrl': Key.ctrl,
            'alt': Key.alt,
            'tab': Key.tab,
            'escape': Key.esc,
            'esc': Key.esc,
            'backspace': Key.backspace,
            'delete': Key.delete,
            'up': Key.up,
            'down': Key.down,
            'left': Key.left,
            'right': Key.right,
            'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
            'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8,
            'f9': Key.f9, 'f10': Key.f10, 'f11': Key.f11, 'f12': Key.f12
        }
        
        if key_name in key_mappings:
            return key_mappings[key_name]
        elif len(key_name) == 1:
            return key_name
        else:
            return None
    
    def _parse_user_friendly_key(self, key_name: str) -> any:
        """Parse user-friendly key input to pynput key object"""
        original_key = key_name.strip()
        key_name = key_name.strip().lower()
        
        # Handle Key.mouse.* format from God of War JSON (case-insensitive)
        if original_key.lower().startswith('key.mouse.'):
            mouse_type = original_key.lower().replace('key.mouse.', '')
            if mouse_type == 'left':
                return 'left_click'
            elif mouse_type == 'right':
                return 'right_click'
            elif mouse_type == 'middle':
                return 'middle_click'
        
        # User-friendly key mappings for voice commands
        user_key_mappings = {
            # Common special keys
            'space': Key.space,
            'spacebar': Key.space,
            'enter': Key.enter,
            'return': Key.enter,
            'shift': Key.shift,
            'ctrl': Key.ctrl,
            'control': Key.ctrl,
            'alt': Key.alt,
            'tab': Key.tab,
            'escape': Key.esc,
            'esc': Key.esc,
            'backspace': Key.backspace,
            'delete': Key.delete,
            
            # Arrow keys
            'up': Key.up,
            'down': Key.down,
            'left': Key.left,
            'right': Key.right,
            
            # Mouse buttons
            'left_click': 'left_click',
            'right_click': 'right_click',
            'middle_click': 'middle_click',
            'click': 'left_click',  # Default to left click
            'mouse_left': 'left_click',
            'mouse_right': 'right_click',
            'mouse_middle': 'middle_click',
            'mouse_wheel_click': 'middle_click',
            'wheel_click': 'middle_click',
            
            # Function keys
            'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
            'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8,
            'f9': Key.f9, 'f10': Key.f10, 'f11': Key.f11, 'f12': Key.f12
        }
        
        # Check for user-friendly mappings first
        if key_name in user_key_mappings:
            return user_key_mappings[key_name]
        
        # Single letter/number keys (w, a, s, d, 1, 2, etc.)
        if len(key_name) == 1 and key_name.isalnum():
            return key_name.lower()
        
        return None  # Not found
    
    def _parse_god_of_war_key(self, key_name: str) -> any:
        """Parse God of War key format to pynput key object"""
        key_name = key_name.strip()
        
        # Handle Key.mouse.* format from God of War JSON
        if key_name.startswith('Key.mouse.'):
            mouse_type = key_name.replace('Key.mouse.', '').lower()
            if mouse_type == 'left':
                return 'left_click'
            elif mouse_type == 'right':
                return 'right_click'
            elif mouse_type == 'middle':
                return 'middle_click'
            elif mouse_type == 'middle (or remappable)':
                return 'middle_click'
        
        # Handle Key.* format for other keys
        if key_name.startswith('Key.'):
            key_suffix = key_name.replace('Key.', '').lower()
            if key_suffix == 'shift_l' or key_suffix == 'shift':
                return Key.shift
            elif key_suffix == 'ctrl_l' or key_suffix == 'ctrl':
                return Key.ctrl
            elif key_suffix == 'space':
                return Key.space
            elif key_suffix == 'esc':
                return Key.esc
        
        # God of War specific key mappings (exact format from JSON)
        gow_key_mappings = {
            'Left Click': 'left_click',
            'Right Click': 'right_click', 
            'Middle Mouse Button': 'middle_click',
            'Middle Mouse Button (Press)': 'middle_click',
            'Left Mouse Button': 'left_click',  # Added for voice commands JSON
            'Right Mouse Button': 'right_click',  # Added for voice commands JSON
            'Mouse Wheel Click': 'middle_click',  # Added for mouse wheel button
            'Middle Mouse Wheel': 'middle_click',  # Alternative format
            'Mouse 4 (usually side button)': 'mouse4',
            'Mouse 5 (usually side button)': 'mouse5',
            'Left Shift': Key.shift,
            'Shift': Key.shift,  # Added for voice commands JSON
            'Left CTRL': Key.ctrl,
            'Space': Key.space,
            'Esc': Key.esc
        }
        
        # Check for exact God of War format first
        if key_name in gow_key_mappings:
            return gow_key_mappings[key_name]
        
        # Single letter keys (W, A, S, D, etc.)
        if len(key_name) == 1:
            return key_name.lower()
        
        # Fallback to standard parsing
        return self._parse_key_name(key_name)
    
    def _fallback_combo_execution(self, key_combo: str, action_name: str):
        """Fallback execution for unrecognized combos"""
        print(f"Fallback: Attempting basic execution for '{key_combo}'")
        
        # Try to execute the first recognizable key
        parts = key_combo.replace('+', ' ').split()
        for part in parts:
            parsed = self._parse_key_name(part)
            if parsed:
                print(f"Executing fallback key: {parsed}")
                self.press_and_release_key(parsed)
                return
        
        print(f"Unable to execute combo: '{key_combo}' - no recognizable keys found")
    
    def _llm_parse_combo_keys(self, key_combo: str, action_name: str):
        """Use LLM to parse complex combo keys into simple key names"""
        
        # Check if we have direct access to groq API key
        if hasattr(self, 'groq_api_key') and self.groq_api_key:
            return self._call_groq_for_key_parsing(key_combo, action_name, self.groq_api_key)
        
        # Fallback: try to get through gesture detector connections
        if hasattr(self, 'gesture_detector') and self.gesture_detector:
            if hasattr(self.gesture_detector, 'parent_controller') and self.gesture_detector.parent_controller:
                try:
                    groq_key = getattr(self.gesture_detector.parent_controller, 'groq_api_key', None)
                    if groq_key:
                        return self._call_groq_for_key_parsing(key_combo, action_name, groq_key)
                except:
                    pass
        
        print("No LLM access available for combo parsing")
        return None
    
    def _call_groq_for_key_parsing(self, key_combo: str, action_name: str, groq_api_key: str):
        """Call Groq LLM to parse combo keys"""
        try:
            import requests
            
            prompt = f"""Parse this game control combo into individual key names that can be executed.

COMBO TO PARSE: "{key_combo}"
ACTION: {action_name}

CONTEXT:
- This is from a game control mapping
- Need to convert complex descriptions into simple key names
- "movement key" means choose a default movement direction (use "w" for forward)
- "spacebar" = "space"
- Return ONLY the key names that should be pressed

EXAMPLES:
- "spacebar + movement key" -> ["space", "w"]
- "Ctrl + Shift + R" -> ["ctrl", "shift", "r"] 
- "Left Mouse Button" -> ["left_click"]
- "F + W" -> ["f", "w"]
- "Alt + Tab" -> ["alt", "tab"]

RULES:
1. Convert complex names to simple names (spacebar -> space)
2. For "movement key" use "w" (forward) as default
3. Return as JSON array of strings
4. Use lowercase
5. Mouse buttons: "left_click", "right_click"

OUTPUT FORMAT: ["key1", "key2", ...]

Parse: "{key_combo}"
"""

            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 100
            }
            
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                                   headers=headers, json=data, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Extract JSON array from response
                import json
                try:
                    # Look for JSON array in the response
                    start = content.find('[')
                    end = content.rfind(']') + 1
                    if start != -1 and end != -1:
                        json_str = content[start:end]
                        parsed_keys = json.loads(json_str)
                        print(f"LLM parsed '{key_combo}' -> {parsed_keys}")
                        return parsed_keys
                except json.JSONDecodeError:
                    print(f"LLM response not valid JSON: {content}")
            
            return None
            
        except Exception as e:
            print(f"LLM key parsing failed: {e}")
            return None

    def press_and_release_key(self, key):
        """Press and release a key (for single actions)"""
        try:
            self.keyboard_controller.press(key)
            time.sleep(0.05)  # Brief press
            self.keyboard_controller.release(key)
        except Exception as e:
            print(f"⚠️ Error pressing key {key}: {e}")
    
    def send_camera_movement(self, mouse_dx: float, mouse_dy: float):
        """Send camera movement - game-specific method selection"""
        # Debug output for camera movement
        if hasattr(self, '_debug_camera_movement') and self._debug_camera_movement:
            if abs(mouse_dx) > 0.1 or abs(mouse_dy) > 0.1:
                print(f"🎥 Camera movement: dx={mouse_dx:.2f}, dy={mouse_dy:.2f}")
        
        # Lower threshold for more sensitive movement detection
        if abs(mouse_dx) < 0.1 and abs(mouse_dy) < 0.1:
            # Release camera keys if no movement
            if self.camera_mode == "keyboard":
                self._release_all_camera_keys()
            return
        
        try:
            if self.camera_mode == "keyboard":
                # Use keyboard arrow keys for God of War
                self._send_keyboard_camera(mouse_dx, mouse_dy)
            else:
                # Use mouse movement for other games with sensitivity scaling
                scaled_dx = mouse_dx * self.mouse_sensitivity_x
                scaled_dy = mouse_dy * self.mouse_sensitivity_y
                
                int_dx = int(round(scaled_dx))
                int_dy = int(round(scaled_dy))
                
                if int_dx != 0 or int_dy != 0:
                    pydirectinput.moveRel(int_dx, int_dy)
                
        except Exception as e:
            print(f"Warning: Camera movement error: {e}")
    
    def toggle_camera_movement_debug(self):
        """Toggle camera movement debugging"""
        self._debug_camera_movement = not self._debug_camera_movement
        status = "ON" if self._debug_camera_movement else "OFF"
        print(f"🎥 Camera movement debug: {status}")
        return self._debug_camera_movement
    
    def emergency_stop(self):
        """Emergency stop - release everything"""
        self.release_all_keys()
        # Release camera keys for keyboard mode
        self._release_all_camera_keys()
        # Also release aiming key if aiming
        if self.is_aiming_with_wink:
            try:
                if self.current_game == "god of war":
                    self.keyboard_controller.release(Key.ctrl_l)
                    print("👁️ Released Left Ctrl (emergency stop)")
                else:
                    self.mouse_controller.release(mouse.Button.right)
                    print("👁️ Released right mouse button (emergency stop)")
                self.is_aiming_with_wink = False
            except:
                pass
        print("🛑 EMERGENCY STOP - All keys released!")
    
    def set_gesture_detector(self, detector):
        """Set reference to gesture detector for hand camera control"""
        self.gesture_detector = detector
    
    def set_game_mode(self, game_name: str):
        """Set camera control mode based on game"""
        self.current_game = game_name.lower()
        
        # Games that need keyboard camera control
        keyboard_camera_games = [
            "god of war",
            "god_of_war", 
            "godofwar",
            "gow"
        ]
        
        # Check if current game needs keyboard camera
        needs_keyboard = any(game in self.current_game for game in keyboard_camera_games)
        
        if needs_keyboard:
            self.camera_mode = "keyboard"
            print(f"🎮 Camera Mode: KEYBOARD (for {game_name})")
            print("   Using arrow keys for camera control due to game compatibility")
        else:
            self.camera_mode = "mouse"
            print(f"🎮 Camera Mode: MOUSE (for {game_name})")
        
        # Release any currently pressed camera keys when switching modes
        self._release_all_camera_keys()
    
    def _release_all_camera_keys(self):
        """Release all camera control keys"""
        camera_keys = ['left', 'right', 'up', 'down']
        for key_name in camera_keys:
            if key_name in self.camera_keys_pressed:
                try:
                    if key_name == 'left':
                        self.keyboard_controller.release(Key.left)
                    elif key_name == 'right':
                        self.keyboard_controller.release(Key.right)
                    elif key_name == 'up':
                        self.keyboard_controller.release(Key.up)
                    elif key_name == 'down':
                        self.keyboard_controller.release(Key.down)
                except:
                    pass
        self.camera_keys_pressed.clear()
    
    def _send_keyboard_camera(self, dx: float, dy: float):
        """Send camera movement using keyboard arrow keys - God of War optimized"""
        import time
        
        # Apply deadzone - higher for less sensitivity
        if abs(dx) < self.camera_key_deadzone and abs(dy) < self.camera_key_deadzone:
            return
        
        # Use PULSE system instead of hold - send brief key presses
        keys_to_pulse = []
        
        # Scale movement to determine pulse strength
        movement_strength = max(abs(dx), abs(dy))
        pulse_count = max(1, int(movement_strength / 15))  # Less pulses = less sensitive
        
        # Determine direction
        if abs(dx) > self.camera_key_deadzone:
            if dx > 0:
                keys_to_pulse.append(('right', Key.right))
            else:
                keys_to_pulse.append(('left', Key.left))
                
        if abs(dy) > self.camera_key_deadzone:
            if dy > 0:
                keys_to_pulse.append(('down', Key.down))
            else:
                keys_to_pulse.append(('up', Key.up))
        
        # Send quick pulses instead of holding
        for key_name, key_obj in keys_to_pulse:
            try:
                # Quick press and release - much less sensitive than holding
                self.keyboard_controller.press(key_obj)
                time.sleep(self.camera_key_hold_time)  # Brief hold
                self.keyboard_controller.release(key_obj)
                
            except Exception as e:
                print(f"Warning: Camera key pulse error ({key_name}): {e}")
        
        # Show debug info occasionally (less spam)
        if keys_to_pulse and abs(dx) > 15:  # Only show for significant movement
            key_names = [k[0] for k in keys_to_pulse]
            print(f"🎮 GoW Camera: {'/'.join(key_names)} (movement={movement_strength:.1f})")
    
    def _handle_hand_camera_command(self, enable: bool):
        """Handle hand camera on/off voice commands"""
        if self.gesture_detector:
            self.gesture_detector.hand_camera_enabled = enable
            status = "ENABLED" if enable else "DISABLED"
            print(f"🎤👋 Hand camera {status} via voice command")
        else:
            print("❌ Could not control hand camera - no detector reference")
    
    def _handle_special_key(self, key_mapping: str, gesture_name: str):
        """Handle special keys from LLM mappings"""
        gesture_key = f"MLP:{gesture_name}"
        current_time = time.time()
        
        # Apply cooldown
        if gesture_key in self.last_action_time:
            if current_time - self.last_action_time[gesture_key] < self.action_cooldown:
                return
        self.last_action_time[gesture_key] = current_time
        
        # Handle special key mappings
        key_mapping_lower = key_mapping.lower()
        
        if key_mapping_lower == "ctrl":
            self.press_and_release_key(Key.ctrl)
            print(f"🎮 Action: {gesture_name} -> CTRL")
        elif key_mapping_lower == "shift":
            # Sprint is a hold action - press and hold while gesture is active
            if Key.shift not in self.pressed_keys:
                self.keyboard_controller.press(Key.shift)
                self.pressed_keys.add(Key.shift)
                print(f"🎮 Action: {gesture_name} -> SHIFT (HOLD)")
        elif key_mapping_lower == "alt":
            self.press_and_release_key(Key.alt)
            print(f"🎮 Action: {gesture_name} -> ALT")
        elif key_mapping_lower == "tab":
            self.press_and_release_key(Key.tab)
            print(f"🎮 Action: {gesture_name} -> TAB")
        elif key_mapping_lower == "esc":
            self.press_and_release_key(Key.esc)
            print(f"🎮 Action: {gesture_name} -> ESC")
        elif key_mapping_lower == "mouse wheel" or key_mapping_lower == "mouse_wheel":
            # Mouse wheel scroll for weapon switching
            self.mouse_controller.scroll(0, 1)  # Scroll up
            print(f"🎮 Action: {gesture_name} -> MOUSE WHEEL UP")
        elif key_mapping_lower == "mouse wheel up" or key_mapping_lower == "mouse_wheel_up":
            self.mouse_controller.scroll(0, 1)
            print(f"🎮 Action: {gesture_name} -> MOUSE WHEEL UP")
        elif key_mapping_lower == "mouse wheel down" or key_mapping_lower == "mouse_wheel_down":
            self.mouse_controller.scroll(0, -1)
            print(f"🎮 Action: {gesture_name} -> MOUSE WHEEL DOWN")
        elif len(key_mapping) == 1:
            # Single character keys (q, r, e, f, etc.)
            self.press_and_release_key(key_mapping_lower)
            print(f"🎮 Action: {gesture_name} -> {key_mapping.upper()}")
        else:
            print(f"⚠️ Unsupported key mapping: {key_mapping}")
    
    def send_wink_action(self, is_aiming: bool, wink_type: str, confidence: float):
        """Handle wink detection for aiming with smoothing - use dynamic aim key from voice commands"""
        current_time = time.time()
        
        # Track confidence history for smoothing
        self.aim_confidence_history.append(confidence)
        if len(self.aim_confidence_history) > self.max_confidence_history:
            self.aim_confidence_history.pop(0)
        
        # Calculate average confidence over recent frames
        avg_confidence = sum(self.aim_confidence_history) / len(self.aim_confidence_history) if self.aim_confidence_history else 0
        
        # Use stricter criteria: either individual confidence is good OR average is stable
        confidence_ok = confidence >= 0.5 or avg_confidence >= 0.4
        
        if is_aiming and confidence_ok and wink_type != "none":
            # Start aiming if not already aiming
            if not self.is_aiming_with_wink:
                current_aim_key = self.get_current_aim_key()
                
                if isinstance(current_aim_key, mouse.Button):
                    self.mouse_controller.press(current_aim_key)
                    print(f"🎯 Started aiming with {wink_type} ({current_aim_key} held)")
                else:
                    self.keyboard_controller.press(current_aim_key)
                    print(f"🎯 Started aiming with {wink_type} ({current_aim_key} held)")
                
                self.is_aiming_with_wink = True
                self.last_aim_change_time = current_time
        
        elif not is_aiming or not confidence_ok or wink_type == "none":
            # Stop aiming - but with delay to prevent jitter
            if self.is_aiming_with_wink:
                # Add release delay to prevent jittery releases
                time_since_change = current_time - self.last_aim_change_time
                
                # Only release if we've been in this state for the delay period
                if time_since_change >= self.aim_release_delay:
                    current_aim_key = self.get_current_aim_key()
                    
                    if isinstance(current_aim_key, mouse.Button):
                        self.mouse_controller.release(current_aim_key)
                        print(f"🎯 Stopped aiming ({current_aim_key} released) - delay: {time_since_change:.2f}s")
                    else:
                        self.keyboard_controller.release(current_aim_key)
                        print(f"🎯 Stopped aiming ({current_aim_key} released) - delay: {time_since_change:.2f}s")
                    
                    self.is_aiming_with_wink = False
                    self.last_aim_change_time = current_time
                # else: Still in release delay period, keep aiming active
    
    def adjust_aim_smoothing(self, increase_delay: bool = True):
        """Adjust aim release delay for smoother or more responsive aiming"""
        if increase_delay:
            self.aim_release_delay = min(1.0, self.aim_release_delay + 0.1)  # Max 1 second
            print(f"🎯 Increased aim smoothing: {self.aim_release_delay:.1f}s delay")
        else:
            self.aim_release_delay = max(0.1, self.aim_release_delay - 0.1)  # Min 0.1 second
            print(f"🎯 Decreased aim smoothing: {self.aim_release_delay:.1f}s delay")
        
        print(f"Current aim release delay: {self.aim_release_delay:.1f}s")
        return self.aim_release_delay

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
    
    def _handle_sprint_toggle(self, command_name: str):
        """Handle sprint toggle based on voice command"""
        command_lower = command_name.lower()
        
        # Commands that start sprinting
        sprint_start_commands = ['sprint', 'run', 'faster', 'speed up']
        # Commands that stop sprinting  
        sprint_stop_commands = ['walk', 'stop sprinting', 'slow down', 'normal speed']
        
        print(f"🔍 Sprint Debug: command_name='{command_name}', command_lower='{command_lower}'")
        print(f"🔍 Current pressed_keys: {self.pressed_keys}")
        print(f"🔍 Looking for 'shift_held' in pressed_keys: {'shift_held' in self.pressed_keys}")
        
        if any(cmd in command_lower for cmd in sprint_start_commands):
            # Start sprinting - hold shift
            if "shift_held" not in self.pressed_keys:
                try:
                    self.keyboard_controller.press(Key.shift)
                    self.pressed_keys.add("shift_held")  # Use string for tracking
                    print(f"🎤🏃 Started sprinting (SHIFT HELD) - try typing to test!")
                except Exception as e:
                    print(f"❌ Error holding shift: {e}")
            else:
                print(f"🎤🏃 Already sprinting")
                
        elif any(cmd in command_lower for cmd in sprint_stop_commands):
            # Stop sprinting - release shift
            if "shift_held" in self.pressed_keys:
                try:
                    self.keyboard_controller.release(Key.shift)
                    self.pressed_keys.remove("shift_held")
                    print(f"🎤🚶 Stopped sprinting (SHIFT RELEASED)")
                except Exception as e:
                    print(f"❌ Error releasing shift: {e}")
            else:
                print(f"🎤🚶 Already walking")
        else:
            # Default toggle behavior
            if "shift_held" in self.pressed_keys:
                try:
                    self.keyboard_controller.release(Key.shift)
                    self.pressed_keys.remove("shift_held")
                    print(f"🎤🚶 Sprint OFF (SHIFT RELEASED)")
                except Exception as e:
                    print(f"❌ Error releasing shift: {e}")
            else:
                try:
                    self.keyboard_controller.press(Key.shift)
                    self.pressed_keys.add("shift_held")
                    print(f"🎤🏃 Sprint ON (SHIFT HELD) - try typing to test!")
                except Exception as e:
                    print(f"❌ Error holding shift: {e}")
    
    def adjust_attack_duration(self, increase: bool = True):
        """Adjust attack hold duration for different games"""
        if increase:
            self.attack_hold_duration = min(0.5, self.attack_hold_duration + 0.05)  # Max 500ms
        else:
            self.attack_hold_duration = max(0.05, self.attack_hold_duration - 0.05)  # Min 50ms
        
        print(f"🎯 Attack hold duration: {self.attack_hold_duration*1000:.0f}ms")
        return self.attack_hold_duration
    
    def adjust_camera_sensitivity(self, increase: bool = True):
        """Adjust camera sensitivity for current mode"""
        if self.camera_mode == "keyboard":
            # Adjust keyboard camera settings for God of War
            if increase:
                self.camera_key_deadzone = max(2.0, self.camera_key_deadzone - 2.0)  # Lower deadzone = more sensitive
                self.camera_key_hold_time = min(0.2, self.camera_key_hold_time + 0.02)  # Longer hold = more movement
            else:
                self.camera_key_deadzone = min(20.0, self.camera_key_deadzone + 2.0)  # Higher deadzone = less sensitive
                self.camera_key_hold_time = max(0.05, self.camera_key_hold_time - 0.02)  # Shorter hold = less movement
            
            print(f"🎮 GoW Camera Sensitivity: Deadzone={self.camera_key_deadzone:.1f}, Hold={self.camera_key_hold_time:.3f}s")
        else:
            # Adjust mouse sensitivity for normal mode games
            if increase:
                self.mouse_sensitivity_x = min(3.0, self.mouse_sensitivity_x + 0.1)  # Max 3x sensitivity
                self.mouse_sensitivity_y = min(3.0, self.mouse_sensitivity_y + 0.1)
            else:
                self.mouse_sensitivity_x = max(0.1, self.mouse_sensitivity_x - 0.1)  # Min 0.1x sensitivity
                self.mouse_sensitivity_y = max(0.1, self.mouse_sensitivity_y - 0.1)
            
            print(f"🖱️ Mouse Camera Sensitivity: {self.mouse_sensitivity_x:.1f}x (horizontal), {self.mouse_sensitivity_y:.1f}x (vertical)")
        
        return self.camera_key_deadzone, self.camera_key_hold_time

    def detect_and_set_aim_key(self, voice_commands_dict):
        """Detect aim key from voice commands and set it for wink mapping"""
        import re
        
        aim_key = None
        
        # Search for aim-related commands using regex
        aim_patterns = [
            r'aim',
            r'sight',
            r'scope',
            r'target',
            r'zoom',
            r'focus',
            r'ads'  # Aim Down Sights
        ]
        
        print("Detecting aim key from voice commands...")
        
        for action_name, config in voice_commands_dict.items():
            # Check if action name matches aim patterns
            action_lower = action_name.lower()
            for pattern in aim_patterns:
                if re.search(pattern, action_lower):
                    aim_key = config.get('key', None)
                    print(f"Found aim command: '{action_name}' -> key: '{aim_key}'")
                    break
            
            # Also check intents for aim-related words
            if not aim_key and 'intents' in config:
                for intent in config['intents']:
                    intent_lower = intent.lower()
                    for pattern in aim_patterns:
                        if re.search(pattern, intent_lower):
                            aim_key = config.get('key', None)
                            print(f"Found aim intent: '{intent}' in '{action_name}' -> key: '{aim_key}'")
                            break
                    if aim_key:
                        break
            
            if aim_key:
                break
        
        if aim_key:
            # Parse the aim key to get the proper format
            parsed_aim_key = self._parse_user_friendly_key(aim_key) or self._parse_god_of_war_key(aim_key)
            
            if parsed_aim_key:
                # Convert string results to proper key objects
                if parsed_aim_key == 'right_click':
                    self.dynamic_aim_key = mouse.Button.right
                    print(f"Dynamic aim key set to: Right Mouse Button")
                elif parsed_aim_key == 'left_click':
                    self.dynamic_aim_key = mouse.Button.left
                    print(f"Dynamic aim key set to: Left Mouse Button")
                elif isinstance(parsed_aim_key, str) and len(parsed_aim_key) == 1:
                    # Single character key
                    self.dynamic_aim_key = parsed_aim_key
                    print(f"Dynamic aim key set to: {parsed_aim_key.upper()}")
                else:
                    # pynput Key object
                    self.dynamic_aim_key = parsed_aim_key
                    print(f"Dynamic aim key set to: {parsed_aim_key}")
            else:
                print(f"Could not parse aim key: {aim_key}")
        else:
            print("No aim command found in voice commands - using default mapping")
            self.dynamic_aim_key = None
        
        return self.dynamic_aim_key
    
    def get_current_aim_key(self):
        """Get the current aim key (dynamic or fallback to game-specific)"""
        if self.dynamic_aim_key:
            return self.dynamic_aim_key
        else:
            # Fallback to game-specific mapping
            return self.wink_mappings.get(self.current_game, self.wink_mappings["default"])
    
    def cleanup_wink_aiming(self):
        """Clean up wink aiming state on shutdown"""
        if self.is_aiming_with_wink:
            try:
                current_aim_key = self.get_current_aim_key()
                
                if isinstance(current_aim_key, mouse.Button):
                    self.mouse_controller.release(current_aim_key)
                    print(f"Cleaned up wink aiming ({current_aim_key})")
                else:
                    self.keyboard_controller.release(current_aim_key)
                    print(f"Cleaned up wink aiming ({current_aim_key})")
                    
                self.is_aiming_with_wink = False
            except:
                pass