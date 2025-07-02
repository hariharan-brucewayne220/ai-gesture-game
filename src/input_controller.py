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
            "rock_sign": "attack"    # Attack (Index + pinky)
        }
        
        
        # Key press timing
        self.last_action_time = {}
        self.action_cooldown = 0.1  # 100ms cooldown between actions
        
        # Camera movement timing
        self.last_camera_move = time.time()
        self.camera_cooldown = 0.01  # 10ms between camera moves
        
        # Configure pydirectinput for camera movement
        pydirectinput.FAILSAFE = False
        pydirectinput.PAUSE = 0
        
    def send_action(self, gesture: str, confidence: float):
        """Send input action based on gesture"""
        if gesture == "none" or confidence < 0.7:
            self.release_all_keys()
            return
            
        action = self.gesture_map.get(gesture)
        if not action:
            return
            
        # Movement keys (W,A,S,D) - continuous hold, no cooldown needed
        movement_keys = ["w", "a", "s", "d"]
        
        # Action keys (Jump, Attack) - single press with cooldown
        if action == "attack" or action == Key.space:
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
        elif isinstance(action, str) and action in movement_keys:
            # Execute continuous movement (no cooldown)
            self.press_key(action, gesture)
    
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
    
    def send_camera_movement(self, mouse_dx: int, mouse_dy: int):
        """Send camera movement using pydirectinput"""
        if mouse_dx == 0 and mouse_dy == 0:
            return
        
        current_time = time.time()
        if current_time - self.last_camera_move > self.camera_cooldown:
            try:
                pydirectinput.moveRel(mouse_dx, mouse_dy)
                self.last_camera_move = current_time
            except Exception as e:
                print(f"⚠️ Camera movement error: {e}")
    
    def emergency_stop(self):
        """Emergency stop - release everything"""
        self.release_all_keys()
        print("🛑 EMERGENCY STOP - All keys released!")