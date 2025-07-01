import time
from pynput import keyboard, mouse
from pynput.keyboard import Key, Listener
import threading
from typing import Dict, Set

class InputController:
    def __init__(self):
        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = mouse.Controller()
        
        # Currently pressed keys (to avoid spam)
        self.pressed_keys: Set[str] = set()
        
        # DIRECTIONAL POINTING GESTURE MAPPINGS
        self.gesture_map = {
            "open_palm": "w",        # Forward
            "closed_fist": "s",      # Backward  
            "point_left": "a",       # Strafe Left (Index pointing left)
            "point_right": "d",      # Strafe Right (Index pointing right)
            "point_up": Key.space,   # Jump (Index pointing up)
            "rock_sign": "attack"    # Attack (Index + pinky)
        }
        
        
        # Key press timing
        self.last_action_time = {}
        self.action_cooldown = 0.1  # 100ms cooldown between actions
        
    def send_action(self, gesture: str, confidence: float):
        """Send input action based on gesture"""
        if gesture == "none" or confidence < 0.7:
            self.release_all_keys()
            return
            
        action = self.gesture_map.get(gesture)
        if not action:
            return
            
        # Check cooldown
        current_time = time.time()
        if gesture in self.last_action_time:
            if current_time - self.last_action_time[gesture] < self.action_cooldown:
                return
                
        self.last_action_time[gesture] = current_time
        
        # Special handling for different action types
        if action == "attack":
            self.attack()
        elif isinstance(action, str):
            self.press_key(action, gesture)
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
    
    def emergency_stop(self):
        """Emergency stop - release everything"""
        self.release_all_keys()
        print("🛑 EMERGENCY STOP - All keys released!")