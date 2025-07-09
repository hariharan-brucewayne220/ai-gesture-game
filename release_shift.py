#!/usr/bin/env python3
"""Quick script to release stuck shift key"""

try:
    import pynput.keyboard as kb
    keyboard = kb.Controller()
    
    # Release both shift keys
    keyboard.release(kb.Key.shift_l)
    keyboard.release(kb.Key.shift_r)
    print("Released both left and right shift keys")
    
except ImportError:
    print("pynput not available, trying alternative...")
    try:
        import keyboard
        keyboard.release('shift')
        print("Released shift key using keyboard module")
    except ImportError:
        print("Neither pynput nor keyboard module available")
        print("Try manually pressing and releasing shift key")

except Exception as e:
    print(f"Error releasing shift key: {e}")
    print("Try manually pressing and releasing shift key")