#!/usr/bin/env python3
"""
Test script to verify all required packages are installed
Run this first before testing the gesture controller
"""

def test_package(package_name, import_name=None):
    """Test if a package can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} installed and working")
        return True
    except ImportError as e:
        print(f"❌ {package_name} missing - Error: {e}")
        print(f"   Install with: pip install {package_name}")
        return False

def main():
    print("🔍 Testing AI Gesture Gaming Controller Dependencies...")
    print("=" * 60)
    
    required_packages = [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
        ("pynput", "pynput"),
        ("pyautogui", "pyautogui"),
        ("Pillow", "PIL")
    ]
    
    all_installed = True
    
    for package, import_name in required_packages:
        if not test_package(package, import_name):
            all_installed = False
    
    print("=" * 60)
    
    if all_installed:
        print("🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
        print("🚀 Ready to run the gesture controller!")
        print("\nNext steps:")
        print("1. Test camera: python tests/test_camera.py")
        print("2. Run controller: python src/main.py")
    else:
        print("❌ MISSING PACKAGES DETECTED")
        print("\nInstall missing packages with:")
        print("pip install opencv-python mediapipe numpy pynput pyautogui Pillow")
        print("\nOr install individually:")
        print("pip install [package-name]")
    
    print("\n🔧 System Info:")
    import sys
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

if __name__ == "__main__":
    main()