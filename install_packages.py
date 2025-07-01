#!/usr/bin/env python3
"""
Automatic package installer for AI Gesture Gaming Controller
This will try different methods to install required packages
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {command}")
            return True
        else:
            print(f"❌ Failed: {command}")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception running {command}: {e}")
        return False

def install_package(package_name):
    """Try multiple methods to install a package"""
    print(f"\n📦 Installing {package_name}...")
    
    # Method 1: Try pip
    if run_command(f"pip install {package_name}"):
        return True
    
    # Method 2: Try pip3
    if run_command(f"pip3 install {package_name}"):
        return True
    
    # Method 3: Try python -m pip
    if run_command(f"python3 -m pip install {package_name}"):
        return True
    
    # Method 4: Try with --user flag
    if run_command(f"python3 -m pip install --user {package_name}"):
        return True
    
    print(f"❌ All installation methods failed for {package_name}")
    return False

def main():
    print("🚀 AI Gesture Gaming Controller - Package Installer")
    print("=" * 60)
    
    packages = [
        "numpy",           # Install numpy first (dependency for others)
        "opencv-python",   # Computer vision
        "mediapipe",       # Hand tracking
        "pynput",          # Input simulation
        "pyautogui",       # Alternative input method
        "Pillow"           # Image processing
    ]
    
    failed_packages = []
    
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print("\n" + "=" * 60)
    
    if not failed_packages:
        print("🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
        print("\n🧪 Testing installation...")
        os.system("python3 test_installation.py")
    else:
        print(f"❌ Failed to install: {', '.join(failed_packages)}")
        print("\n🔧 Manual installation required:")
        for package in failed_packages:
            print(f"   pip install {package}")
        
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure you have internet connection")
        print("2. Try running as administrator/sudo")
        print("3. Update pip: python3 -m pip install --upgrade pip")
        print("4. Install packages one by one manually")

if __name__ == "__main__":
    main()