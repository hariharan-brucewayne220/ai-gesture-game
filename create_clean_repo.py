#!/usr/bin/env python3
"""
Create Clean Repository
Creates a new clean repository with only essential files
This is the safest way to avoid large file issues
"""

import os
import shutil
import sys
from pathlib import Path

def create_clean_repo():
    """Create a clean repository with only essential files"""
    
    print("="*60)
    print("CREATING CLEAN REPOSITORY")
    print("="*60)
    
    # Create clean directory
    clean_dir = "../ai-gesture-gaming-clean"
    
    if os.path.exists(clean_dir):
        print(f"Removing existing clean directory: {clean_dir}")
        shutil.rmtree(clean_dir)
    
    print(f"Creating clean directory: {clean_dir}")
    os.makedirs(clean_dir)
    
    # Essential files and directories to copy
    essential_items = [
        # Core source code
        "src/",
        
        # RAG system
        "game_rag/",
        
        # Configuration
        "config/",
        "custom_controller/",
        
        # PDF guides (small files)
        "guides/",
        
        # Main scripts
        "query_rag.py",
        "demo_rag_usage.py",
        "save_model.py",
        "setup_rag_env.py",
        
        # Utilities
        "debug_audio.py",
        "fix_microphone.py", 
        "fix_microphone_input.py",
        "install_packages.py",
        "run_original.py",
        "release_shift.py",
        
        # Documentation
        "README.md",
        "requirements.txt",
        ".gitignore",
        
        # All markdown files
        "*.md",
        
        # Configuration files
        "voice_commands.json",
        "trained_gestures.json",
    ]
    
    # Copy essential files
    for item in essential_items:
        if item.endswith("*"):
            # Handle wildcards
            import glob
            pattern = item
            files = glob.glob(pattern)
            for file in files:
                if os.path.isfile(file):
                    print(f"Copying: {file}")
                    shutil.copy2(file, clean_dir)
        elif os.path.exists(item):
            dest_path = os.path.join(clean_dir, item)
            if os.path.isdir(item):
                print(f"Copying directory: {item}")
                shutil.copytree(item, dest_path, ignore=shutil.ignore_patterns(
                    '__pycache__', '*.pyc', '*.pyo', '.git', 'chroma_db'
                ))
            else:
                print(f"Copying file: {item}")
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(item, dest_path)
        else:
            print(f"Skipping (not found): {item}")
    
    # Copy important single files that might not be in patterns
    important_files = [
        "thumbsdown.png",
        "for_claude.png"
    ]
    
    for file in important_files:
        if os.path.exists(file):
            print(f"Copying: {file}")
            shutil.copy2(file, clean_dir)
    
    print(f"\n✓ Clean repository created in: {clean_dir}")
    print(f"Total size of clean repo:")
    
    # Calculate size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(clean_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    
    size_mb = total_size / (1024 * 1024)
    print(f"  {size_mb:.2f} MB")
    
    # Create initialization script
    init_script = f"""#!/bin/bash
# Initialize clean repository

cd {clean_dir}

# Initialize git
git init
git add .
git commit -m "Initial commit - Clean AI Gesture Gaming System

Complete AI gaming control system with:
- Gesture control using MediaPipe
- Voice commands with multiple engine support  
- RAG system with PDF guide integration
- Interactive CLI interface
- 1,297 documents across 20+ games
- Professional documentation and setup

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"

echo "Repository initialized!"
echo "Next steps:"
echo "1. Create new GitHub repository"
echo "2. git remote add origin <your-new-repo-url>"
echo "3. git push -u origin main"
echo ""
echo "Remember to:"
echo "- Download vosk_models/ separately"
echo "- Download whisper.cpp/ separately"
echo "- Set up virtual environments as per README"
"""
    
    with open(os.path.join(clean_dir, "initialize_repo.sh"), "w") as f:
        f.write(init_script)
    
    # Create setup instructions
    setup_instructions = """# Clean Repository Setup Instructions

## What's Included
This clean repository contains all essential files:
- Complete source code (src/)
- RAG system (game_rag/)
- Configuration files
- PDF guides
- Documentation
- Utility scripts

## What's NOT Included (download separately)
- vosk_models/ (speech recognition models)
- whisper.cpp/ (speech processing)
- mediapipe/ (large dependency)
- Virtual environments (create fresh)

## Setup Steps

### 1. Initialize Git Repository
```bash
cd ai-gesture-gaming-clean
chmod +x initialize_repo.sh
./initialize_repo.sh
```

### 2. Create GitHub Repository
1. Go to GitHub and create new repository
2. Don't initialize with README (we have one)
3. Copy the repository URL

### 3. Connect and Push
```bash
git remote add origin <your-repo-url>
git push -u origin main
```

### 4. Set Up Environments
Follow the README.md instructions to:
- Create mp_env for gesture/voice control
- Create rag_env for RAG system
- Install dependencies

### 5. Download Large Models (if needed)
- Vosk models for voice recognition
- Whisper.cpp for advanced speech processing
- MediaPipe will install automatically with pip

## Repository Size
The clean repository is approximately 2-5 MB instead of 200+ MB.
This ensures fast cloning and no GitHub size limits.

## Benefits
- Fast clone/download times
- No large file warnings from GitHub
- Clean git history
- Professional appearance
- All functionality preserved
"""
    
    with open(os.path.join(clean_dir, "SETUP_INSTRUCTIONS.md"), "w") as f:
        f.write(setup_instructions)
    
    print("\n" + "="*60)
    print("CLEAN REPOSITORY READY!")
    print("="*60)
    print(f"Location: {os.path.abspath(clean_dir)}")
    print(f"Size: {size_mb:.2f} MB")
    print()
    print("Next steps:")
    print(f"1. cd {clean_dir}")
    print("2. Run: ./initialize_repo.sh")
    print("3. Create new GitHub repository")
    print("4. Push to GitHub")
    print()
    print("See SETUP_INSTRUCTIONS.md for detailed steps")

if __name__ == "__main__":
    create_clean_repo()