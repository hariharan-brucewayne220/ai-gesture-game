#!/usr/bin/env python3
"""
Git History Cleanup Script
Safely removes large files from git history while preserving all our work
"""

import os
import subprocess
import shutil
import sys

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n{description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Success")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"✗ Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

def backup_current_work():
    """Create a backup of current work"""
    print("\n" + "="*60)
    print("STEP 1: BACKING UP CURRENT WORK")
    print("="*60)
    
    backup_dir = "../ai-gesture-gaming-backup"
    
    if os.path.exists(backup_dir):
        print(f"Removing existing backup: {backup_dir}")
        shutil.rmtree(backup_dir)
    
    print(f"Creating backup: {backup_dir}")
    shutil.copytree(".", backup_dir, ignore=shutil.ignore_patterns(
        '.git', '__pycache__', '*.pyc', 'gesture-env', 'mp_env', 'rag_env',
        'vosk_models', 'whisper.cpp', 'mediapipe', 'backups'
    ))
    print("✓ Backup created successfully")

def commit_current_changes():
    """Commit all our current work"""
    print("\n" + "="*60)
    print("STEP 2: COMMITTING CURRENT WORK")
    print("="*60)
    
    # Add only essential files
    essential_files = [
        ".gitignore",
        "README.md", 
        "requirements.txt",
        "src/",
        "game_rag/",
        "query_rag.py",
        "demo_rag_usage.py", 
        "save_model.py",
        "setup_rag_env.py",
        "guides/",
        "custom_controller/",
        "config/",
        "*.md"
    ]
    
    for file_pattern in essential_files:
        run_command(f"git add {file_pattern}", f"Adding {file_pattern}")
    
    # Commit changes
    commit_msg = """Complete RAG system integration and cleanup

- Added comprehensive RAG system with PDF integration
- Created interactive CLI interface for game assistance
- Integrated God of War PDF guide with 1,265 document chunks
- Updated documentation and requirements
- Cleaned up temporary and test files
- Fixed Unicode encoding issues for Windows compatibility

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"""
    
    run_command(f'git commit -m "{commit_msg}"', "Committing current work")

def clean_git_history():
    """Clean git history using git filter-repo"""
    print("\n" + "="*60)
    print("STEP 3: CLEANING GIT HISTORY")
    print("="*60)
    
    # Create list of large files to remove from history
    large_files = [
        "vosk_models/",
        "whisper.cpp/", 
        "mediapipe/",
        "gesture-env/",
        "mp_env/",
        "rag_env/",
        "backups/",
        "*.dll",
        "*.pyd",
        "*.so",
        "*.dylib",
        "*.pkl",
        "*.bin"
    ]
    
    print("Files/patterns to remove from history:")
    for file in large_files:
        print(f"  - {file}")
    
    # Use git filter-repo if available, otherwise git filter-branch
    filter_repo_available = run_command("git filter-repo --help > /dev/null 2>&1", "Checking for git filter-repo")
    
    if filter_repo_available:
        print("\nUsing git filter-repo (recommended)")
        
        # Build filter-repo command
        cmd_parts = ["git filter-repo"]
        for file in large_files:
            cmd_parts.append(f"--path-glob '{file}' --invert-paths")
        
        cmd = " ".join(cmd_parts)
        run_command(cmd, "Removing large files from history")
        
    else:
        print("\ngit filter-repo not available, using alternative method")
        print("Installing git filter-repo...")
        
        # Try to install git filter-repo
        install_success = run_command("pip install git-filter-repo", "Installing git-filter-repo")
        
        if install_success:
            # Retry with filter-repo
            cmd_parts = ["git filter-repo"]
            for file in large_files:
                cmd_parts.append(f"--path-glob '{file}' --invert-paths")
            
            cmd = " ".join(cmd_parts)
            run_command(cmd, "Removing large files from history")
        else:
            print("⚠️  Could not install git filter-repo")
            print("Manual cleanup required - see instructions below")

def verify_cleanup():
    """Verify the cleanup worked"""
    print("\n" + "="*60)
    print("STEP 4: VERIFYING CLEANUP")
    print("="*60)
    
    # Check repository size
    run_command("du -sh .git", "Checking .git folder size")
    
    # Check for large files
    run_command("git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '/^blob/ {print substr($0,6)}' | sort --numeric-sort --key=2 | tail -10", "Checking for remaining large files")
    
    # Show current status
    run_command("git status", "Checking git status")

def create_instructions():
    """Create instructions for manual cleanup if needed"""
    instructions = """
# Manual Git History Cleanup Instructions

If the automatic cleanup didn't work, here's how to do it manually:

## Option 1: Fresh Repository (Recommended)
1. Create a new repository on GitHub
2. Copy only essential files to a new folder:
   - src/
   - game_rag/
   - guides/
   - config/
   - custom_controller/
   - README.md
   - requirements.txt
   - query_rag.py
   - .gitignore

3. Initialize new git repo and push:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Clean AI Gesture Gaming System"
   git remote add origin <new-repo-url>
   git push -u origin main
   ```

## Option 2: BFG Repo Cleaner
1. Download BFG: https://rtyley.github.io/bfg-repo-cleaner/
2. Run: java -jar bfg.jar --delete-folders "{vosk_models,whisper.cpp,mediapipe,gesture-env,mp_env,rag_env}" .
3. Run: git reflog expire --expire=now --all && git gc --prune=now --aggressive

## Option 3: Force Push (Use with caution)
```bash
git push --force-with-lease origin main
```

## What to do after cleanup:
1. Update your remote repository
2. Ask collaborators to fresh clone
3. Download required models separately as per README
"""
    
    with open("GIT_CLEANUP_INSTRUCTIONS.md", "w") as f:
        f.write(instructions)
    
    print("✓ Created GIT_CLEANUP_INSTRUCTIONS.md")

def main():
    """Main cleanup process"""
    print("AI Gesture Gaming - Git History Cleanup")
    print("="*60)
    print("This script will safely remove large files from git history")
    print("while preserving all our development work.")
    print("="*60)
    
    # Get user confirmation
    response = input("\nProceed with cleanup? (y/N): ").strip().lower()
    if response != 'y':
        print("Cleanup cancelled.")
        return
    
    try:
        # Step 1: Backup
        backup_current_work()
        
        # Step 2: Commit current work
        commit_current_changes()
        
        # Step 3: Clean history
        clean_git_history()
        
        # Step 4: Verify
        verify_cleanup()
        
        # Step 5: Create instructions
        create_instructions()
        
        print("\n" + "="*60)
        print("CLEANUP COMPLETE!")
        print("="*60)
        print("Next steps:")
        print("1. Check the repository size")
        print("2. Try pushing: git push origin main")
        print("3. If push still fails, see GIT_CLEANUP_INSTRUCTIONS.md")
        print("4. Your backup is in ../ai-gesture-gaming-backup/")
        
    except KeyboardInterrupt:
        print("\n\nCleanup interrupted by user")
    except Exception as e:
        print(f"\n\nError during cleanup: {e}")
        print("Your work is backed up in ../ai-gesture-gaming-backup/")
        print("See GIT_CLEANUP_INSTRUCTIONS.md for manual options")

if __name__ == "__main__":
    main()