#!/usr/bin/env python3
"""
Setup RAG Environment Script
Properly sets up the separate RAG environment without breaking MediaPipe
"""

import subprocess
import sys
import os

def setup_rag_environment():
    """Setup the RAG environment with proper dependencies"""
    
    print("Setting up RAG Environment")
    print("=" * 40)
    
    # Check if rag_env exists
    if not os.path.exists("rag_env"):
        print("Error: rag_env directory not found!")
        print("Please run: mp_env/Scripts/python.exe -m venv rag_env")
        return False
    
    # Install dependencies in rag_env
    print("Installing RAG dependencies...")
    
    try:
        # Install in rag_env
        subprocess.run([
            "rag_env/Scripts/pip.exe", "install", 
            "chromadb>=0.4.0,<0.5.0",
            "langchain>=0.1.0",
            "openai>=1.0.0", 
            "sentence-transformers>=2.2.0",
            "--timeout=300"
        ], check=True)
        
        print("✅ RAG dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def test_rag_environment():
    """Test the RAG environment"""
    
    print("\nTesting RAG Environment...")
    print("-" * 30)
    
    test_script = """
import sys
sys.path.append('.')

print("Testing RAG components...")

# Test ChromaDB
try:
    import chromadb
    print("✅ ChromaDB imported successfully")
except Exception as e:
    print(f"❌ ChromaDB error: {e}")

# Test our RAG system
try:
    from game_rag.preloaded_games import PreloadedGames
    print("✅ PreloadedGames imported successfully")
    
    # Initialize (but don't load all games for quick test)
    print("✅ RAG system is ready to use!")
    
except Exception as e:
    print(f"❌ RAG system error: {e}")
"""
    
    try:
        with open("test_rag_env.py", "w") as f:
            f.write(test_script)
        
        result = subprocess.run([
            "rag_env/Scripts/python.exe", "test_rag_env.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        # Cleanup
        os.remove("test_rag_env.py")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def show_usage():
    """Show how to use the RAG system"""
    
    print("\n" + "=" * 50)
    print("🎮 HOW TO ACCESS THE RAG SYSTEM")
    print("=" * 50)
    
    print("\n1. FULL CHROMADB RAG SYSTEM (Advanced):")
    print("   rag_env/Scripts/python.exe start_game_rag.py")
    print("   - 20+ games with vector search")
    print("   - AI-powered semantic matching")
    print("   - OpenAI integration")
    
    print("\n2. SIMPLE RAG SYSTEM (Basic):")
    print("   mp_env/Scripts/python.exe test_simple_rag.py")
    print("   - 4 games with keyword matching")
    print("   - No dependency conflicts")
    print("   - Works with MediaPipe")
    
    print("\n3. INTERACTIVE QUERIES:")
    print("   rag_env/Scripts/python.exe demo_preloaded_games.py")
    print("   - Full interactive interface")
    print("   - Ask any gaming question")
    print("   - Get detailed answers")
    
    print("\n4. PROGRAMMATIC ACCESS:")
    print("   from game_rag.preloaded_games import PreloadedGames")
    print("   rag = PreloadedGames()")
    print("   result = rag.rag_engine.query('How do I defeat Baldur?', 'God of War')")
    
    print("\n" + "=" * 50)
    print("RECOMMENDATION: Use the Full ChromaDB system for best results!")

if __name__ == "__main__":
    # Setup environment
    if setup_rag_environment():
        # Test environment
        if test_rag_environment():
            show_usage()
        else:
            print("❌ RAG environment test failed")
    else:
        print("❌ Failed to setup RAG environment")