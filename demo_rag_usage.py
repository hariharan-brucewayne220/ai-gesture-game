#!/usr/bin/env python3
"""
Demo RAG Usage - Shows how to use the RAG system programmatically
"""

import sys
sys.path.append('.')

def demo_rag_usage():
    """Demo the RAG system usage"""
    
    print("=" * 60)
    print("RAG SYSTEM USAGE DEMO")
    print("=" * 60)
    
    try:
        from game_rag.query_engine import GameRAGEngine
        
        # Initialize RAG engine
        print("1. Initializing RAG engine...")
        rag_engine = GameRAGEngine()
        
        # Show stats
        stats = rag_engine.get_stats()
        print(f"   - Documents: {stats['vector_store']['document_count']}")
        print(f"   - Games: {len(stats['available_games'])}")
        
        # Test queries
        queries = [
            ("How do I defeat Sigrun?", "God of War"),
            ("What's the best armor?", "God of War"),
            ("Combat tips", "God of War"),
            ("How to parry?", "Dark Souls 3"),
            ("Best starting class", "Elden Ring")
        ]
        
        print("\n2. Testing queries...")
        print("-" * 40)
        
        for i, (question, game) in enumerate(queries, 1):
            print(f"\n{i}. Question: {question}")
            print(f"   Game: {game}")
            
            result = rag_engine.query(question, game)
            
            print(f"   Answer: {result['answer'][:80]}...")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Sources: {len(result['sources'])}")
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("To use the interactive interface, run:")
        print("   rag_env/Scripts/python.exe rag_cli_clean.py")
        print()
        print("Features:")
        print("  - Interactive menu system")
        print("  - Game selection")
        print("  - Query history")
        print("  - System statistics")
        print("  - Help system")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_rag_usage()