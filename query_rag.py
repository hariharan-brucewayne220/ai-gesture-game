#!/usr/bin/env python3
"""
RAG System - Interactive CLI Interface
Clean version without Unicode characters for Windows compatibility
"""

import sys
import os
import time
from typing import Optional, List, Dict

# Add current directory to path
sys.path.append('.')

class RAGInterface:
    def __init__(self):
        self.rag_engine = None
        self.available_games = []
        self.current_game = None
        self.history = []
        
    def initialize_rag(self):
        """Initialize the RAG engine"""
        try:
            from game_rag.query_engine import GameRAGEngine
            
            print("Initializing RAG system...")
            self.rag_engine = GameRAGEngine()
            
            # Get available games
            stats = self.rag_engine.get_stats()
            self.available_games = stats.get('available_games', [])
            
            print(f"[OK] RAG system initialized")
            print(f"[OK] Vector database contains {stats['vector_store']['document_count']} documents")
            print(f"[OK] {len(self.available_games)} games available")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize RAG system: {e}")
            return False
    
    def show_header(self):
        """Show the application header"""
        print("\n" + "=" * 70)
        print("           AI GESTURE GAMING - RAG SYSTEM")
        print("           Interactive Game Guide Assistant")
        print("=" * 70)
        
        if self.current_game:
            print(f"Current Game: {self.current_game}")
        else:
            print("Current Game: All Games")
        print()
    
    def show_main_menu(self):
        """Show the main menu"""
        print("MAIN MENU")
        print("-" * 20)
        print("1. Ask a question")
        print("2. Select game")
        print("3. Browse available games")
        print("4. View query history")
        print("5. System stats")
        print("6. Help")
        print("7. Exit")
        print()
    
    def show_games_menu(self):
        """Show available games menu"""
        print("AVAILABLE GAMES")
        print("-" * 25)
        
        if not self.available_games:
            print("No games found in database")
            return
        
        print("0. All Games (no filter)")
        for i, game in enumerate(self.available_games, 1):
            print(f"{i}. {game}")
        print()
    
    def select_game(self):
        """Game selection interface"""
        self.show_games_menu()
        
        try:
            choice = input("Select game (number): ").strip()
            
            if not choice.isdigit():
                print("[ERROR] Please enter a number")
                return False
                
            choice = int(choice)
            
            if choice == 0:
                self.current_game = None
                print("[OK] Set to search all games")
                return True
            elif 1 <= choice <= len(self.available_games):
                self.current_game = self.available_games[choice - 1]
                print(f"[OK] Selected game: {self.current_game}")
                return True
            else:
                print("[ERROR] Invalid selection")
                return False
                
        except ValueError:
            print("[ERROR] Please enter a valid number")
            return False
    
    def ask_question(self):
        """Interactive question asking"""
        print("ASK A QUESTION")
        print("-" * 20)
        
        if self.current_game:
            print(f"Searching in: {self.current_game}")
        else:
            print("Searching in: All Games")
        
        print("\nExample questions:")
        print("  - How do I defeat Sigrun?")
        print("  - What's the best armor?")
        print("  - Combat strategies and tips")
        print("  - How to unlock secret areas?")
        print()
        
        question = input("Your question: ").strip()
        
        if not question:
            print("[ERROR] Please enter a question")
            return
        
        if question.lower() in ['quit', 'exit', 'back']:
            return
        
        # Show searching animation
        print("\nSearching for answer", end="")
        for i in range(3):
            print(".", end="", flush=True)
            time.sleep(0.5)
        print()
        
        try:
            # Query the RAG system
            result = self.rag_engine.query(question, self.current_game)
            
            # Display results
            print("\n" + "-" * 50)
            print("ANSWER")
            print("-" * 50)
            print(result['answer'])
            
            print(f"\nDETAILS")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Sources: {len(result['sources'])}")
            print(f"   Documents: {result['retrieved_docs']}")
            
            if result['sources']:
                print(f"\nSOURCES")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"   {i}. {source['source']} ({source['game']})")
            
            # Add to history
            self.history.append({
                'question': question,
                'game': self.current_game,
                'answer': result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer'],
                'confidence': result['confidence']
            })
            
            print("\n" + "-" * 50)
            
        except Exception as e:
            print(f"[ERROR] {e}")
    
    def show_history(self):
        """Show query history"""
        print("QUERY HISTORY")
        print("-" * 20)
        
        if not self.history:
            print("No queries yet")
            return
        
        for i, entry in enumerate(self.history[-10:], 1):  # Show last 10
            game_info = f" ({entry['game']})" if entry['game'] else ""
            print(f"{i}. Q: {entry['question']}{game_info}")
            print(f"   A: {entry['answer']}")
            print(f"   Confidence: {entry['confidence']:.2f}")
            print()
    
    def show_stats(self):
        """Show system statistics"""
        print("SYSTEM STATISTICS")
        print("-" * 25)
        
        try:
            stats = self.rag_engine.get_stats()
            
            print(f"Vector Database:")
            print(f"  Total documents: {stats['vector_store']['document_count']}")
            print(f"  Available games: {len(stats['available_games'])}")
            print(f"  LLM available: {'Yes' if stats['llm_available'] else 'No'}")
            print(f"  Model: {stats['model_name']}")
            print(f"  Queries processed: {stats['queries_processed']}")
            
            if stats['available_games']:
                print(f"\nGames in database:")
                for game in stats['available_games']:
                    print(f"  - {game}")
            
        except Exception as e:
            print(f"[ERROR] Error getting stats: {e}")
    
    def show_help(self):
        """Show help information"""
        print("HELP")
        print("-" * 10)
        print("This is an interactive RAG (Retrieval-Augmented Generation) system")
        print("for game guides and strategies.")
        print()
        print("Features:")
        print("  - Game-specific searches")
        print("  - PDF guide integration")
        print("  - Semantic search")
        print("  - Query history")
        print()
        print("Tips:")
        print("  - Be specific in your questions")
        print("  - Select a game for better results")
        print("  - Ask about strategies, tips, walkthroughs")
        print("  - Use natural language questions")
        print()
        print("Example questions:")
        print("  - How do I defeat [boss name]?")
        print("  - What's the best [weapon/armor] for [situation]?")
        print("  - Where can I find [item]?")
        print("  - Tips for [specific area/level]?")
        print()
        print("Current database contains:")
        print("  - God of War guide (from PDF)")
        print("  - 20+ other games (basic info)")
        print()
    
    def run(self):
        """Main application loop"""
        # Initialize system
        if not self.initialize_rag():
            print("[ERROR] Failed to start RAG system")
            return
        
        # Main loop
        while True:
            try:
                self.show_header()
                self.show_main_menu()
                
                choice = input("Select option (1-7): ").strip()
                
                if choice == '1':
                    self.ask_question()
                    input("\nPress Enter to continue...")
                elif choice == '2':
                    self.select_game()
                    input("\nPress Enter to continue...")
                elif choice == '3':
                    self.show_games_menu()
                    input("\nPress Enter to continue...")
                elif choice == '4':
                    self.show_history()
                    input("\nPress Enter to continue...")
                elif choice == '5':
                    self.show_stats()
                    input("\nPress Enter to continue...")
                elif choice == '6':
                    self.show_help()
                    input("\nPress Enter to continue...")
                elif choice == '7':
                    print("\nThanks for using the RAG system!")
                    break
                else:
                    print("[ERROR] Invalid choice. Please enter 1-7.")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")
                time.sleep(2)

def main():
    """Main entry point"""
    interface = RAGInterface()
    interface.run()

if __name__ == "__main__":
    main()