#!/usr/bin/env python3
"""
Game RAG CLI Interface
Simple command-line interface for testing the RAG system
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from game_rag.document_loader import GameDocumentLoader
from game_rag.query_engine import GameRAGEngine
from game_rag.vector_store import GameVectorStore

class GameRAGCLI:
    def __init__(self):
        """Initialize CLI interface"""
        self.rag_engine = None
        self.document_loader = None
        self.vector_store = None
        
        print("🎮 Game RAG System CLI")
        print("=" * 40)
        
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ OpenAI API key not found in environment")
            print("💡 Set OPENAI_API_KEY environment variable for full functionality")
            print("🔄 You can still use the system without LLM features")
        
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize RAG components"""
        try:
            self.document_loader = GameDocumentLoader()
            self.rag_engine = GameRAGEngine()
            self.vector_store = GameVectorStore()
            
            print("✅ RAG system initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            sys.exit(1)
    
    def run(self):
        """Main CLI loop"""
        while True:
            print("\\n" + "=" * 40)
            print("🎮 Game RAG System")
            print("=" * 40)
            print("1. 📄 Add documents")
            print("2. 🔍 Query knowledge base")
            print("3. 📊 View statistics")
            print("4. 🎯 List available games")
            print("5. 🧹 Clear database")
            print("6. 📤 Export/Import")
            print("7. ❓ Help")
            print("8. 🚪 Exit")
            print("=" * 40)
            
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == "1":
                self.add_documents_menu()
            elif choice == "2":
                self.query_menu()
            elif choice == "3":
                self.show_statistics()
            elif choice == "4":
                self.list_games()
            elif choice == "5":
                self.clear_database()
            elif choice == "6":
                self.export_import_menu()
            elif choice == "7":
                self.show_help()
            elif choice == "8":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def add_documents_menu(self):
        """Document addition menu"""
        print("\\n📄 Add Documents")
        print("1. 📁 Load from directory")
        print("2. 📄 Load single PDF")
        print("3. 📝 Load text file")
        print("4. 🌐 Load from URL")
        print("5. ⬅️ Back to main menu")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            self.load_directory()
        elif choice == "2":
            self.load_single_pdf()
        elif choice == "3":
            self.load_text_file()
        elif choice == "4":
            self.load_url()
        elif choice == "5":
            return
        else:
            print("❌ Invalid choice")
    
    def load_directory(self):
        """Load documents from directory"""
        directory = input("Enter directory path: ").strip()
        game_name = input("Enter game name: ").strip()
        
        if not directory or not game_name:
            print("❌ Directory path and game name are required")
            return
        
        if not os.path.exists(directory):
            print(f"❌ Directory not found: {directory}")
            return
        
        print(f"📁 Loading documents from {directory}...")
        documents = self.document_loader.load_directory(directory, game_name)
        
        if documents:
            success = self.rag_engine.add_documents_from_loader(documents)
            if success:
                print(f"✅ Successfully added {len(documents)} documents")
            else:
                print("❌ Failed to add documents to vector store")
        else:
            print("❌ No documents found or loading failed")
    
    def load_single_pdf(self):
        """Load single PDF file"""
        file_path = input("Enter PDF file path: ").strip()
        game_name = input("Enter game name: ").strip()
        
        if not file_path or not game_name:
            print("❌ File path and game name are required")
            return
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        print(f"📄 Loading PDF: {file_path}...")
        documents = self.document_loader.load_pdf(file_path, game_name)
        
        if documents:
            success = self.rag_engine.add_documents_from_loader(documents)
            if success:
                print(f"✅ Successfully added {len(documents)} document chunks")
            else:
                print("❌ Failed to add documents to vector store")
        else:
            print("❌ PDF loading failed")
    
    def load_text_file(self):
        """Load text file"""
        file_path = input("Enter text file path: ").strip()
        game_name = input("Enter game name: ").strip()
        
        if not file_path or not game_name:
            print("❌ File path and game name are required")
            return
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        print(f"📝 Loading text file: {file_path}...")
        documents = self.document_loader.load_text_file(file_path, game_name)
        
        if documents:
            success = self.rag_engine.add_documents_from_loader(documents)
            if success:
                print(f"✅ Successfully added {len(documents)} document chunks")
            else:
                print("❌ Failed to add documents to vector store")
        else:
            print("❌ Text file loading failed")
    
    def load_url(self):
        """Load content from URL"""
        url = input("Enter URL: ").strip()
        game_name = input("Enter game name: ").strip()
        
        if not url or not game_name:
            print("❌ URL and game name are required")
            return
        
        print(f"🌐 Loading content from: {url}...")
        documents = self.document_loader.load_web_content(url, game_name)
        
        if documents:
            success = self.rag_engine.add_documents_from_loader(documents)
            if success:
                print(f"✅ Successfully added {len(documents)} document chunks")
            else:
                print("❌ Failed to add documents to vector store")
        else:
            print("❌ URL loading failed")
    
    def query_menu(self):
        """Query interface"""
        print("\\n🔍 Query Knowledge Base")
        
        # Get available games
        games = self.rag_engine.list_available_games()
        if not games:
            print("❌ No games in knowledge base. Add some documents first.")
            return
        
        print(f"Available games: {', '.join(games)}")
        
        # Get query
        question = input("\\nEnter your question: ").strip()
        if not question:
            print("❌ Question cannot be empty")
            return
        
        # Optional game filter
        game_filter = input("Filter by game (optional, press Enter for all): ").strip()
        if game_filter and game_filter not in games:
            print(f"⚠️ Game '{game_filter}' not found. Searching all games.")
            game_filter = None
        
        # Perform query
        print(f"🔍 Searching for: {question}")
        if game_filter:
            print(f"🎮 Filtering by game: {game_filter}")
        
        response = self.rag_engine.query(question, game_filter=game_filter)
        
        # Display results
        print("\\n" + "=" * 40)
        print("📋 ANSWER:")
        print("=" * 40)
        print(response["answer"])
        
        print("\\n" + "=" * 40)
        print("📊 METADATA:")
        print("=" * 40)
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Retrieved documents: {response['retrieved_docs']}")
        
        if response["sources"]:
            print("\\nSources:")
            for i, source in enumerate(response["sources"], 1):
                print(f"  {i}. {source['game']} - {source['source']} ({source['type']})")
    
    def show_statistics(self):
        """Show system statistics"""
        print("\\n📊 System Statistics")
        print("=" * 40)
        
        stats = self.rag_engine.get_stats()
        
        print(f"Documents in vector store: {stats['vector_store']['document_count']}")
        print(f"LLM available: {'✅' if stats['llm_available'] else '❌'}")
        print(f"Model: {stats['model_name']}")
        print(f"Queries processed: {stats['queries_processed']}")
        print(f"Available games: {len(stats['available_games'])}")
        
        if stats['available_games']:
            print(f"Games: {', '.join(stats['available_games'])}")
    
    def list_games(self):
        """List available games"""
        print("\\n🎯 Available Games")
        print("=" * 40)
        
        games = self.rag_engine.list_available_games()
        
        if not games:
            print("❌ No games in knowledge base")
        else:
            for i, game in enumerate(games, 1):
                print(f"{i}. {game}")
    
    def clear_database(self):
        """Clear the database"""
        print("\\n🧹 Clear Database")
        print("⚠️ This will remove ALL documents from the knowledge base!")
        
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        
        if confirm == "yes":
            success = self.vector_store.clear_collection()
            if success:
                print("✅ Database cleared successfully")
            else:
                print("❌ Failed to clear database")
        else:
            print("❌ Clear cancelled")
    
    def export_import_menu(self):
        """Export/Import menu"""
        print("\\n📤 Export/Import")
        print("1. 📤 Export database")
        print("2. 📥 Import database")
        print("3. 📤 Export query history")
        print("4. ⬅️ Back to main menu")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            filepath = input("Enter export file path (with .json extension): ").strip()
            if filepath:
                success = self.vector_store.export_collection(filepath)
                if success:
                    print("✅ Database exported successfully")
                else:
                    print("❌ Export failed")
        
        elif choice == "2":
            filepath = input("Enter import file path: ").strip()
            if filepath and os.path.exists(filepath):
                success = self.vector_store.import_collection(filepath)
                if success:
                    print("✅ Database imported successfully")
                else:
                    print("❌ Import failed")
            else:
                print("❌ File not found")
        
        elif choice == "3":
            filepath = input("Enter export file path (with .json extension): ").strip()
            if filepath:
                success = self.rag_engine.export_history(filepath)
                if success:
                    print("✅ Query history exported successfully")
                else:
                    print("❌ Export failed")
        
        elif choice == "4":
            return
        else:
            print("❌ Invalid choice")
    
    def show_help(self):
        """Show help information"""
        print("\\n❓ Help")
        print("=" * 40)
        print("Game RAG System - AI-powered game guide assistant")
        print()
        print("🔧 Setup:")
        print("1. Install requirements: pip install -r game_rag/requirements.txt")
        print("2. Set OpenAI API key: export OPENAI_API_KEY=your_key_here")
        print()
        print("📄 Supported formats:")
        print("- PDF files (.pdf)")
        print("- Text files (.txt, .md)")
        print("- Web pages (URLs)")
        print()
        print("🎮 Usage:")
        print("1. Add game documents to build knowledge base")
        print("2. Query the system with natural language questions")
        print("3. Get AI-powered answers with source citations")
        print()
        print("💡 Tips:")
        print("- Use specific game names when adding documents")
        print("- Ask detailed questions for better results")
        print("- Use game filters to narrow search results")

def main():
    """Main entry point"""
    try:
        cli = GameRAGCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\\n\\n👋 Goodbye!")
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()