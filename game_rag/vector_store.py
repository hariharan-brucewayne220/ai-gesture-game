#!/usr/bin/env python3
"""
Vector Database Management using Chroma
Handles embedding storage and similarity search for game documents
"""

import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Any, Optional
import hashlib
import json

class GameVectorStore:
    def __init__(self, persist_directory: str = "game_rag/chroma_db"):
        """
        Initialize Chroma vector database
        
        Args:
            persist_directory: Directory to store the vector database
        """
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection for game documents
        self.collection_name = "game_documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Game guides and documentation"}
        )
        
        print(f"Vector store initialized: {persist_directory}")
        print(f"Collection '{self.collection_name}' ready")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], 
                     ids: Optional[List[str]] = None) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries for each document
            ids: Optional list of document IDs (auto-generated if None)
            
        Returns:
            bool: Success status
        """
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [self._generate_id(doc) for doc in documents]
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def query_documents(self, query: str, n_results: int = 5, 
                       game_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for relevant documents using similarity search
        
        Args:
            query: Search query string
            n_results: Number of results to return
            game_filter: Optional game name to filter results
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Build where clause for filtering
            where_clause = {}
            if game_filter:
                where_clause["game"] = game_filter
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Format results
            formatted_results = {
                "documents": results["documents"][0],
                "metadatas": results["metadatas"][0],
                "distances": results["distances"][0],
                "ids": results["ids"][0]
            }
            
            print(f"Found {len(formatted_results['documents'])} results for: '{query}'")
            return formatted_results
            
        except Exception as e:
            print(f"Error querying documents: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            bool: Success status
        """
        try:
            self.collection.delete(ids=ids)
            print(f"🗑️ Deleted {len(ids)} documents")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting documents: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics
        
        Returns:
            Dictionary with collection stats
        """
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {"document_count": 0, "collection_name": "", "persist_directory": ""}
    
    def list_games(self) -> List[str]:
        """
        List all games in the collection
        
        Returns:
            List of game names
        """
        try:
            # Get all documents to extract game names
            results = self.collection.get()
            games = set()
            
            for metadata in results["metadatas"]:
                if "game" in metadata:
                    games.add(metadata["game"])
            
            return sorted(list(games))
            
        except Exception as e:
            print(f"❌ Error listing games: {e}")
            return []
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection
        
        Returns:
            bool: Success status
        """
        try:
            # Delete collection and recreate
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Game guides and documentation"}
            )
            
            print("🧹 Collection cleared")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")
            return False
    
    def _generate_id(self, document: str) -> str:
        """
        Generate a unique ID for a document
        
        Args:
            document: Document text
            
        Returns:
            Unique document ID
        """
        # Create hash from document content
        doc_hash = hashlib.md5(document.encode()).hexdigest()
        return f"doc_{doc_hash[:12]}"
    
    def export_collection(self, filepath: str) -> bool:
        """
        Export collection to JSON file
        
        Args:
            filepath: Path to save the export
            
        Returns:
            bool: Success status
        """
        try:
            results = self.collection.get()
            
            export_data = {
                "collection_name": self.collection_name,
                "documents": results["documents"],
                "metadatas": results["metadatas"],
                "ids": results["ids"]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"📤 Collection exported to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting collection: {e}")
            return False
    
    def import_collection(self, filepath: str) -> bool:
        """
        Import collection from JSON file
        
        Args:
            filepath: Path to import file
            
        Returns:
            bool: Success status
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Clear existing collection
            self.clear_collection()
            
            # Import documents
            success = self.add_documents(
                documents=import_data["documents"],
                metadatas=import_data["metadatas"],
                ids=import_data["ids"]
            )
            
            if success:
                print(f"📥 Collection imported from: {filepath}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error importing collection: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize vector store
    vector_store = GameVectorStore()
    
    # Test data
    test_documents = [
        "To defeat the final boss in God of War, you need to dodge his hammer attacks and strike when he's vulnerable.",
        "The best strategy for Elden Ring is to level up your character and upgrade your weapons before facing tough enemies.",
        "In The Witcher 3, use Quen sign for protection and Igni for dealing fire damage to enemies."
    ]
    
    test_metadatas = [
        {"game": "God of War", "topic": "boss_fight", "difficulty": "hard"},
        {"game": "Elden Ring", "topic": "strategy", "difficulty": "medium"},
        {"game": "The Witcher 3", "topic": "combat", "difficulty": "easy"}
    ]
    
    # Test adding documents
    vector_store.add_documents(test_documents, test_metadatas)
    
    # Test querying
    results = vector_store.query_documents("how to defeat boss", n_results=2)
    print(f"Query results: {results}")
    
    # Test stats
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Test listing games
    games = vector_store.list_games()
    print(f"Available games: {games}")