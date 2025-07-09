#!/usr/bin/env python3
"""
RAG Query Engine with LangChain
Handles question answering using retrieved documents and LLM
"""

import os
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# LLM and embedding imports
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
    LANGCHAIN_SUPPORT = True
except ImportError:
    LANGCHAIN_SUPPORT = False

# Local imports
from .vector_store import GameVectorStore

class GameRAGEngine:
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_tokens: int = 500):
        """
        Initialize RAG engine
        
        Args:
            openai_api_key: OpenAI API key (or from environment)
            model_name: LLM model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize vector store
        self.vector_store = GameVectorStore()
        
        # Initialize LLM if API key is available
        if self.api_key and LANGCHAIN_SUPPORT:
            self.llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            self.llm_available = True
            print(f"LLM initialized: {model_name}")
        else:
            self.llm = None
            self.llm_available = False
            print("LLM not available (missing API key or LangChain)")
        
        # Initialize prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Query history for debugging
        self.query_history = []
        
        print(f"RAG engine initialized")
    
    def query(self, question: str, game_filter: Optional[str] = None, 
              n_results: int = 3) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User's question
            game_filter: Optional game name to filter results
            n_results: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant documents
        retrieval_results = self.vector_store.query_documents(
            query=question,
            n_results=n_results,
            game_filter=game_filter
        )
        
        # If no documents found, try OpenAI knowledge fallback
        if not retrieval_results["documents"]:
            return self._dynamic_fallback_query(question, game_filter)
        
        # Generate answer using LLM
        if self.llm_available:
            answer = self._generate_answer(question, retrieval_results)
            confidence = self._calculate_confidence(retrieval_results)
        else:
            # Fallback: return concatenated document snippets
            answer = self._fallback_answer(question, retrieval_results)
            confidence = 0.5  # Medium confidence for fallback
        
        # Format sources
        sources = self._format_sources(retrieval_results)
        
        # Create response
        response = {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "retrieved_docs": len(retrieval_results["documents"]),
            "query": question,
            "game_filter": game_filter,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        self.query_history.append(response)
        
        return response
    
    def add_documents_from_loader(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents from document loader to vector store
        
        Args:
            documents: List of documents from DocumentLoader
            
        Returns:
            Success status
        """
        if not documents:
            return False
        
        # Extract content and metadata
        doc_texts = [doc["content"] for doc in documents]
        doc_metadatas = [doc["metadata"] for doc in documents]
        doc_ids = [doc["id"] for doc in documents]
        
        # Add to vector store
        return self.vector_store.add_documents(doc_texts, doc_metadatas, doc_ids)
    
    def list_available_games(self) -> List[str]:
        """
        List all games in the knowledge base
        
        Returns:
            List of game names
        """
        return self.vector_store.list_games()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics
        
        Returns:
            Dictionary with system stats
        """
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "vector_store": vector_stats,
            "llm_available": self.llm_available,
            "model_name": self.model_name,
            "queries_processed": len(self.query_history),
            "available_games": self.list_available_games()
        }
    
    def _create_prompt_template(self) -> str:
        """
        Create prompt template for RAG
        
        Returns:
            Prompt template string
        """
        template = """You are a helpful gaming assistant. Use the following game guide excerpts to answer the user's question accurately and concisely.

Context from game guides:
{context}

User Question: {question}

Instructions:
1. Answer based primarily on the provided context
2. If the context doesn't contain the answer, say so clearly
3. Be specific about game mechanics, strategies, or solutions
4. Keep responses concise but helpful
5. If mentioning specific games, be clear about which game you're referring to

Answer:"""
        
        return template
    
    def _generate_answer(self, question: str, retrieval_results: Dict[str, Any]) -> str:
        """
        Generate answer using LLM
        
        Args:
            question: User's question
            retrieval_results: Retrieved documents
            
        Returns:
            Generated answer
        """
        # Combine retrieved documents as context
        context = "\n\n".join([
            f"[From {meta.get('game', 'Unknown')} - {meta.get('source', 'Unknown')}]:\\n{doc}"
            for doc, meta in zip(retrieval_results["documents"], retrieval_results["metadatas"])
        ])
        
        # Format prompt
        prompt = self.prompt_template.format(context=context, question=question)
        
        try:
            # Generate response
            response = self.llm.predict(prompt)
            return response.strip()
            
        except Exception as e:
            print(f"ERROR generating answer: {e}")
            return self._fallback_answer(question, retrieval_results)
    
    def _fallback_answer(self, question: str, retrieval_results: Dict[str, Any]) -> str:
        """
        Fallback answer when LLM is not available
        
        Args:
            question: User's question
            retrieval_results: Retrieved documents
            
        Returns:
            Fallback answer
        """
        # Simple concatenation of top results
        answer_parts = []
        
        for i, (doc, meta) in enumerate(zip(retrieval_results["documents"], retrieval_results["metadatas"])):
            game = meta.get("game", "Unknown")
            source = meta.get("source", "Unknown")
            
            # Take first 200 characters of each document
            snippet = doc[:200] + "..." if len(doc) > 200 else doc
            answer_parts.append(f"From {game} ({source}): {snippet}")
            
            if i >= 2:  # Limit to top 3 results
                break
        
        return "\\n\\n".join(answer_parts)
    
    def _calculate_confidence(self, retrieval_results: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on retrieval results
        
        Args:
            retrieval_results: Retrieved documents with distances
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not retrieval_results["distances"]:
            return 0.0
        
        # Use average distance (lower is better)
        avg_distance = sum(retrieval_results["distances"]) / len(retrieval_results["distances"])
        
        # Convert distance to confidence (rough heuristic)
        # Distance typically ranges from 0.0 to 2.0
        confidence = max(0.0, min(1.0, 1.0 - (avg_distance / 2.0)))
        
        return confidence
    
    def _format_sources(self, retrieval_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Format sources for response
        
        Args:
            retrieval_results: Retrieved documents with metadata
            
        Returns:
            List of formatted sources
        """
        sources = []
        
        for meta in retrieval_results["metadatas"]:
            source = {
                "game": meta.get("game", "Unknown"),
                "source": meta.get("source", "Unknown"),
                "type": meta.get("source_type", "unknown")
            }
            
            # Add additional info if available
            if "url" in meta:
                source["url"] = meta["url"]
            if "file_path" in meta:
                source["file"] = meta["file_path"]
            
            sources.append(source)
        
        return sources
    
    def _dynamic_fallback_query(self, question: str, game_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Dynamic fallback when no local documents found - uses OpenAI knowledge
        
        Args:
            question: User's question
            game_filter: Optional game name to filter results
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.llm_available:
            return {
                "answer": "I don't have information about that in my knowledge base. Please add more game guides or provide an OpenAI API key.",
                "confidence": 0.0,
                "sources": [],
                "retrieved_docs": 0,
                "query": question,
                "method": "no_fallback"
            }
        
        # Use OpenAI's knowledge about games
        game_context = f" in {game_filter}" if game_filter else ""
        
        dynamic_prompt = f"""
        You are a gaming expert assistant. The user is asking about a game topic that isn't in our local knowledge base.
        
        Question: {question}{game_context}
        
        Please provide a helpful, accurate answer based on your knowledge of games. Include:
        1. Specific strategies or tips
        2. Step-by-step instructions if applicable
        3. Any important warnings or notes
        
        If you don't know about this specific topic, say so honestly.
        
        Answer:
        """
        
        try:
            # Generate response using OpenAI
            response = self.llm.predict(dynamic_prompt)
            
            return {
                "answer": f"[Generated from AI knowledge]: {response.strip()}",
                "confidence": 0.7,  # Good confidence for AI knowledge
                "sources": [{"source": "OpenAI Knowledge", "game": game_filter or "General"}],
                "retrieved_docs": 0,
                "query": question,
                "method": "openai_fallback",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"ERROR with OpenAI fallback: {e}")
            return {
                "answer": f"I don't have information about that in my knowledge base, and couldn't access external knowledge. Error: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "retrieved_docs": 0,
                "query": question,
                "method": "error_fallback"
            }
    
    def clear_history(self):
        """Clear query history"""
        self.query_history = []
        print("Query history cleared")
    
    def export_history(self, filepath: str) -> bool:
        """
        Export query history to JSON file
        
        Args:
            filepath: Path to save history
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.query_history, f, indent=2, ensure_ascii=False)
            
            print(f"Query history exported to: {filepath}")
            return True
            
        except Exception as e:
            print(f"ERROR exporting history: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG engine
    rag_engine = GameRAGEngine()
    
    # Test with sample data (if vector store has data)
    stats = rag_engine.get_stats()
    print(f"System stats: {stats}")
    
    # Test query
    if stats["vector_store"]["document_count"] > 0:
        response = rag_engine.query("How do I defeat the boss?")
        print(f"Sample query result: {response}")
    else:
        print("No documents in vector store. Add some documents first.")
    
    # Test games listing
    games = rag_engine.list_available_games()
    print(f"Available games: {games}")