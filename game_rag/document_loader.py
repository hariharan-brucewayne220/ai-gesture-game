#!/usr/bin/env python3
"""
Document Loader for Game Guides
Handles PDF loading, text extraction, and chunking for RAG system
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

# PDF processing
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Text processing
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_SUPPORT = True
except ImportError:
    LANGCHAIN_SUPPORT = False

class GameDocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document loader
        
        Args:
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        if LANGCHAIN_SUPPORT:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            self.text_splitter = self._simple_text_splitter
        
        print(f"Document loader initialized")
        print(f"PDF support: {'YES' if PDF_SUPPORT else 'NO'}")
        print(f"LangChain support: {'YES' if LANGCHAIN_SUPPORT else 'NO'}")
    
    def load_pdf(self, file_path: str, game_name: str) -> List[Dict[str, Any]]:
        """
        Load and process PDF file
        
        Args:
            file_path: Path to PDF file
            game_name: Name of the game for metadata
            
        Returns:
            List of document chunks with metadata
        """
        if not PDF_SUPPORT:
            print("ERROR: PDF support not available. Install PyPDF2: pip install PyPDF2")
            return []
        
        try:
            documents = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    full_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                
                # Clean and chunk the text
                cleaned_text = self._clean_text(full_text)
                chunks = self._chunk_text(cleaned_text)
                
                # Create documents with metadata
                for i, chunk in enumerate(chunks):
                    doc_id = self._generate_chunk_id(file_path, i)
                    
                    document = {
                        "id": doc_id,
                        "content": chunk,
                        "metadata": {
                            "game": game_name,
                            "source": os.path.basename(file_path),
                            "source_type": "pdf",
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "file_path": file_path
                        }
                    }
                    documents.append(document)
                
                print(f"Loaded PDF: {file_path}")
                print(f"Created {len(documents)} chunks")
                
                return documents
                
        except Exception as e:
            print(f"ERROR loading PDF {file_path}: {e}")
            return []
    
    def load_text_file(self, file_path: str, game_name: str) -> List[Dict[str, Any]]:
        """
        Load and process text file
        
        Args:
            file_path: Path to text file
            game_name: Name of the game for metadata
            
        Returns:
            List of document chunks with metadata
        """
        try:
            documents = []
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Clean and chunk the text
            cleaned_text = self._clean_text(content)
            chunks = self._chunk_text(cleaned_text)
            
            # Create documents with metadata
            for i, chunk in enumerate(chunks):
                doc_id = self._generate_chunk_id(file_path, i)
                
                document = {
                    "id": doc_id,
                    "content": chunk,
                    "metadata": {
                        "game": game_name,
                        "source": os.path.basename(file_path),
                        "source_type": "text",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_path": file_path
                    }
                }
                documents.append(document)
            
            print(f"Loaded text file: {file_path}")
            print(f"Created {len(documents)} chunks")
            
            return documents
            
        except Exception as e:
            print(f"ERROR loading text file {file_path}: {e}")
            return []
    
    def load_directory(self, directory_path: str, game_name: str) -> List[Dict[str, Any]]:
        """
        Load all supported files from a directory
        
        Args:
            directory_path: Path to directory containing game documents
            game_name: Name of the game for metadata
            
        Returns:
            List of all document chunks with metadata
        """
        if not os.path.exists(directory_path):
            print(f"ERROR: Directory not found: {directory_path}")
            return []
        
        all_documents = []
        supported_extensions = ['.pdf', '.txt', '.md']
        
        for file_path in Path(directory_path).rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                if file_path.suffix.lower() == '.pdf':
                    documents = self.load_pdf(str(file_path), game_name)
                else:
                    documents = self.load_text_file(str(file_path), game_name)
                
                all_documents.extend(documents)
        
        print(f"Loaded {len(all_documents)} total chunks from {directory_path}")
        return all_documents
    
    def load_web_content(self, url: str, game_name: str) -> List[Dict[str, Any]]:
        """
        Load content from web URL (basic implementation)
        
        Args:
            url: URL to load content from
            game_name: Name of the game for metadata
            
        Returns:
            List of document chunks with metadata
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text from paragraphs and headers
            text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            content = '\n\n'.join([elem.get_text().strip() for elem in text_elements])
            
            # Clean and chunk the text
            cleaned_text = self._clean_text(content)
            chunks = self._chunk_text(cleaned_text)
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_id = self._generate_chunk_id(url, i)
                
                document = {
                    "id": doc_id,
                    "content": chunk,
                    "metadata": {
                        "game": game_name,
                        "source": url,
                        "source_type": "web",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "url": url
                    }
                }
                documents.append(document)
            
            print(f"Loaded web content: {url}")
            print(f"Created {len(documents)} chunks")
            
            return documents
            
        except ImportError:
            print("ERROR: Web loading requires: pip install requests beautifulsoup4")
            return []
        except Exception as e:
            print(f"ERROR loading web content {url}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()[\]]', '', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines
        
        return text.strip()
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if LANGCHAIN_SUPPORT:
            return self.text_splitter.split_text(text)
        else:
            return self._simple_text_splitter(text)
    
    def _simple_text_splitter(self, text: str) -> List[str]:
        """
        Simple text splitter fallback
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        words = text.split()
        
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= self.chunk_size:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _generate_chunk_id(self, source: str, chunk_index: int) -> str:
        """
        Generate unique ID for a document chunk
        
        Args:
            source: Source file path or URL
            chunk_index: Index of the chunk
            
        Returns:
            Unique chunk ID
        """
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"chunk_{source_hash}_{chunk_index}"
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats
        
        Returns:
            List of supported file extensions
        """
        formats = ['.txt', '.md']
        if PDF_SUPPORT:
            formats.append('.pdf')
        return formats

# Example usage and testing
if __name__ == "__main__":
    # Initialize loader
    loader = GameDocumentLoader(chunk_size=500, chunk_overlap=50)
    
    # Test with sample text
    sample_text = """
    God of War Combat Guide
    
    Chapter 1: Basic Combat
    Combat in God of War is all about timing and positioning. Use your axe to deal heavy damage.
    
    Chapter 2: Advanced Techniques
    Master the dodge roll and parry system to become unstoppable in combat.
    
    Chapter 3: Boss Strategies
    Each boss has unique patterns. Study their movements and strike when they're vulnerable.
    """
    
    # Save sample text and test loading
    with open('sample_guide.txt', 'w') as f:
        f.write(sample_text)
    
    # Test loading
    documents = loader.load_text_file('sample_guide.txt', 'God of War')
    
    print(f"\\nSample loading result:")
    for i, doc in enumerate(documents):
        print(f"Chunk {i}: {doc['content'][:100]}...")
        print(f"Metadata: {doc['metadata']}")
        print()
    
    # Clean up
    os.remove('sample_guide.txt')
    
    print(f"Supported formats: {loader.get_supported_formats()}")