# 🎮 Game RAG System

AI-powered game guide assistant using Retrieval-Augmented Generation (RAG) to answer gaming questions from your document collection.

## ✨ Features

- 📄 **Multi-format support**: PDF, text files, web pages
- 🔍 **Semantic search**: Find relevant information using vector similarity
- 🤖 **AI-powered answers**: Generate contextual responses using LLM
- 🎯 **Game filtering**: Search within specific games
- 💾 **Persistent storage**: Documents saved locally using Chroma
- 📊 **Query history**: Track and export your interactions
- 🌐 **Web scraping**: Load guides directly from URLs

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r game_rag/requirements.txt
```

### 2. Set OpenAI API Key (Optional)

```bash
export OPENAI_API_KEY=your_key_here
```

*Note: The system works without OpenAI - it will use document snippets instead of generated answers.*

### 3. Run CLI Interface

```bash
python game_rag/cli.py
```

### 4. Add Documents

- Load PDF game guides
- Import text files with strategies
- Scrape web guides
- Organize by game names

### 5. Query the System

```
🔍 Query: "How do I defeat the final boss in God of War?"
🎮 Filter: God of War
📋 Answer: Based on the guide, the final boss requires...
```

## 📁 Project Structure

```
game_rag/
├── __init__.py           # Package initialization
├── vector_store.py       # Chroma vector database management
├── document_loader.py    # PDF/text/web content loading
├── query_engine.py       # RAG pipeline with LangChain
├── cli.py               # Command-line interface
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🛠️ Core Components

### Vector Store (`vector_store.py`)
- **Chroma database** for persistent vector storage
- **Semantic search** using embeddings
- **Metadata filtering** by game, source, etc.
- **Export/import** functionality

### Document Loader (`document_loader.py`)
- **PDF parsing** with PyPDF2
- **Text chunking** for optimal embedding
- **Web scraping** with BeautifulSoup
- **Metadata extraction** (game, source, type)

### Query Engine (`query_engine.py`)
- **RAG pipeline** with LangChain
- **OpenAI integration** for LLM responses
- **Confidence scoring** based on retrieval quality
- **Source citation** for transparency

## 🎯 Usage Examples

### Adding Documents

```python
from game_rag.document_loader import GameDocumentLoader
from game_rag.query_engine import GameRAGEngine

# Initialize components
loader = GameDocumentLoader()
rag_engine = GameRAGEngine()

# Load PDF guide
documents = loader.load_pdf("elden_ring_guide.pdf", "Elden Ring")
rag_engine.add_documents_from_loader(documents)

# Load from directory
documents = loader.load_directory("./game_guides/", "The Witcher 3")
rag_engine.add_documents_from_loader(documents)
```

### Querying

```python
# Ask a question
response = rag_engine.query("How do I level up quickly?")
print(response["answer"])
print(f"Confidence: {response['confidence']}")

# Filter by game
response = rag_engine.query(
    "Best weapons for boss fights", 
    game_filter="Dark Souls"
)
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for LLM responses
- `CHROMA_DB_PATH`: Custom path for vector database (optional)

### Customization
- **Chunk size**: Adjust in `DocumentLoader(chunk_size=1000)`
- **Model**: Change in `GameRAGEngine(model_name="gpt-4")`
- **Temperature**: Modify in `GameRAGEngine(temperature=0.1)`

## 📊 Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Requires PyPDF2 |
| Text | `.txt`, `.md` | Direct text processing |
| Web | URLs | Requires requests + BeautifulSoup |

## 🎮 Game Integration

### Future Integration with Voice System
```python
# Integration with existing voice controller
def handle_voice_query(voice_text):
    if voice_text.startswith("ask guide"):
        question = voice_text[10:]  # Remove "ask guide "
        response = rag_engine.query(question)
        return response["answer"]
```

### Hotkey Integration
```python
# Add to main.py hotkey handler
elif key == keyboard.Key.f1:  # F1 for game guide
    question = input("Ask game guide: ")
    response = rag_engine.query(question)
    print(response["answer"])
```

## 🛡️ Error Handling

- **Graceful degradation**: Works without OpenAI API
- **Fallback responses**: Document snippets when LLM fails
- **Input validation**: Checks file existence and formats
- **Exception handling**: Detailed error messages

## 📈 Performance

- **Vector search**: Sub-second retrieval for 1000+ documents
- **Chunk processing**: Optimal 1000-character chunks with overlap
- **Memory usage**: Efficient with Chroma's disk-based storage
- **Caching**: Automatic caching of embeddings

## 🔮 Future Enhancements

- **Local LLM support**: Ollama/HuggingFace models
- **Advanced chunking**: Semantic chunking strategies
- **Multi-modal**: Image and video guide support
- **Real-time updates**: Live web scraping
- **User feedback**: Learning from query results

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

MIT License - Free to use, modify, and distribute.

---

**Built with ❤️ to enhance your gaming experience with AI-powered assistance!**