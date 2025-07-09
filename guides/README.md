# PDF Guides Directory

This directory is where you should place your game guide PDFs for enhanced RAG functionality.

## How to Add PDF Guides

1. **Download** any game guide PDF
2. **Place it here**: `guides/your_game_guide.pdf`
3. **Run integration**: Use the RAG CLI to automatically detect and integrate

## Supported Files
- Any PDF game guide
- Examples: `god_of_war_guide.pdf`, `last_of_us_2_guide.pdf`, `elden_ring_guide.pdf`

## Integration Steps
```bash
# After adding a PDF to this folder:
rag_env/Scripts/python.exe query_rag.py

# The system will automatically:
# 1. Detect new PDFs
# 2. Extract and chunk content
# 3. Add to vector database
# 4. Make searchable immediately
```

## Why PDFs Aren't Included
PDFs are excluded from the repository due to:
- File size limitations (GitHub 25MB limit)
- Copyright considerations
- User preference for specific guides

## Example Usage
After adding PDFs, you can ask detailed questions like:
- "How do I defeat Sigrun in God of War?"
- "What's the best armor for late game?"
- "Combat strategies for boss battles"
- "Where to find secret areas?"

## Supported PDF Types
- Game strategy guides
- Walkthrough documents
- Boss battle guides
- Equipment/item lists
- Any gaming-related PDF content

The RAG system will automatically process and integrate any PDF you add to this directory!