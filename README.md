---
title: RAG System with Unsloth 4-bit
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
license: mit
---

# 🤖 RAG System with Unsloth 4-bit Quantization

A Retrieval-Augmented Generation system powered by Unsloth's 4-bit quantized LLMs, FAISS vector search, and Streamlit.

## Features

- ✅ **4-bit Quantized LLM**: Memory-efficient inference with Unsloth
- ✅ **Semantic Search**: Fast vector similarity with FAISS
- ✅ **Interactive UI**: Clean Streamlit interface
- ✅ **Customizable**: Adjust context, temperature, and response length

## How to Use

1. Type your question in the text area
2. Adjust settings in the sidebar (optional)
3. Click "Generate Answer"
4. View the AI response and retrieved context

## Configuration

**Settings you can adjust:**
- **Context chunks**: Number of relevant documents to retrieve (1-5)
- **Max tokens**: Length of generated response (50-500)
- **Temperature**: Creativity level (0.1-1.0)

## Sample Questions

- What is Retrieval-Augmented Generation?
- How does 4-bit quantization work?
- What are vector databases?
- Explain transfer learning

## Add Your Own Documents

To use your own data, edit the `domain_documents` list in `app.py`:

```python
domain_documents = [
    "Your document 1 text here...",
    "Your document 2 text here...",
    # Add more documents
]
```

## Technical Details

- **Model**: Llama-3.2-3B-Instruct (4-bit quantized)
- **Embeddings**: all-MiniLM-L6-v2 (384 dimensions)
- **Vector DB**: FAISS with cosine similarity
- **VRAM**: ~4-6GB

## Credits

Built with:
- [Unsloth](https://github.com/unslothai/unsloth) - Fast 4-bit quantization
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [Streamlit](https://streamlit.io) - Web framework
