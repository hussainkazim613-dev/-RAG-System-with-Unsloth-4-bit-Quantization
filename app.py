import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import time
from typing import List, Tuple

# Page configuration
st.set_page_config(
    page_title="RAG System with 4-bit LLM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def chunk_documents(documents: List[str], chunk_size: int = 300, overlap: int = 50) -> List[Tuple[str, int]]:
    """Split documents into overlapping chunks"""
    chunks = []
    for doc_idx, doc in enumerate(documents):
        doc = re.sub(r'\s+', ' ', doc).strip()
        sentences = re.split(r'(?<=[.!?])\s+', doc)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append((' '.join(current_chunk), doc_idx))
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append((' '.join(current_chunk), doc_idx))
    
    return chunks

# ============================================================================
# LOAD MODELS (CACHED)
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_models():
    """Load and cache all models"""
    
    # Domain documents - REPLACE WITH YOUR OWN DATA
    domain_documents = [
        """Artificial Intelligence (AI) is transforming healthcare through predictive 
        analytics, personalized medicine, and diagnostic automation. Machine learning 
        models can analyze medical images with accuracy comparable to human radiologists, 
        while natural language processing helps extract insights from clinical notes.""",
        
        """Deep learning architectures like transformers have revolutionized natural 
        language processing. Models such as GPT, BERT, and their variants use attention 
        mechanisms to understand context and generate human-like text. These models are 
        pre-trained on large corpora and fine-tuned for specific tasks.""",
        
        """Retrieval-Augmented Generation (RAG) combines information retrieval with 
        generative models. It retrieves relevant documents from a knowledge base and 
        uses them as context for generating accurate, grounded responses. This approach 
        reduces hallucinations and enables models to access up-to-date information.""",
        
        """Quantization is a model compression technique that reduces memory footprint 
        and computational requirements. 4-bit quantization can reduce model size by 
        75% while maintaining most of the model's performance. Dynamic quantization 
        preserves precision for critical parameters.""",
        
        """Vector databases like FAISS, Pinecone, and Chroma enable efficient similarity 
        search over high-dimensional embeddings. They use techniques like approximate 
        nearest neighbor search to quickly retrieve relevant documents based on semantic 
        similarity rather than keyword matching.""",
        
        """Transfer learning allows models pre-trained on large datasets to be adapted 
        for specific tasks with limited data. Fine-tuning adjusts model parameters for 
        domain-specific applications, while approaches like LoRA enable parameter-efficient 
        fine-tuning by updating only a small subset of weights.""",
        
        """Large Language Models (LLMs) have emergent capabilities including in-context 
        learning, chain-of-thought reasoning, and few-shot adaptation. These models can 
        perform complex reasoning tasks when properly prompted, though they may still 
        produce factually incorrect information without proper grounding.""",
        
        """Embeddings are dense vector representations that capture semantic meaning. 
        Sentence transformers produce embeddings where semantically similar texts are 
        close in vector space. These embeddings enable semantic search, clustering, and 
        other downstream tasks in NLP applications.""",
    ]
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load LLM with 4-bit quantization
    model_name = "meta-llama/Llama-3.2-3B-Instruct"  # or "microsoft/phi-2" for smaller
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        # Fallback to smaller model if main model fails
        st.warning(f"Main model failed, using fallback model: {e}")
        model_name = "microsoft/phi-2"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = 384
    
    # Create chunks
    document_chunks = chunk_documents(domain_documents, chunk_size=200, overlap=30)
    chunk_texts = [chunk[0] for chunk in document_chunks]
    
    # Create embeddings
    chunk_embeddings = embedding_model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    # Build FAISS index
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(chunk_embeddings.astype('float32'))
    
    return model, tokenizer, embedding_model, index, document_chunks, model_name

# ============================================================================
# RAG FUNCTIONS
# ============================================================================

def retrieve_chunks(query: str, embedding_model, index, document_chunks, top_k: int = 3):
    """Retrieve relevant chunks"""
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    scores, indices = index.search(query_embedding.astype('float32'), top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= 0.3:
            chunk_text = document_chunks[idx][0]
            source_doc_idx = document_chunks[idx][1]
            results.append((chunk_text, float(score), source_doc_idx))
    
    return results

def generate_response(
    query: str,
    model,
    tokenizer,
    embedding_model,
    index,
    document_chunks,
    top_k: int = 3,
    max_tokens: int = 256,
    temperature: float = 0.7
):
    """Generate RAG response"""
    
    # Retrieve context
    retrieved = retrieve_chunks(query, embedding_model, index, document_chunks, top_k)
    
    if not retrieved:
        return {
            "response": "I don't have enough relevant information to answer this query.",
            "context": [],
            "query": query
        }
    
    # Build context
    context_parts = []
    for i, (chunk_text, score, doc_idx) in enumerate(retrieved, 1):
        context_parts.append(f"[Context {i}]: {chunk_text}")
    
    context = "\n\n".join(context_parts)
    
    # Create prompt
    prompt = f"""Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer (after "Answer:")
    if "Answer:" in full_response:
        response = full_response.split("Answer:")[-1].strip()
    else:
        response = full_response[len(prompt):].strip()
    
    return {
        "response": response,
        "context": retrieved,
        "query": query
    }

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🤖 RAG System with 4-bit Quantization</p>', unsafe_allow_html=True)
    st.markdown("### Retrieval-Augmented Generation powered by 4-bit quantized LLM")
    
    # Load models
    with st.spinner("🔄 Loading models... (This may take 1-2 minutes on first run)"):
        try:
            model, tokenizer, embedding_model, index, document_chunks, model_name = load_models()
            st.success(f"✅ Models loaded successfully! Using: {model_name}")
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            st.info("💡 This app requires GPU. Make sure you're using Hugging Face Spaces with GPU enabled.")
            st.stop()
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Settings
    top_k = st.sidebar.slider(
        "📚 Number of context chunks",
        min_value=1,
        max_value=5,
        value=3,
        help="How many relevant document chunks to retrieve"
    )
    
    max_tokens = st.sidebar.slider(
        "📝 Max response tokens",
        min_value=50,
        max_value=500,
        value=200,
        help="Maximum length of generated response"
    )
    
    temperature = st.sidebar.slider(
        "🌡️ Temperature",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    # System info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Info")
    
    st.sidebar.info(f"**Model**: {model_name.split('/')[-1]}")
    
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.sidebar.metric("GPU", torch.cuda.get_device_name(0))
        st.sidebar.metric("VRAM Usage", f"{vram_used:.2f} / {vram_total:.1f} GB")
    else:
        st.sidebar.warning("⚠️ No GPU detected. Performance will be slow.")
    
    st.sidebar.metric("Index Size", f"{index.ntotal} vectors")
    st.sidebar.metric("Embedding Dim", embedding_model.get_sentence_embedding_dimension())
    
    # Sample questions
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Sample Questions")
    sample_questions = [
        "What is Retrieval-Augmented Generation?",
        "How does 4-bit quantization work?",
        "What are vector databases?",
        "Explain transfer learning",
        "What are embeddings?"
    ]
    
    for i, question in enumerate(sample_questions):
        if st.sidebar.button(f"📌 {question}", key=f"sample_{i}"):
            st.session_state.query = question
    
    # Main interface
    st.markdown("---")
    
    # Query input
    query = st.text_area(
        "🔍 Enter your question:",
        height=100,
        placeholder="Type your question here...",
        value=st.session_state.get('query', ''),
        key='query_input'
    )
    
    # Buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        generate_button = st.button("🚀 Generate Answer", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.query = ""
            st.rerun()
    
    # Generate response
    if generate_button and query:
        with st.spinner("🔄 Generating response..."):
            start_time = time.time()
            
            try:
                result = generate_response(
                    query,
                    model,
                    tokenizer,
                    embedding_model,
                    index,
                    document_chunks,
                    top_k=top_k,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                duration = time.time() - start_time
                
                # Display results
                st.markdown("---")
                st.success(f"✅ Response generated in {duration:.2f} seconds")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Response Time", f"{duration:.2f}s")
                with col2:
                    st.metric("📚 Context Chunks", len(result['context']))
                with col3:
                    st.metric("📏 Response Length", f"{len(result['response'])} chars")
                
                # Answer
                st.markdown("### 💡 Answer")
                st.markdown(f"**{result['response']}**")
                
                # Context
                st.markdown("### 📚 Retrieved Context")
                for i, (text, score, doc_idx) in enumerate(result['context'], 1):
                    with st.expander(f"📄 Context {i} - Relevance: {score:.3f}"):
                        st.markdown(f"**Source Document:** {doc_idx}")
                        st.markdown(text)
                
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
                st.info("Try reducing max_tokens or using a smaller context window.")
    
    elif generate_button and not query:
        st.warning("⚠️ Please enter a question first!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            Built with ❤️ using 
            <a href='https://huggingface.co/transformers' target='_blank'>Transformers</a> • 
            <a href='https://streamlit.io' target='_blank'>Streamlit</a> • 
            <a href='https://github.com/facebookresearch/faiss' target='_blank'>FAISS</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
