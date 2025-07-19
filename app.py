import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import os
import json
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Belle II Physics Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Header styling */
    .header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        min-height: 500px;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid #e0e0e0;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Sources styling */
    .sources-container {
        background: rgba(248, 249, 250, 0.9);
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .source-item {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 16px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Loading animation */
    .loading {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }
    
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots:after {
        content: '';
        animation: dots 1.5s steps(5, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
    
    /* Stats cards */
    .stats-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stats-label {
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Configurations
QA_FILE = "belle2_qa_detailed_full.jsonl"
CHROMA_COLLECTION_NAME = "belle2_qa"
CHROMA_DB_DIR = os.path.abspath("./chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
HUGGINGFACE_API_URL = "https://router.huggingface.co/v1/chat/completions"
HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")
TOP_K = 3


@st.cache_resource
def load_rag_system():
    """Load and initialize the RAG system"""
    with st.spinner("Loading Belle II knowledge base..."):
        # Load Q&A pairs
        qa_pairs = []
        with open(QA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                qa_pairs.append(item)

        # Initialize embedding model
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Prepare data
        questions = [item['question'] for item in qa_pairs]
        answers = [item['answer'] for item in qa_pairs]
        metadatas = [{"answer": ans} for ans in answers]
        ids = [f"qa_{i}" for i in range(len(qa_pairs))]

        # Generate embeddings
        embeddings = embed_model.encode(questions, convert_to_numpy=True)

        # Store in Chroma
        client = chromadb.Client(Settings(persist_directory=CHROMA_DB_DIR))
        collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        collection.add(
            embeddings=embeddings,
            documents=questions,
            metadatas=metadatas,
            ids=ids
        )

        return collection, embed_model, len(qa_pairs)


def embed_query(query, model):
    return model.encode([query], convert_to_numpy=True)[0]


def retrieve_top_k(query, collection, model, k=3):
    query_embedding = embed_query(query, model)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )
    return results


def compose_prompt_improved(user_query, retrieved):
    context_parts = []
    for i, (doc, metadata) in enumerate(zip(retrieved['documents'][0], retrieved['metadatas'][0]), 1):
        context_parts.append(f"Source {i}:\nQ: {doc}\nA: {metadata['answer']}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a Belle II physics assistant. Answer the user's question using ONLY the information provided in the context below. Do not use any external knowledge or make up information.

If the context doesn't contain enough information to answer the question completely, say so clearly.

Context:
{context}

User Question: {user_query}

Instructions:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information in my knowledge base to answer this question completely."
3. Do not add information that is not present in the context
4. Be accurate and precise

Answer:"""

    return prompt


def query_huggingface(prompt):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
               "Content-Type": "application/json"}
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 1000
    }
    response = requests.post(
        HUGGINGFACE_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    else:
        return f"[Error from model API]: {result}"


def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1 style="text-align: center; color: #333; margin-bottom: 0.5rem;">
            🔬 Belle II Physics Assistant
        </h1>
        <p style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 0;">
            Your AI-powered guide to Belle II particle physics experiments
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")

        # Load RAG system
        try:
            collection, embed_model, qa_count = load_rag_system()

            # Stats cards
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{qa_count}</div>
                    <div class="stats-label">Knowledge Base</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">🔬</div>
                    <div class="stats-label">Belle II</div>
                </div>
                """, unsafe_allow_html=True)

            st.success("✅ System Ready")

        except Exception as e:
            st.error(f"❌ System Error: {e}")
            st.stop()

        st.markdown("---")
        st.markdown("### 🎯 Quick Questions")

        quick_questions = [
            "How was integrated luminosity measured during Belle II Phase 2?",
            "What is the purpose of the beam-energy-constrained mass?",
            "How does inclusive tagging work?",
            "What are the main backgrounds in rare decay searches?"
        ]

        for question in quick_questions:
            if st.button(question, key=f"quick_{question[:20]}"):
                st.session_state.user_input = question
                st.rerun()

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This assistant uses advanced RAG (Retrieval-Augmented Generation) technology to provide accurate answers about Belle II physics experiments.
        
        **Features:**
        - 🔍 Semantic search across 16 Q&A pairs
        - 🤖 AI-powered answer generation
        - 📚 Source attribution
        - 🎯 Context-aware responses
        """)

    # Main chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize processed input tracking
    if "processed_inputs" not in st.session_state:
        st.session_state.processed_inputs = set()

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <strong>🔬 Belle II Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

            # Show sources if available
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, (doc, metadata) in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-item">
                            <strong>Source {i}:</strong><br>
                            <strong>Q:</strong> {doc}<br>
                            <strong>A:</strong> {metadata['answer'][:200]}...
                        </div>
                        """, unsafe_allow_html=True)

    # Chat input
    user_input = st.text_input(
        "Ask me about Belle II physics...",
        key="user_input",
        placeholder="e.g., How was integrated luminosity measured during Belle II Phase 2?"
    )

    # Process user input (only if not already processed)
    if user_input and user_input not in st.session_state.processed_inputs:
        # Add to processed inputs to prevent duplicates
        st.session_state.processed_inputs.add(user_input)

        # Add user message to chat
        st.session_state.messages.append(
            {"role": "user", "content": user_input})

        # Show loading animation
        with st.spinner("🔍 Searching knowledge base..."):
            # Retrieve relevant documents
            retrieved = retrieve_top_k(
                user_input, collection, embed_model, k=TOP_K)

        with st.spinner("🤖 Generating answer..."):
            # Generate answer
            prompt = compose_prompt_improved(user_input, retrieved)
            answer = query_huggingface(prompt)

        # Add assistant message to chat
        sources = list(zip(retrieved['documents']
                       [0], retrieved['metadatas'][0]))
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        # Rerun to update the chat
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🔬 Powered by RAG Technology | Belle II Physics Knowledge Base</p>
        <p style="font-size: 0.8rem;">Built with Streamlit, ChromaDB, and Hugging Face</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
