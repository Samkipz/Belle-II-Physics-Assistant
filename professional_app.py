import streamlit as st
import os
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List
import requests

# Import our advanced RAG system
from advanced_rag_system import AdvancedRAGSystem, ModelType, RetrievalResult

# Page configuration
st.set_page_config(
    page_title="Belle II Professional Physics Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Professional styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
    }
    
    /* Header styling */
    .header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        min-height: 600px;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #e0e0e0;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Sources styling */
    .sources-container {
        background: rgba(248, 249, 250, 0.95);
        border: 1px solid #e9ecef;
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .source-item {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease;
    }
    
    .source-item:hover {
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 1rem 2.5rem;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 30px;
        border: 2px solid #e0e0e0;
        padding: 1.2rem;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
    }
    
    /* Stats cards */
    .stats-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .stats-label {
        color: #666;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Confidence indicator */
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Model selector */
    .model-selector {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
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
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_system():
    """Load and initialize the advanced RAG system"""
    with st.spinner("🔬 Loading Belle II Professional Knowledge Base..."):
        try:
            CHROMA_DB_DIR = r"C:/Users/Sam/Desktop/Wrt/Random/codezn/cdzn/chroma_db"
            rag_system = AdvancedRAGSystem(
                chroma_db_dir=CHROMA_DB_DIR, collection_name="belle2_advanced")
            return rag_system
        except Exception as e:
            st.error(f"Failed to load RAG system: {e}")
            return None


def get_confidence_class(confidence: float) -> str:
    """Get CSS class for confidence level"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"


def get_confidence_text(confidence: float) -> str:
    """Get text description for confidence level"""
    if confidence >= 0.7:
        return "High Confidence"
    elif confidence >= 0.4:
        return "Medium Confidence"
    else:
        return "Low Confidence"


def create_analytics_dashboard(stats: Dict[str, Any]):
    """Create analytics dashboard"""
    st.markdown("### 📊 Knowledge Base Analytics")

    # Create columns for stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats.get('total_chunks', 0):,}</div>
            <div class="stats-label">Total Chunks</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats.get('unique_documents', 0)}</div>
            <div class="stats-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        chunk_types = stats.get('chunk_type_distribution', {})
        total_types = len(chunk_types)
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{total_types}</div>
            <div class="stats-label">Content Types</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">🔬</div>
            <div class="stats-label">Belle II</div>
        </div>
        """, unsafe_allow_html=True)

    # Create chart for chunk type distribution
    if 'chunk_type_distribution' in stats and stats['chunk_type_distribution']:
        st.markdown("### 📈 Content Type Distribution")

        chunk_data = stats['chunk_type_distribution']
        df = pd.DataFrame(list(chunk_data.items()), columns=['Type', 'Count'])

        fig = px.pie(df, values='Count', names='Type',
                     title="Distribution of Content Types",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1 style="text-align: center; color: #333; margin-bottom: 0.5rem; font-size: 2.5rem;">
            🔬 Belle II Professional Physics Assistant
        </h1>
        <p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 0;">
            Advanced AI-powered research assistant for Belle II particle physics experiments
        </p>
        <p style="text-align: center; color: #888; font-size: 1rem; margin-top: 0.5rem;">
            Powered by 57 scientific documents • Multi-model RAG system • Professional-grade accuracy
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")

        # Load RAG system
        rag_system = load_rag_system()

        if rag_system:
            # Get system stats
            stats = rag_system.get_system_stats()

            if 'error' not in stats:
                st.success("✅ System Ready")

                # Display basic stats
                st.markdown(
                    f"**Total Chunks:** {stats.get('total_chunks', 0):,}")
                st.markdown(
                    f"**Documents:** {stats.get('unique_documents', 0)}")
                st.markdown(
                    f"**Embedding Model:** {stats.get('embedding_model', 'Unknown')}")

                # Model selection
                st.markdown("### 🤖 Model Selection")
                model_options = {
                    "Mistral-7B": ModelType.MISTRAL_7B,
                    "Llama-2": ModelType.LLAMA_2,
                    "Vicuna": ModelType.VICUNA
                }

                selected_model_name = st.selectbox(
                    "Choose Foundation Model:",
                    list(model_options.keys()),
                    index=0
                )
                selected_model = model_options[selected_model_name]

                # Retrieval strategy
                st.markdown("### 🔍 Retrieval Strategy")
                strategy = st.selectbox(
                    "Choose Retrieval Method:",
                    ["hybrid", "semantic", "multi_vector"],
                    index=0,
                    help="Hybrid combines semantic and keyword search for best results"
                )

                # Advanced options
                st.markdown("### ⚡ Advanced Options")
                top_k = st.slider("Number of Sources:", 3, 10, 5)
                temperature = st.slider(
                    "Creativity Level:", 0.0, 1.0, 0.1, 0.1)

                # Filters
                st.markdown("### 🔧 Filters")
                chunk_type_filter = st.selectbox(
                    "Content Type Filter:",
                    ["All", "text", "table", "equation", "figure"],
                    index=0
                )

                if chunk_type_filter == "All":
                    chunk_type_filter = None

            else:
                st.error(f"❌ System Error: {stats['error']}")
                st.stop()
        else:
            st.error("❌ Failed to load RAG system")
            st.stop()

        st.markdown("---")
        st.markdown("### 🎯 Quick Questions")

        quick_questions = [
            "How is integrated luminosity measured in Belle II?",
            "What are the main backgrounds in rare decay searches?",
            "How does inclusive tagging work?",
            "What is the beam-energy-constrained mass?",
            "What are the systematic uncertainties in luminosity measurement?",
            "How does the electromagnetic calorimeter work?"
        ]

        for question in quick_questions:
            if st.button(question, key=f"quick_{hash(question)}"):
                st.session_state.user_input = question
                st.rerun()

        st.markdown("---")
        st.markdown("### 📚 About")
        st.markdown("""
        **Professional Features:**
        - 🔬 57 scientific documents
        - 🤖 Multi-model support
        - 🔍 Advanced retrieval strategies
        - 📊 Real-time analytics
        - 🎯 Confidence scoring
        - 📈 Content type analysis
        """)

    # Main content area
    if rag_system and 'error' not in stats:
        # Analytics Dashboard
        create_analytics_dashboard(stats)

        # Chat Interface
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

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
                # Display assistant message with confidence
                confidence_class = get_confidence_class(
                    message.get("confidence", 0))
                confidence_text = get_confidence_text(
                    message.get("confidence", 0))

                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🔬 Belle II Assistant:</strong><br>
                    {message["content"]}
                    <br><br>
                    <small>
                        <span class="{confidence_class}">Confidence: {confidence_text} ({message.get('confidence', 0):.1%})</span> | 
                        Model: {message.get('model', 'Unknown')} | 
                        Time: {message.get('processing_time', 0):.2f}s
                    </small>
                </div>
                """, unsafe_allow_html=True)

                # Show sources if available
                if "sources" in message:
                    with st.expander(f"📚 View Sources ({len(message['sources'])} found)"):
                        for i, source in enumerate(message["sources"], 1):
                            chunk_type_emoji = {
                                "text": "📄",
                                "table": "📊",
                                "equation": "🧮",
                                "figure": "📈"
                            }.get(source.chunk_type, "📄")

                            st.markdown(f"""
                            <div class="source-item">
                                <strong>{chunk_type_emoji} Source {i}:</strong><br>
                                <strong>Document:</strong> {source.document}<br>
                                <strong>Page:</strong> {source.page_number}<br>
                                <strong>Type:</strong> {source.chunk_type}<br>
                                <strong>Similarity:</strong> {source.similarity_score:.2f}<br>
                                <strong>Content:</strong> {source.content[:300]}...
                            </div>
                            """, unsafe_allow_html=True)

        # Chat input
        user_input = st.text_input(
            "Ask me about Belle II physics research...",
            key="user_input",
            placeholder="e.g., How is integrated luminosity measured in Belle II experiments?"
        )

        # Process user input
        if user_input and user_input not in st.session_state.processed_inputs:
            st.session_state.processed_inputs.add(user_input)
            st.session_state.messages.append(
                {"role": "user", "content": user_input})

            # Show loading animation
            with st.spinner("🔍 Searching knowledge base..."):
                # Retrieve relevant documents
                results = rag_system.retrieve(
                    user_input,
                    strategy=strategy,
                    top_k=top_k,
                    chunk_type_filter=chunk_type_filter
                )

            with st.spinner("🤖 Generating answer..."):
                # Generate answer
                response = rag_system.generate_answer(
                    user_input,
                    results,
                    model_type=selected_model,
                    temperature=temperature
                )

            # Add assistant message to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "sources": results,
                "confidence": response.confidence_score,
                "model": response.model_used,
                "processing_time": response.processing_time,
                "chunk_types": response.chunk_types_used
            })

            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>🔬 Belle II Professional Physics Assistant | Advanced RAG Technology</p>
            <p style="font-size: 0.8rem;">Built with Streamlit, ChromaDB, and Multi-Model AI</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
