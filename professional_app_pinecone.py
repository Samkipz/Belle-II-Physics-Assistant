from typing import Dict, Any
import pinecone
import streamlit as st
import os
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Import the ModelType enum from the RAG system
from advanced_rag_system_pinecone import ModelType

# Page configuration
st.set_page_config(
    page_title="Belle II Professional Physics Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_stats' not in st.session_state:
    st.session_state.system_stats = None


@st.cache_resource
def load_rag_system():
    """Load and initialize the advanced RAG system with Pinecone"""
    with st.spinner("🔬 Loading Belle II Professional Knowledge Base..."):
        try:
            from advanced_rag_system_pinecone import AdvancedRAGSystem

            rag_system = AdvancedRAGSystem(
                index_name="belle2-advanced",
                use_integrated_embeddings=True
            )
            return rag_system
        except Exception as e:
            st.error(f"Failed to load RAG system: {e}")
            return None


def get_confidence_class(confidence):
    """Get CSS class for confidence score"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"


def display_source_card(source):
    """Display a source card with metadata"""
    with st.container():
        st.markdown(f"""
        <div class="source-card">
            <strong>📄 {source.document}</strong> (Page {source.page_number})<br>
            <small>Type: {source.chunk_type} | Score: {source.similarity_score:.3f}</small><br>
            <details>
                <summary>View Content</summary>
                <pre style="font-size: 0.8rem; white-space: pre-wrap;">{source.content[:300]}{'...' if len(source.content) > 300 else ''}</pre>
            </details>
        </div>
        """, unsafe_allow_html=True)


def create_analytics_dashboard(stats):
    """Create analytics dashboard"""
    st.markdown("### 📊 System Analytics")

    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Chunks", f"{stats.get('total_chunks', 0):,}")

    with col2:
        st.metric("Documents", stats.get('unique_documents', 0))

    with col3:
        st.metric("Embedding Model", stats.get('embedding_model', 'Unknown'))

    with col4:
        st.metric("Index Name", stats.get('index_name', 'Unknown'))

    # Chunk type distribution
    chunk_types = stats.get('chunk_type_distribution', {})
    if chunk_types:
        st.markdown("#### Chunk Type Distribution")
        fig = px.pie(
            values=list(chunk_types.values()),
            names=list(chunk_types.keys()),
            title="Distribution of Content Types"
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Belle II Professional Physics Assistant</h1>',
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced RAG System with Pinecone Vector Database</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # Model selection with proper mapping to ModelType enum
        model_options = {
            "Mistral-7B": ModelType.MISTRAL_7B,
            "Llama-2-7B": ModelType.LLAMA_2,
            "Vicuna-7B": ModelType.VICUNA
        }
        selected_model_name = st.selectbox(
            "Select Model",
            list(model_options.keys()),
            index=0,
            help="Choose the language model for answer generation"
        )
        selected_model = model_options[selected_model_name]

        # Retrieval strategy
        strategy_options = ["hybrid", "semantic", "multi_vector"]
        selected_strategy = st.selectbox(
            "Retrieval Strategy",
            strategy_options,
            index=0,
            help="Hybrid: Combines semantic and keyword search\nSemantic: Pure embedding-based search\nMulti-vector: Advanced multi-vector retrieval"
        )

        # Number of results
        top_k = st.slider("Number of Results (top_k)", 1, 20, 5)

        # Temperature
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1,
                                help="Higher values make responses more creative, lower values more focused")

        # Filters
        st.markdown("### 🔍 Filters")
        chunk_type_filter = st.selectbox(
            "Chunk Type Filter",
            ["All", "text", "table", "equation", "figure"],
            index=0,
            help="Filter results by content type"
        )
        chunk_type_filter = None if chunk_type_filter == "All" else chunk_type_filter

        # System stats
        st.markdown("### 📈 System Status")
        if st.button("🔄 Refresh Stats"):
            st.session_state.system_stats = None

        if st.session_state.system_stats is None:
            rag_system = load_rag_system()
            if rag_system:
                st.session_state.system_stats = rag_system.get_system_stats()

        if st.session_state.system_stats:
            stats = st.session_state.system_stats
            if 'error' not in stats:
                st.metric("Total Chunks", f"{stats.get('total_chunks', 0):,}")
                st.metric("Documents", stats.get('unique_documents', 0))
                st.metric("Index", stats.get('index_name', 'Unknown'))
            else:
                st.error(f"Error loading stats: {stats['error']}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 💬 Chat Interface")

        # Display current configuration
        st.info(
            f"🤖 **Model**: {selected_model_name} | 🔍 **Strategy**: {selected_strategy} | 📊 **Results**: {top_k} | 🌡️ **Temperature**: {temperature}")

        # Chat input
        user_query = st.text_area(
            "Ask a question about Belle II physics:",
            height=100,
            placeholder="e.g., How is integrated luminosity measured in Belle II experiments?"
        )

        # Query button
        if st.button("🔍 Query Knowledge Base", type="primary"):
            if user_query.strip():
                with st.spinner("🔍 Searching knowledge base..."):
                    try:
                        rag_system = load_rag_system()
                        if rag_system:
                            # Retrieve relevant documents using selected strategy
                            retrieval_results = rag_system.retrieve(
                                user_query,
                                strategy=selected_strategy,
                                top_k=top_k,
                                chunk_type_filter=chunk_type_filter
                            )

                            if retrieval_results:
                                # Generate answer using selected model
                                with st.spinner("🤖 Generating answer..."):
                                    response = rag_system.generate_answer(
                                        user_query,
                                        retrieval_results,
                                        model_type=selected_model,  # Pass the selected model
                                        temperature=temperature
                                    )

                                # Display answer
                                st.markdown("### 🤖 Answer")
                                st.markdown(response.answer)

                                # Display confidence and metadata
                                confidence_class = get_confidence_class(
                                    response.confidence_score)
                                st.markdown(f"""
                                <div class="metric-card">
                                    <strong>Confidence Score:</strong>
                                    <span class="{confidence_class}">{response.confidence_score:.3f}</span><br>
                                    <strong>Processing Time:</strong> {response.processing_time:.2f}s<br>
                                    <strong>Model Used:</strong> {response.model_used}<br>
                                    <strong>Sources Used:</strong> {len(response.sources)}<br>
                                    <strong>Chunk Types:</strong> {', '.join(response.chunk_types_used)}
                                </div>
                                """, unsafe_allow_html=True)

                                # Add to chat history
                                st.session_state.chat_history.append({
                                    'query': user_query,
                                    'answer': response.answer,
                                    'confidence': response.confidence_score,
                                    'sources': response.sources,
                                    'model_used': response.model_used,
                                    'strategy_used': selected_strategy,
                                    'timestamp': datetime.now()
                                })

                                # Display sources
                                st.markdown("### 📚 Sources")
                                for source in response.sources:
                                    display_source_card(source)
                            else:
                                st.warning(
                                    "No relevant documents found for your query.")
                        else:
                            st.error("Failed to load RAG system.")
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
            else:
                st.warning("Please enter a question.")

    with col2:
        st.markdown("### 📋 Recent Queries")

        if st.session_state.chat_history:
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q: {chat['query'][:50]}..."):
                    st.markdown(f"**Answer:** {chat['answer'][:200]}...")
                    st.markdown(f"**Confidence:** {chat['confidence']:.3f}")
                    st.markdown(
                        f"**Model:** {chat.get('model_used', 'Unknown')}")
                    st.markdown(
                        f"**Strategy:** {chat.get('strategy_used', 'Unknown')}")
                    st.markdown(
                        f"**Time:** {chat['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.info("No queries yet. Start by asking a question!")

    # Analytics section
    if st.session_state.system_stats and 'error' not in st.session_state.system_stats:
        st.markdown("---")
        create_analytics_dashboard(st.session_state.system_stats)


if __name__ == "__main__":
    main()
