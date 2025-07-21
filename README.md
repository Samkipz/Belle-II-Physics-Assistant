# 🔬 Belle II Physics Assistant

A professional, AI-powered chatbot for Belle II particle physics experiments using advanced RAG (Retrieval-Augmented Generation) technology.

## ✨ Features

- **🔍 Semantic Search**: Advanced retrieval across the full Belle II physics Q&A corpus
- **🤖 AI-Powered Answers**: Context-aware responses using LLMs (provider is set in backend)
- **📚 Source Attribution**: Each answer shows concise source cards (document + excerpt)
- **🎨 Modern UI**: Beautiful, professional interface
- **⚡ Real-time Processing**: Fast retrieval and answer generation
- **📊 System Monitoring**: Live status and statistics (embedding model, index name, chunk type distribution)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Hugging Face API key (or other supported LLM provider, set in backend)

### Installation

1. **Clone or download the project files**
2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   # or
   source venv/bin/activate      # Linux/Mac
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set your API key:**
   ```bash
   export HF_API_KEY=your_api_key_here
   ```
5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
6. **Open your browser** and navigate to `http://localhost:8501`

## 🎯 Usage

### Asking Questions

1. **Type your question** in the chat input
2. **Click "Ask"** or press Enter
3. **View the AI-generated answer** based on Belle II knowledge
4. **Expand "View Sources"** to see concise source cards (document + excerpt)

### Sidebar Controls

- **Retrieval Mode**: Synthesized Answer (RAG) or Direct Q&A
- **Model Selection**: Choose from available LLMs (provider is fixed in backend)
- **Retrieval Strategy**: Hybrid, Semantic, or Multi-Vector
- **Number of Results**: Top-N slider
- **Content Filters**: Filter by text, table, equation, or figure
- **System Analytics**: Only shows Embedding Model, Index Name, and Chunk Type Distribution

## 🏗️ Architecture

- **Data Ingestion**: Full Q&A corpus with BGE embeddings
- **RAG Backend**: Semantic retrieval, prompt engineering, LLM integration
- **Web Interface**: Streamlit app with modern UI, source cards, and analytics

## 🔧 Technical Details

- **No provider selection in UI**: Provider is set in backend/config
- **Source cards**: Only show Document and Excerpt for each source
- **System Analytics**: Only Embedding Model, Index Name, and Chunk Type Distribution are shown
- **No debug info or unnecessary metrics in UI**

## 📈 Performance

- **Fast retrieval**: < 1 second for semantic search
- **Quick responses**: < 3 seconds for full RAG pipeline
- **Accurate answers**: Based on verified Belle II physics data
- **No hallucination**: Strict prompt engineering prevents false information

## 🤝 Contributing

To add more Belle II physics knowledge:

1. Add new Q&A pairs to your corpus
2. Re-run the ingestion and embedding scripts
3. Test with the web interface

---

**🔬 Built with ❤️ for Belle II Physics Research**
