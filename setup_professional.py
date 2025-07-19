#!/usr/bin/env python3
"""
Professional Belle II Physics Assistant Setup Script
====================================================

This script will help you set up the professional version of the Belle II Physics Assistant
with support for 57 scientific documents.

Usage:
    python setup_professional.py

Requirements:
    - 50+ PDF documents in a 'documents' folder
    - Hugging Face API key set as environment variable HF_API_KEY
    - Sufficient disk space for embeddings and processed data
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import json

CHROMA_DB_DIR = os.path.abspath("chroma_db")


def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False

    # Check if documents folder exists
    documents_dir = Path("documents")
    if not documents_dir.exists():
        print("❌ 'documents' folder not found")
        print("   Please create a 'documents' folder and place your 57 PDF files there")
        return False

    # Count PDF files
    pdf_files = list(documents_dir.glob("*.pdf"))
    if len(pdf_files) == 0:
        print("❌ No PDF files found in 'documents' folder")
        return False

    print(f"✅ Found {len(pdf_files)} PDF files")

    # Check API key
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        print("❌ HF_API_KEY environment variable not set")
        print("   Please set your Hugging Face API key:")
        print("   export HF_API_KEY=your_api_key_here")
        return False

    print("✅ Hugging Face API key found")
    return True


def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")

    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements_professional.txt"
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def process_documents():
    """Process all documents using the document processor, with checkpointing."""
    output_file = "belle2_processed_chunks.jsonl"
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(
            f"✅ Found existing '{output_file}', skipping document processing.")
        return True
    print("\n📄 Processing documents...")
    try:
        from document_processor import ScientificDocumentProcessor

        processor = ScientificDocumentProcessor()
        all_chunks = processor.process_all_documents("documents")
        processor.save_processed_data(all_chunks, output_file)

        total_chunks = sum(len(chunks) for chunks in all_chunks.values())
        print(f"✅ Processing complete!")
        print(f"   Documents processed: {len(all_chunks)}")
        print(f"   Total chunks created: {total_chunks}")

        chunk_types = {}
        for chunks in all_chunks.values():
            for chunk in chunks:
                chunk_types[chunk.chunk_type] = chunk_types.get(
                    chunk.chunk_type, 0) + 1

        print("\n   Chunk type breakdown:")
        for chunk_type, count in chunk_types.items():
            print(f"     {chunk_type}: {count}")

        return True

    except Exception as e:
        print(f"❌ Failed to process documents: {e}")
        return False


def setup_rag_system():
    """Set up the RAG system with processed documents, and confirm API key visibility."""
    print("\n🤖 Setting up RAG system...")

    # Print (masked) API key for debug
    api_key = os.getenv("HF_API_KEY")
    if api_key:
        print(f"HF_API_KEY detected: {api_key[:6]}...{api_key[-4:]}")
    else:
        print("❌ HF_API_KEY not found in environment at RAG setup step.")

    try:
        from advanced_rag_system import AdvancedRAGSystem

        rag_system = AdvancedRAGSystem(
            chroma_db_dir=CHROMA_DB_DIR, collection_name="belle2_advanced")
        rag_system.ingest_documents("belle2_processed_chunks.jsonl")
        stats = rag_system.get_system_stats()

        if 'error' not in stats:
            print("✅ RAG system setup complete!")
            print(f"   Total chunks: {stats.get('total_chunks', 0):,}")
            print(f"   Documents: {stats.get('unique_documents', 0)}")
            print(
                f"   Embedding model: {stats.get('embedding_model', 'Unknown')}")
            return True
        else:
            print(f"❌ RAG system setup failed: {stats['error']}")
            return False

    except Exception as e:
        print(f"❌ Failed to setup RAG system: {e}")
        return False


def create_launch_scripts():
    """Create launch scripts for easy startup"""
    print("\n🚀 Creating launch scripts...")

    # Create batch file for Windows
    with open("run_professional.bat", "w") as f:
        f.write("@echo off\n")
        f.write("echo Starting Belle II Professional Physics Assistant...\n")
        f.write("streamlit run professional_app.py\n")
        f.write("pause\n")

    # Create shell script for Linux/Mac
    with open("run_professional.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'Starting Belle II Professional Physics Assistant...'\n")
        f.write("streamlit run professional_app.py\n")

    # Make shell script executable
    os.chmod("run_professional.sh", 0o755)

    print("✅ Launch scripts created:")
    print("   - run_professional.bat (Windows)")
    print("   - run_professional.sh (Linux/Mac)")


def create_config_file():
    """Create a configuration file for the system"""
    print("\n⚙️ Creating configuration file...")

    config = {
        "system_name": "Belle II Professional Physics Assistant",
        "version": "2.0.0",
        "description": "Advanced RAG system for Belle II physics research",
        "features": [
            "Multi-model support (Mistral, Llama-2, Vicuna)",
            "Advanced retrieval strategies",
            "Real-time analytics",
            "Confidence scoring",
            "Content type analysis"
        ],
        "document_count": len(list(Path("documents").glob("*.pdf"))),
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "default_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "setup_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open("system_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Configuration file created: system_config.json")


def main():
    """Main setup function"""
    print("🔬 Belle II Professional Physics Assistant Setup")
    print("=" * 50)

    # Step 1: Check requirements
    if not check_requirements():
        print("\n❌ Setup failed. Please fix the issues above and try again.")
        return False

    # Step 2: Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed. Please check your internet connection and try again.")
        return False

    # Step 3: Process documents
    if not process_documents():
        print("\n❌ Setup failed. Please check your PDF files and try again.")
        return False

    # Step 4: Setup RAG system
    if not setup_rag_system():
        print("\n❌ Setup failed. Please check your API key and try again.")
        return False

    # Step 5: Create launch scripts
    create_launch_scripts()

    # Step 6: Create config file
    create_config_file()

    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Run the application:")
    print("      Windows: run_professional.bat")
    print("      Linux/Mac: ./run_professional.sh")
    print("   2. Open your browser to http://localhost:8501")
    print("   3. Start asking questions about Belle II physics!")

    print("\n📊 System Overview:")
    print("   - Professional-grade RAG system")
    print("   - Multi-model AI support")
    print("   - Advanced analytics dashboard")
    print("   - Confidence scoring")
    print("   - Real-time performance monitoring")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
