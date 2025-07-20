#!/usr/bin/env python3
"""
Cleanup ChromaDB Files
=====================

This script removes all ChromaDB-related files and old test files
since we're now using Pinecone.
"""

import os
import shutil
from pathlib import Path


def cleanup_chroma_files():
    """Remove all ChromaDB-related files and old test files"""

    print("🧹 Cleaning up ChromaDB and old test files...")

    # Files to remove (ChromaDB-related)
    chroma_files = [
        "chroma_db/",  # ChromaDB database directory
        "check_chroma.py",  # ChromaDB check script
        "advanced_rag_system.py",  # Original ChromaDB RAG system
        "setup_professional.py",  # Original ChromaDB setup
        "professional_app.py",  # Original ChromaDB Streamlit app
        "system_config.json",  # Original config
        "run_professional.sh",  # Original launch script
        "run_professional.bat",  # Original launch script
        "ingest_to_chroma.py",  # ChromaDB ingestion script
    ]

    # Old test files to remove
    test_files = [
        "test_complete_rag.py",
        "test_prompt_only.py",
        "quick_test.py",
        "test_full_rag.py",
        "test_retrieval.py",
        "rag_cli_chat.py",
        "rag_cli_chat_fixed.py",
        "rag_cli_chat_improved.py",
        "test_stats.py",
        "belle2_qa_detailed_full.jsonl",
    ]

    # Old app files
    old_app_files = [
        "app.py",  # Old basic app
        "run_test.bat",
        "run_streamlit.bat",
    ]

    # Old requirements
    old_requirements = [
        # Old requirements (keep requirements_professional.txt)
        "requirements.txt",
    ]

    # Combine all files to remove
    files_to_remove = chroma_files + test_files + old_app_files + old_requirements

    removed_count = 0

    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"🗑️  Removed directory: {file_path}")
                else:
                    os.remove(file_path)
                    print(f"🗑️  Removed file: {file_path}")
                removed_count += 1
            else:
                print(f"ℹ️  File not found (already removed): {file_path}")
        except Exception as e:
            print(f"⚠️  Error removing {file_path}: {e}")

    # Clean up __pycache__ directories
    print("\n🧹 Cleaning up Python cache files...")
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            cache_dir = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(cache_dir)
                print(f"🗑️  Removed cache: {cache_dir}")
                removed_count += 1
            except Exception as e:
                print(f"⚠️  Error removing cache {cache_dir}: {e}")

    print(f"\n✅ Cleanup complete! Removed {removed_count} files/directories")

    # Show what remains
    print("\n📋 Remaining Pinecone files:")
    pinecone_files = [
        "advanced_rag_system_pinecone.py",
        "setup_professional_pinecone.py",
        "professional_app_pinecone.py",
        "check_pinecone.py",
        "test_pinecone_setup.py",
        "reset_pinecone_index.py",
        "system_config_pinecone.json",
        "run_professional_pinecone.sh",
        "run_professional_pinecone.bat",
        "requirements_professional.txt",
        "belle2_processed_chunks.jsonl",
        "document_processor.py",
        "README_PROFESSIONAL.md",
    ]

    for file_path in pinecone_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")


if __name__ == "__main__":
    print("🚀 ChromaDB Cleanup Script")
    print("=" * 40)

    # Ask for confirmation
    response = input(
        "Are you sure you want to remove all ChromaDB files? (y/N): ")
    if response.lower() in ['y', 'yes']:
        cleanup_chroma_files()
        print("\n🎉 Cleanup completed! Your project is now Pinecone-only.")
    else:
        print("❌ Cleanup cancelled.")
