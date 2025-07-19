#!/usr/bin/env python3
"""
Reset Pinecone Index
===================

This script deletes and recreates the Pinecone index with correct dimensions.
"""

import os
from pinecone import Pinecone, ServerlessSpec
import time


def reset_pinecone_index(index_name: str = "belle2-advanced"):
    """Delete and recreate the Pinecone index"""

    # Check API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not set")
        return False

    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)

        # Check if index exists
        if index_name in pc.list_indexes().names():
            print(f"🗑️  Deleting existing index: {index_name}")
            pc.delete_index(index_name)
            time.sleep(10)  # Wait for deletion to complete

        # Create new index with correct dimensions
        print(f"🔨 Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1024,  # For BAAI/bge-large-en-v1.5
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Wait for index to be ready
        print("⏳ Waiting for index to be ready...")
        time.sleep(15)

        print("✅ Index reset successfully!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("🔄 Resetting Pinecone Index")
    print("=" * 30)

    success = reset_pinecone_index("belle2-advanced")

    if success:
        print("\n✅ Index reset complete!")
        print("You can now run: python setup_professional_pinecone.py")
    else:
        print("\n❌ Index reset failed!")
