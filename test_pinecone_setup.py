#!/usr/bin/env python3
"""
Test Pinecone Setup
==================

This script tests the current Pinecone setup and displays statistics.
"""

import os
from advanced_rag_system_pinecone import AdvancedRAGSystem


def test_pinecone_setup():
    """Test the Pinecone setup"""
    print("🔍 Testing Pinecone Setup...")

    # Check if environment variables are set
    if not os.getenv("HF_API_KEY"):
        print("❌ Error: HF_API_KEY environment variable not set")
        return False

    if not os.getenv("PINECONE_API_KEY"):
        print("❌ Error: PINECONE_API_KEY environment variable not set")
        return False

    try:
        # Initialize RAG system
        rag_system = AdvancedRAGSystem(
            index_name="belle2-advanced",
            use_integrated_embeddings=False
        )

        # Get system stats
        stats = rag_system.get_system_stats()

        if 'error' in stats:
            print(f"❌ Error: {stats['error']}")
            return False

        # Display stats
        print(f"✅ Total Chunks: {stats.get('total_chunks', 0):,}")
        print(f"✅ Documents: {stats.get('unique_documents', 0)}")
        print(f"✅ Index Name: {stats.get('index_name', 'Unknown')}")
        print(f"✅ Embedding Model: {stats.get('embedding_model', 'Unknown')}")

        # Display chunk types
        chunk_types = stats.get('chunk_type_distribution', {})
        if chunk_types:
            print("\n📋 Chunk Type Distribution:")
            for chunk_type, count in chunk_types.items():
                print(f"   {chunk_type}: {count}")

        # Test a simple query
        print("\n🔍 Testing Query...")
        query = "How is integrated luminosity measured in Belle II experiments?"
        results = rag_system.retrieve(query, strategy="hybrid", top_k=3)

        if results:
            print(f"✅ Query successful! Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(
                    f"   {i}. {result.document} (p{result.page_number}) - Score: {result.similarity_score:.3f}")
        else:
            print("❌ No results found for query")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Testing Pinecone Setup")
    print("=" * 30)

    success = test_pinecone_setup()

    if success:
        print("\n✅ Pinecone setup test passed!")
    else:
        print("\n❌ Pinecone setup test failed!")
