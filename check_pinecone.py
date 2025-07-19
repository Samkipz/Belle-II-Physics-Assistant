#!/usr/bin/env python3
"""
Check Pinecone Index Status
===========================

This script checks the status of your Pinecone index and displays statistics.
"""

import os
from pinecone import Pinecone
from typing import Dict, Any


def check_pinecone_index(index_name: str = "belle2-advanced") -> Dict[str, Any]:
    """Check the status of a Pinecone index"""

    # Check API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        return {"error": "PINECONE_API_KEY environment variable not set"}

    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)

        # Check if index exists
        if index_name not in pc.list_indexes().names():
            return {"error": f"Index '{index_name}' not found"}

        # Connect to index
        index = pc.Index(index_name)

        # Get index stats
        stats = index.describe_index_stats()

        # Get sample data
        sample_results = index.query(
            vector=[0] * 1024,  # Dummy vector for sampling (updated from 768)
            top_k=100,
            include_metadata=True
        )

        # Analyze metadata
        chunk_types = {}
        documents = set()

        for match in sample_results.matches:
            metadata = match.metadata
            chunk_type = metadata.get('chunk_type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            documents.add(metadata.get('document', 'unknown'))

        return {
            "index_name": index_name,
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "metric": stats.metric,
            "chunk_types": chunk_types,
            "unique_documents": len(documents),
            "sample_size": len(sample_results.matches)
        }

    except Exception as e:
        return {"error": str(e)}


def main():
    """Main function"""
    print("🔍 Checking Pinecone Index Status")
    print("=" * 50)

    # Check the index
    result = check_pinecone_index("belle2-advanced")

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    # Display results
    print(f"✅ Index Name: {result['index_name']}")
    print(f"📊 Total Vectors: {result['total_vectors']:,}")
    print(f"🔢 Dimension: {result['dimension']}")
    print(f"📏 Metric: {result['metric']}")
    print(f"📄 Unique Documents: {result['unique_documents']}")
    print(f"🔍 Sample Size: {result['sample_size']}")

    # Display chunk types
    if result['chunk_types']:
        print("\n📋 Chunk Type Distribution:")
        for chunk_type, count in result['chunk_types'].items():
            print(f"   {chunk_type}: {count}")

    print("\n✅ Pinecone index is ready!")


if __name__ == "__main__":
    main()
