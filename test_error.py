#!/usr/bin/env python3
"""
Test Error Reproduction
=======================

This script reproduces the exact error to see what's causing the ModelType enum issue.
"""

import os
import traceback


def test_error():
    """Test to reproduce the error"""
    print("🔍 Testing Error Reproduction...")

    # Check environment variables
    hf_api_key = os.getenv("HF_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    print(f"HF_API_KEY set: {'Yes' if hf_api_key else 'No'}")
    print(f"PINECONE_API_KEY set: {'Yes' if pinecone_api_key else 'No'}")

    if not hf_api_key or not pinecone_api_key:
        print("❌ Missing environment variables")
        return False

    try:
        from advanced_rag_system_pinecone import AdvancedRAGSystem, ModelType

        # Initialize RAG system
        print("\n🔧 Initializing RAG system...")
        rag_system = AdvancedRAGSystem(
            index_name="belle2-advanced",
            use_integrated_embeddings=False
        )
        print("✅ RAG system initialized")

        # Test query
        query = "How was integrated luminosity measured during Belle II Phase 2, and what are the key processes used?"

        print(f"\n📝 Query: {query}")

        # Retrieve documents
        print("\n🔍 Retrieving documents...")
        retrieval_results = rag_system.retrieve(
            query, strategy="hybrid", top_k=3)

        print(f"✅ Retrieved {len(retrieval_results)} documents")

        if not retrieval_results:
            print("❌ No retrieval results found")
            return False

        # Generate answer - this is where the error occurs
        print("\n🤖 Generating answer...")
        try:
            response = rag_system.generate_answer(
                query,
                retrieval_results,
                model_type=ModelType.MISTRAL_7B,
                temperature=0.1
            )
            print("✅ Answer generated successfully!")
            print(f"Answer: {response.answer[:200]}...")
            return True

        except Exception as e:
            print(f"❌ Error in generate_answer: {e}")
            print(f"Error type: {type(e)}")
            print(f"Error args: {e.args}")
            print("\n🔍 Full traceback:")
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Testing Error Reproduction")
    print("=" * 40)

    success = test_error()

    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
