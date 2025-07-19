import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

# Configurations
CHROMA_COLLECTION_NAME = "belle2_qa"
CHROMA_DB_DIR = os.path.abspath("./chroma_db")  # Use absolute path
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
TOP_K = 3


def get_chroma_collection():
    print(f"Initializing ChromaDB with directory: {CHROMA_DB_DIR}")
    client = chromadb.Client(Settings(persist_directory=CHROMA_DB_DIR))
    return client.get_or_create_collection(CHROMA_COLLECTION_NAME)


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


def test_retrieval():
    print("Testing RAG Retrieval System")
    print("=" * 50)

    # Initialize
    collection = get_chroma_collection()
    print(f"Collection count: {collection.count()}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Test queries
    test_queries = [
        "How was integrated luminosity measured during Belle II Phase 2, and what are the key processes used?",
        "What is the purpose of the beam-energy-constrained mass?",
        "How does inclusive tagging work?",
        "What are the main backgrounds in rare decay searches?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\nTest Query {i}: {query}")
        print("-" * 40)

        try:
            retrieved = retrieve_top_k(query, collection, embed_model, k=TOP_K)

            print(f"Retrieved {len(retrieved['documents'][0])} documents:")
            for j, (doc, metadata) in enumerate(zip(retrieved['documents'][0], retrieved['metadatas'][0]), 1):
                print(f"\n{j}. Question: {doc}")
                print(f"   Answer: {metadata['answer'][:200]}...")

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 50)
    print("Retrieval test completed!")


if __name__ == "__main__":
    test_retrieval()
