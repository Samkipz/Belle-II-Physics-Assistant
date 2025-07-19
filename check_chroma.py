import chromadb
from chromadb.config import Settings
import os

# Configurations
CHROMA_COLLECTION_NAME = "belle2_advanced"
CHROMA_DB_DIR = os.path.abspath("chroma_db")


def check_chroma():
    print("Checking ChromaDB Status")
    print("=" * 50)
    print(f"[DEBUG] Checking ChromaDB directory: {CHROMA_DB_DIR}")

    try:
        # Initialize client
        print(f"Initializing ChromaDB with directory: {CHROMA_DB_DIR}")
        client = chromadb.Client(Settings(persist_directory=CHROMA_DB_DIR))
        print(f"ChromaDB client initialized successfully")

        # List all collections
        collections = client.list_collections()
        print(f"Available collections: {[col.name for col in collections]}")

        # Check if our collection exists
        if CHROMA_COLLECTION_NAME in [col.name for col in collections]:
            collection = client.get_collection(CHROMA_COLLECTION_NAME)
            print(f"Collection '{CHROMA_COLLECTION_NAME}' found!")

            # Get collection info
            count = collection.count()
            print(f"Number of documents in collection: {count}")

            if count > 0:
                # Get a sample of documents
                print("\nSample documents:")
                sample = collection.get(limit=3)
                for i, (doc, metadata) in enumerate(zip(sample['documents'], sample['metadatas']), 1):
                    print(f"\n{i}. Question: {doc}")
                    print(f"   Answer: {metadata['answer'][:100]}...")
            else:
                print("Collection is empty!")

        else:
            print(f"Collection '{CHROMA_COLLECTION_NAME}' not found!")
            print("You may need to run the ingestion script first.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Scan for all chroma_db directories under the workspace root
    import glob
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    print(
        "\n[SCAN] Searching for all 'chroma_db' directories under the workspace root...")
    for dirpath, dirnames, filenames in os.walk(workspace_root):
        if 'chroma_db' in dirnames:
            chroma_path = os.path.join(dirpath, 'chroma_db')
            print(f"\n[SCAN] Found: {chroma_path}")
            try:
                client = chromadb.Client(
                    Settings(persist_directory=chroma_path))
                collections = client.list_collections()
                print(f"  Collections: {[col.name for col in collections]}")
                for col in collections:
                    collection = client.get_collection(col.name)
                    print(
                        f"    Collection '{col.name}' has {collection.count()} documents.")
            except Exception as e:
                print(f"  [ERROR] Could not read collections: {e}")
    print("\n[SCAN] Done.")
    # Also run the original check
    check_chroma()
