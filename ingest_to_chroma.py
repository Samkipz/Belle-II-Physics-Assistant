import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os

# File containing Q&A pairs
QA_FILE = "belle2_qa_detailed_full.jsonl"
CHROMA_COLLECTION_NAME = "belle2_qa"
CHROMA_DB_DIR = os.path.abspath("./chroma_db")  # Use absolute path

# Load Q&A pairs from JSONL file


def load_qa_pairs(filename):
    qa_pairs = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            qa_pairs.append(item)
    return qa_pairs

# Initialize Chroma client and collection


def get_chroma_collection(collection_name):
    print(f"Initializing ChromaDB with directory: {CHROMA_DB_DIR}")
    client = chromadb.Client(Settings(persist_directory=CHROMA_DB_DIR))
    collection = client.get_or_create_collection(collection_name)
    print(f"Collection '{collection_name}' created/retrieved successfully")
    return collection


def main():
    # Load Q&A pairs
    qa_pairs = load_qa_pairs(QA_FILE)
    print(f"Loaded {len(qa_pairs)} Q&A pairs.")

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')

    # Prepare data for Chroma
    questions = [item['question'] for item in qa_pairs]
    answers = [item['answer'] for item in qa_pairs]
    metadatas = [{"answer": ans} for ans in answers]
    ids = [f"qa_{i}" for i in range(len(qa_pairs))]

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(
        questions, show_progress_bar=True, convert_to_numpy=True)

    # Store in Chroma
    collection = get_chroma_collection(CHROMA_COLLECTION_NAME)
    print("Adding documents to Chroma collection...")
    collection.add(
        embeddings=embeddings,
        documents=questions,
        metadatas=metadatas,
        ids=ids
    )
    print(
        f"Added {len(qa_pairs)} Q&A pairs to Chroma collection '{CHROMA_COLLECTION_NAME}'.")

    # Verify the data was added
    count = collection.count()
    print(f"Collection now contains {count} documents")

    # Check if directory was created
    if os.path.exists(CHROMA_DB_DIR):
        print(f"ChromaDB directory created at: {CHROMA_DB_DIR}")
    else:
        print(f"Warning: ChromaDB directory not found at: {CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()
