import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import os
import json

# Configurations
QA_FILE = "belle2_qa_detailed_full.jsonl"
CHROMA_COLLECTION_NAME = "belle2_qa"
CHROMA_DB_DIR = os.path.abspath("./chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
HUGGINGFACE_API_URL = "https://router.huggingface.co/v1/chat/completions"
HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")
TOP_K = 3


def load_qa_pairs(filename):
    qa_pairs = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            qa_pairs.append(item)
    return qa_pairs


def get_chroma_collection():
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


def compose_prompt_improved(user_query, retrieved):
    # Create a more structured context
    context_parts = []
    for i, (doc, metadata) in enumerate(zip(retrieved['documents'][0], retrieved['metadatas'][0]), 1):
        context_parts.append(f"Source {i}:\nQ: {doc}\nA: {metadata['answer']}")

    context = "\n\n".join(context_parts)

    # Improved prompt with stronger instructions
    prompt = f"""You are a Belle II physics assistant. Answer the user's question using ONLY the information provided in the context below. Do not use any external knowledge or make up information.

If the context doesn't contain enough information to answer the question completely, say so clearly.

Context:
{context}

User Question: {user_query}

Instructions:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information in my knowledge base to answer this question completely."
3. Do not add information that is not present in the context
4. Be accurate and precise

Answer:"""

    return prompt


def query_huggingface(prompt):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
               "Content-Type": "application/json"}
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "stream": False,
        "temperature": 0.1,  # Lower temperature for more focused answers
        "max_tokens": 1000
    }
    response = requests.post(
        HUGGINGFACE_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    else:
        return f"[Error from model API]: {result}"


def initialize_rag_system():
    """Initialize the RAG system by loading data and creating embeddings"""
    print("Initializing RAG system...")

    # Load Q&A pairs
    qa_pairs = load_qa_pairs(QA_FILE)
    print(f"Loaded {len(qa_pairs)} Q&A pairs")

    # Initialize embedding model
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Prepare data
    questions = [item['question'] for item in qa_pairs]
    answers = [item['answer'] for item in qa_pairs]
    metadatas = [{"answer": ans} for ans in answers]
    ids = [f"qa_{i}" for i in range(len(qa_pairs))]

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = embed_model.encode(
        questions, show_progress_bar=True, convert_to_numpy=True)

    # Store in Chroma
    collection = get_chroma_collection()
    print("Adding documents to Chroma collection...")
    collection.add(
        embeddings=embeddings,
        documents=questions,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Added {len(qa_pairs)} Q&A pairs to Chroma collection")
    print(f"Collection count: {collection.count()}")

    return collection, embed_model


def main():
    print("Welcome to the Belle II RAG CLI Chatbot (Fixed Version)!")
    print("Type 'exit' to quit.")
    print("=" * 60)

    # Initialize the RAG system (load data and create embeddings)
    collection, embed_model = initialize_rag_system()

    while True:
        user_query = input("\nYour question: ").strip()
        if user_query.lower() in ("exit", "quit"):
            break

        print("\n[Retrieving relevant documents...]")
        retrieved = retrieve_top_k(
            user_query, collection, embed_model, k=TOP_K)

        print(
            f"\nRetrieved {len(retrieved['documents'][0])} relevant documents:")
        for i, (doc, metadata) in enumerate(zip(retrieved['documents'][0], retrieved['metadatas'][0]), 1):
            print(f"{i}. {doc[:100]}...")

        print("\n[Generating answer from LLM...]")
        try:
            prompt = compose_prompt_improved(user_query, retrieved)
            answer = query_huggingface(prompt)
            print(f"\nAnswer: {answer}")

            # Show sources option
            show_sources = input(
                "\nShow full retrieved sources? (y/n): ").strip().lower()
            if show_sources == 'y':
                print("\n" + "="*60)
                print("RETRIEVED SOURCES:")
                for i, (doc, metadata) in enumerate(zip(retrieved['documents'][0], retrieved['metadatas'][0]), 1):
                    print(f"\nSource {i}:")
                    print(f"Q: {doc}")
                    print(f"A: {metadata['answer']}")
                    print("-" * 40)

        except Exception as e:
            print(f"\n[Error]: {e}")


if __name__ == "__main__":
    main()
