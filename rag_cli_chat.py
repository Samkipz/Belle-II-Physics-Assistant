import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import os

# Configurations
CHROMA_COLLECTION_NAME = "belle2_qa"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# Use Hugging Face Router API (OpenAI-compatible)
HUGGINGFACE_API_URL = "https://router.huggingface.co/v1/chat/completions"
HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")
TOP_K = 3

# Initialize Chroma and embedding model


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


def compose_prompt(user_query, retrieved):
    context = "\n".join([
        f"Q: {q}\nA: {a['answer']}" for q, a in zip(retrieved['documents'][0], retrieved['metadatas'][0])
    ])
    prompt = f"You are a helpful assistant for Belle II physics. Use the following context to answer the user's question.\n\n{context}\n\nUser question: {user_query}\nAnswer:"
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
        "stream": False
    }
    response = requests.post(
        HUGGINGFACE_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    # Extract the content from the OpenAI-compatible response
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    else:
        return f"[Error from model API]: {result}"


def main():
    print("Welcome to the Belle II RAG CLI Chatbot! Type 'exit' to quit.")
    collection = get_chroma_collection()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    while True:
        user_query = input("\nYour question: ").strip()
        if user_query.lower() in ("exit", "quit"):
            break
        retrieved = retrieve_top_k(
            user_query, collection, embed_model, k=TOP_K)
        prompt = compose_prompt(user_query, retrieved)
        print("\n[Retrieving answer from LLM...]")
        try:
            answer = query_huggingface(prompt)
        except Exception as e:
            answer = f"[Error querying LLM API]: {e}"
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
