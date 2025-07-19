import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import requests

# Configurations
QA_FILE = "belle2_qa_detailed_full.jsonl"
CHROMA_COLLECTION_NAME = "belle2_qa"
CHROMA_DB_DIR = os.path.abspath("./chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
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
        "temperature": 0.1,
        "max_tokens": 1000
    }
    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    else:
        return f"[Error from model API]: {result}"


def test_complete_rag():
    print("Testing Complete RAG Pipeline")
    print("=" * 60)

    # Step 1: Load and ingest data
    print("Step 1: Loading and ingesting data...")
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

    # Step 2: Test retrieval and LLM
    print("\nStep 2: Testing retrieval and LLM...")
    test_query = "How was integrated luminosity measured during Belle II Phase 2?"

    print(f"\nTest Query: {test_query}")
    print("-" * 50)

    try:
        # Retrieve relevant documents
        retrieved = retrieve_top_k(
            test_query, collection, embed_model, k=TOP_K)
        print(f"Retrieved {len(retrieved['documents'][0])} documents:")

        for j, (doc, metadata) in enumerate(zip(retrieved['documents'][0], retrieved['metadatas'][0]), 1):
            print(f"\n{j}. Question: {doc}")
            print(f"   Answer: {metadata['answer'][:150]}...")

        # Test LLM integration
        if HUGGINGFACE_API_KEY:
            print(f"\nStep 3: Testing LLM integration...")
            prompt = compose_prompt_improved(test_query, retrieved)
            print(f"Generated prompt length: {len(prompt)} characters")

            try:
                answer = query_huggingface(prompt)
                print(f"\nLLM Answer: {answer}")

                # Check if answer matches expected content
                has_bhabha = "bhabha" in answer.lower()
                has_digamma = "digamma" in answer.lower()
                has_qed = "qed" in answer.lower()

                print(f"\nAnalysis:")
                print(f"✅ Mentions Bhabha scattering: {has_bhabha}")
                print(f"✅ Mentions digamma production: {has_digamma}")
                print(f"✅ Mentions QED processes: {has_qed}")

                if has_bhabha and has_digamma:
                    print("\n🎉 SUCCESS: Answer correctly uses the source data!")
                else:
                    print("\n❌ ISSUE: Answer doesn't match the source data")

            except Exception as e:
                print(f"LLM Error: {e}")
        else:
            print("No API key available for LLM test")

    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Complete RAG pipeline test finished!")


if __name__ == "__main__":
    test_complete_rag()
