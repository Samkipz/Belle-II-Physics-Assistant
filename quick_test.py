import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import requests

# Configurations
CHROMA_COLLECTION_NAME = "belle2_qa"
CHROMA_DB_DIR = os.path.abspath("./chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")


def test_improved_rag():
    print("Quick Test of Improved RAG System")
    print("=" * 50)

    # Load original data
    with open("belle2_qa_detailed_full.jsonl", 'r', encoding='utf-8') as f:
        qa_pairs = [json.loads(line) for line in f]
    print(f"Loaded {len(qa_pairs)} Q&A pairs from JSONL")

    # Find the exact question about luminosity
    luminosity_qa = None
    for qa in qa_pairs:
        if "integrated luminosity" in qa['question'].lower():
            luminosity_qa = qa
            break

    if luminosity_qa:
        print(f"\nFound luminosity Q&A:")
        print(f"Q: {luminosity_qa['question']}")
        print(f"A: {luminosity_qa['answer'][:200]}...")

        # Test the improved prompt with this exact question
        test_query = "How was integrated luminosity measured during Belle II Phase 2?"

        # Create a mock retrieval result with the exact matching document
        mock_retrieved = {
            'documents': [[luminosity_qa['question']]],
            'metadatas': [[{'answer': luminosity_qa['answer']}]]
        }

        # Test improved prompt
        context = f"Source 1:\nQ: {luminosity_qa['question']}\nA: {luminosity_qa['answer']}"

        improved_prompt = f"""You are a Belle II physics assistant. Answer the user's question using ONLY the information provided in the context below. Do not use any external knowledge or make up information.

If the context doesn't contain enough information to answer the question completely, say so clearly.

Context:
{context}

User Question: {test_query}

Instructions:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information in my knowledge base to answer this question completely."
3. Do not add information that is not present in the context
4. Be accurate and precise

Answer:"""

        print(f"\nImproved prompt length: {len(improved_prompt)} characters")
        print(f"Prompt preview: {improved_prompt[:300]}...")

        if HUGGINGFACE_API_KEY:
            print("\nTesting with LLM...")
            try:
                headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                           "Content-Type": "application/json"}
                payload = {
                    "messages": [{"role": "user", "content": improved_prompt}],
                    "model": "mistralai/Mistral-7B-Instruct-v0.3",
                    "stream": False,
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
                response = requests.post(
                    "https://router.huggingface.co/v1/chat/completions",
                    headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    answer = result["choices"][0]["message"]["content"]
                    print(f"\nLLM Answer: {answer}")

                    # Check if answer matches the source
                    if "Bhabha scattering" in answer and "digamma production" in answer:
                        print("\n✅ SUCCESS: Answer matches the source data!")
                    else:
                        print("\n❌ ISSUE: Answer doesn't match the source data")
                        print("Expected: Bhabha scattering and digamma production")
                        print(f"Got: {answer[:200]}...")
                else:
                    print(f"LLM Error: {result}")

            except Exception as e:
                print(f"LLM Error: {e}")
        else:
            print("No API key available for LLM test")
    else:
        print("Could not find luminosity Q&A in the data")


if __name__ == "__main__":
    test_improved_rag()
