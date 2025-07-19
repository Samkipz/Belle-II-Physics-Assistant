import json
import requests
import os

# Test the improved prompt with the exact data from JSONL


def test_prompt_only():
    print("Testing Improved Prompt Engineering")
    print("=" * 50)

    # Load the exact data from JSONL
    with open("belle2_qa_detailed_full.jsonl", 'r', encoding='utf-8') as f:
        qa_pairs = [json.loads(line) for line in f]

    # Find the luminosity question
    luminosity_qa = None
    for qa in qa_pairs:
        if "integrated luminosity" in qa['question'].lower():
            luminosity_qa = qa
            break

    if not luminosity_qa:
        print("Could not find luminosity Q&A")
        return

    print(f"Testing with question: {luminosity_qa['question']}")
    print(f"Expected answer should mention: Bhabha scattering and digamma production")

    # Create the improved prompt
    context = f"Source 1:\nQ: {luminosity_qa['question']}\nA: {luminosity_qa['answer']}"

    improved_prompt = f"""You are a Belle II physics assistant. Answer the user's question using ONLY the information provided in the context below. Do not use any external knowledge or make up information.

If the context doesn't contain enough information to answer the question completely, say so clearly.

Context:
{context}

User Question: How was integrated luminosity measured during Belle II Phase 2?

Instructions:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information in my knowledge base to answer this question completely."
3. Do not add information that is not present in the context
4. Be accurate and precise

Answer:"""

    print(f"\nPrompt length: {len(improved_prompt)} characters")

    # Test with LLM
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        print("No API key found")
        return

    try:
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        payload = {
            "messages": [{"role": "user", "content": improved_prompt}],
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 1000
        }

        print("\nSending request to LLM...")
        response = requests.post(
            "https://router.huggingface.co/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"]
            print(f"\nLLM Answer:\n{answer}")

            # Check for key terms
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
                print("Expected: Bhabha scattering and digamma production")

        else:
            print(f"LLM Error: {result}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_prompt_only()
