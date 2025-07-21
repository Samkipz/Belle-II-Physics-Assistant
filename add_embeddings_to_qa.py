from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
qa_pairs = []
with open("belle2_qa_corpus.jsonl", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        qa = json.loads(line)
        question = qa.get("question") or qa.get(
            "answer") or qa.get("content") or ""
        qa["embedding"] = model.encode(question).tolist()
        qa_pairs.append(qa)
        if idx % 10 == 0:
            print(f"Processed {idx} entries...")

with open("belle2_qa_corpus_embedded.jsonl", "w", encoding="utf-8") as f:
    for qa in qa_pairs:
        f.write(json.dumps(qa, ensure_ascii=False) + "\n")
print(
    f"✅ Embeddings added to belle2_qa_corpus_embedded.jsonl ({len(qa_pairs)} entries)")
