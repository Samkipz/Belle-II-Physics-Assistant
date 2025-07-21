#!/usr/bin/env python3
"""
Generate Q&A Corpus from Chunks
==============================

This script reads belle2_processed_chunks.jsonl, generates a Q&A pair for each chunk using the configured LLM provider/model, and writes the results to belle2_qa_corpus.jsonl. Includes metadata (source, page, chunk_type).
"""

from llm_provider import call_llm, get_llm_config
from tqdm import tqdm
import json
import os
from dotenv import load_dotenv
load_dotenv()


INPUT_FILE = "belle2_processed_chunks.jsonl"
OUTPUT_FILE = "belle2_qa_corpus.jsonl"

PROMPT_TEMPLATE = (
    """
You are a physics assistant. Based on the following content, generate a question and a detailed answer suitable for a physics chatbot. If the content contains equations, output them in LaTeX (surrounded by $$). If it contains tables, format them in Markdown. If it references a figure, mention the caption.\n\nCONTENT:\n{content}\n\nReturn your response as JSON with keys 'question' and 'answer'.
"""
)


def main():
    cfg = get_llm_config()
    if not cfg["api_key"]:
        print("❌ LLM API key environment variable not set.")
        return
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return
    print(
        f"Using provider: {cfg['provider']}, model: {cfg['model']}, API: {cfg['api_url']}")
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Generating Q&A pairs"):
            try:
                chunk = json.loads(line)
                content = chunk.get("content", "")
                if not content.strip():
                    continue
                prompt = PROMPT_TEMPLATE.format(content=content.strip())
                llm_response = call_llm(
                    prompt, model=cfg["model"], provider=cfg["provider"])
                # Try to parse as JSON
                try:
                    qa = json.loads(llm_response)
                except Exception:
                    # Fallback: try to extract Q/A heuristically
                    qa = {"question": "", "answer": llm_response.strip()}
                qa_entry = {
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "source": chunk.get("document", "unknown"),
                    "page": chunk.get("page_number", -1),
                    "chunk_type": chunk.get("chunk_type", "text")
                }
                fout.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Error] Skipping chunk: {e}")
                continue
    print(f"✅ Q&A corpus written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
