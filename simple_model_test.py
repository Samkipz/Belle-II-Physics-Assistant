#!/usr/bin/env python3
"""
Simple Model Test
================

This script tests if the model is actually being called and generating responses.
"""

import os
import requests
import json


def test_model_directly():
    """Test the model directly without RAG"""
    print("🔍 Testing Model Directly...")

    # Check environment variables
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        print("❌ Error: HF_API_KEY environment variable not set")
        return False

    try:
        # Test with a simple prompt
        prompt = "Please explain what integrated luminosity means in particle physics experiments. Write in your own words."

        headers = {
            "Authorization": f"Bearer {hf_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 500
        }

        print(f"📝 Testing prompt: {prompt}")
        print(f"🤖 Model: mistralai/Mistral-7B-Instruct-v0.3")

        response = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers=headers,
            json=payload
        )

        print(f"📡 Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                model_response = result["choices"][0]["message"]["content"]
                print(f"\n✅ Model Response:")
                print(f"{model_response}")
                print(f"\n📊 Response length: {len(model_response)} characters")
                return True
            else:
                print(f"❌ Unexpected response format: {result}")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Simple Model Test")
    print("=" * 30)

    success = test_model_directly()

    if success:
        print("\n✅ Model test passed!")
    else:
        print("\n❌ Model test failed!")
