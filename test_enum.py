#!/usr/bin/env python3
"""
Test ModelType Enum
==================

This script tests if the ModelType enum is working correctly.
"""

from advanced_rag_system_pinecone import ModelType


def test_enum():
    """Test the ModelType enum"""
    print("🔍 Testing ModelType Enum...")

    try:
        # Test enum values
        print(f"MISTRAL_7B: {ModelType.MISTRAL_7B}")
        print(f"MISTRAL_7B.value: {ModelType.MISTRAL_7B.value}")
        print(f"MISTRAL_7B type: {type(ModelType.MISTRAL_7B)}")

        # Test enum comparison
        test_model = ModelType.MISTRAL_7B
        print(f"test_model: {test_model}")
        print(f"test_model.value: {test_model.value}")
        print(
            f"test_model == ModelType.MISTRAL_7B: {test_model == ModelType.MISTRAL_7B}")

        # Test string conversion
        print(f"str(test_model): {str(test_model)}")
        print(f"repr(test_model): {repr(test_model)}")

        # Test dictionary key
        test_dict = {ModelType.MISTRAL_7B: "test"}
        print(f"test_dict: {test_dict}")
        print(
            f"ModelType.MISTRAL_7B in test_dict: {ModelType.MISTRAL_7B in test_dict}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Testing ModelType Enum")
    print("=" * 30)

    success = test_enum()

    if success:
        print("\n✅ Enum test passed!")
    else:
        print("\n❌ Enum test failed!")
