"""Test universal LLM client with multiple providers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_universal_client():
    """Test Universal LLM Client with auto-fallback."""
    print("\n" + "="*70)
    print("Testing Universal LLM Client")
    print("="*70)
    
    try:
        from src.llm_engine import UniversalLLMClient
        
        # Test with auto provider selection
        print("\n1. Initializing Universal Client (auto provider)...")
        client = UniversalLLMClient(provider="auto", enable_fallback=True)
        
        print(f"✓ Client initialized")
        print(f"  Primary provider: {client.provider}")
        print(f"  Model: {client.model_name}")
        print(f"  Fallback providers: {len(client.fallback_clients)}")
        
        # Test 1: Simple generation
        print("\n2. Testing simple text generation...")
        prompt = "Explain artificial intelligence in one sentence."
        response = client.generate(prompt, max_tokens=100)
        
        if response:
            print(f"✓ Generation successful")
            print(f"  Prompt: {prompt}")
            print(f"  Response: {response[:150]}...")
        else:
            print(f"✗ Generation failed")
            return False
        
        # Test 2: Generate with metadata
        print("\n3. Testing generation with metadata...")
        result = client.generate_with_metadata(prompt, max_tokens=100)
        
        if result.get('text'):
            print(f"✓ Metadata generation successful")
            print(f"  Provider: {result.get('provider', 'unknown')}")
            print(f"  Model: {result.get('model', 'unknown')}")
            print(f"  Prompt tokens: {result.get('prompt_tokens', 0)}")
            print(f"  Response tokens: {result.get('response_tokens', 0)}")
            print(f"  Used fallback: {result.get('used_fallback', False)}")
        else:
            print(f"✗ Metadata generation failed")
        
        # Test 3: Document classification
        print("\n4. Testing document classification...")
        from src.llm_engine import PromptTemplates
        
        doc = "Quarterly earnings report shows 25% revenue growth."
        categories = ["Financial", "Technical", "Marketing", "Legal"]
        
        class_prompt = PromptTemplates.document_classification(doc, categories)
        class_result = client.generate(class_prompt, max_tokens=50)
        
        if class_result:
            print(f"✓ Classification successful")
            print(f"  Document: {doc}")
            print(f"  Result: {class_result[:100]}...")
        else:
            print(f"✗ Classification failed")
        
        return True
        
    except Exception as e:
        print(f"✗ Universal client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_specific_providers():
    """Test individual providers."""
    print("\n" + "="*70)
    print("Testing Individual Providers")
    print("="*70)
    
    results = []
    
    # Test Gemini
    print("\n1. Testing Gemini Client...")
    try:
        from src.llm_engine import GeminiClientV2
        gemini = GeminiClientV2()
        response = gemini.generate("Say hello in one word", max_tokens=10)
        
        if response:
            print(f"✓ Gemini works: {response}")
            results.append(("Gemini", True))
        else:
            print(f"✗ Gemini: No response")
            results.append(("Gemini", False))
    except Exception as e:
        print(f"✗ Gemini failed: {e}")
        results.append(("Gemini", False))
    
    # Test Hugging Face
    print("\n2. Testing Hugging Face Client...")
    try:
        from src.llm_engine import HuggingFaceClient
        hf = HuggingFaceClient()
        response = hf.generate("Say hello in one word", max_tokens=10)
        
        if response:
            print(f"✓ Hugging Face works: {response}")
            results.append(("Hugging Face", True))
        else:
            print(f"✗ Hugging Face: No response")
            results.append(("Hugging Face", False))
    except Exception as e:
        print(f"✗ Hugging Face failed: {e}")
        results.append(("Hugging Face", False))
    
    return results


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Universal LLM Client - Comprehensive Test Suite")
    print("="*70)
    
    # Test 1: Universal client
    universal_ok = test_universal_client()
    
    # Test 2: Individual providers
    provider_results = test_specific_providers()
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    print(f"\n{'Test':<30} {'Status':<10}")
    print("-" * 40)
    print(f"{'Universal Client':<30} {'PASS' if universal_ok else 'FAIL':<10}")
    
    for provider, result in provider_results:
        print(f"{provider + ' Client':<30} {'PASS' if result else 'FAIL':<10}")
    
    passed = sum([universal_ok] + [r for _, r in provider_results])
    total = 1 + len(provider_results)
    
    print()
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print()
    
    if passed > 0:
        print("✓ At least one provider is working!")
        print("  Your LLM system is operational.")
    else:
        print("✗ All providers failed.")
        print("  Please check API keys and internet connection.")
    
    print()
    return passed > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
