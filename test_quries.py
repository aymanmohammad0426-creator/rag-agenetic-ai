import requests
import json
import time

BASE_URL = "http://localhost:8000"

test_queries = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",  # Out-of-domain test
]

print("🧪 Testing Agentic AI RAG Chatbot (Google Gemini)")
print("=" * 70)

for i, query in enumerate(test_queries, 1):
    print(f"\n📌 Test {i}/{len(test_queries)}")
    print(f"❓ Query: {query}\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"query": query},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Answer:")
            print(f"   {result['answer'][:300]}..." if len(result['answer']) > 300 else f"   {result['answer']}")
            print(f"\n📊 Confidence Score: {result['confidence_score']}")
            print(f"📚 Context Chunks Retrieved: {len(result['retrieved_chunks'])}")
            
            if result['retrieved_chunks']:
                print(f"\n📖 First Context Chunk:")
                print(f"   {result['retrieved_chunks'][0][:200]}...")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
    
    except requests.exceptions.Timeout:
        print("❌ Request timeout (API might be slow)")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("-" * 70)
    time.sleep(1)  # Rate limiting

print("\n✅ All tests completed!")