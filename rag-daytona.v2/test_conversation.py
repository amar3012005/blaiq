
import requests
import json
import time

BASE_URL = "http://localhost:5100"
API_URL = f"{BASE_URL}/api/v1/query"

def test_query(query, history=None, context=None):
    payload = {
        "query": query,
        "history_context": history or [],
        "context": context or {}
    }
    
    print(f"\n--- Query: '{query}' ---")
    start_time = time.time()
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        latency = (time.time() - start_time) * 1000
        response.raise_for_status()
        data = response.json()
        
        print(f"Latency (TTFT approx): {latency:.2f}ms")
        print(f"Response: {data.get('answer', 'NO ANSWER')}")
        print(f"Sources: {data.get('sources', [])}")
        print(f"Metadata: {data.get('metadata', {})}")
        return data.get('answer')
    except Exception as e:
        print(f"Error: {e}")
        try:
            if hasattr(e, 'response') and e.response is not None:
                print(f"Error Details: {e.response.text}")
        except:
            pass
        return None

def run_tests():
    print(f"Testing RAG Service at {BASE_URL}")
    
    # 1. Installation (Zone A pattern)
    ans1 = test_query("How do I install Daytona?")
    
    # 2. German (Language Rule)
    ans2 = test_query("Was kostet Daytona?")
    
    # 3. Context/History (Zone C)
    history = [
        {"role": "user", "content": "How do I install Daytona?"},
        {"role": "assistant", "content": ans1 if ans1 else "Install via curl..."}
    ]
    # Simple follow-up
    ans3 = test_query("Does it work on Mac?", history=history)
    
    # 4. Ambiguous (Pattern detection)
    ans4 = test_query("What do you mean?")

if __name__ == "__main__":
    # Wait for service to be fully ready if we just restarted
    # time.sleep(2) 
    run_tests()
