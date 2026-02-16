import os
import requests
import json
from dotenv import load_dotenv
from utils.bge_m3_embedding import BGEM3Embeddings

load_dotenv()

def test_vector_search():
    q_url = os.getenv("QDRANT_URL", "https://qdrant.api.blaiq.ai").rstrip("/")
    collection = os.getenv("QDRANT_COLLECTION")
    api_key = os.getenv("QDRANT_API_KEY")
    
    print(f"--- Qdrant Config ---")
    print(f"URL: {q_url}")
    print(f"Collection: {collection}")
    print(f"API Key present: {'Yes' if api_key else 'No'}")
    
    # 1. Test Embedding
    print(f"\n--- Testing Embedding ---")
    embedder = BGEM3Embeddings()
    query = "Wärmepumpe SolvisLea Pro"
    embedding = embedder.embed_query(query)
    print(f"Embedding length: {len(embedding)}")
    if all(v == 0.0 for v in embedding):
        print("❌ ERROR: Embedding is all zeros!")
        return

    # 2. Test Qdrant Search
    print(f"\n--- Testing Qdrant Search ---")
    search_url = f"{q_url}/collections/{collection}/points/search"
    headers = {"api-key": api_key} if api_key else {}
    payload = {
        "vector": embedding,
        "limit": 5,
        "with_payload": True,
        "score_threshold": 0.0
    }
    
    try:
        resp = requests.post(search_url, json=payload, headers=headers, verify=False, timeout=10.0)
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            results = resp.json().get("result", [])
            print(f"Results found: {len(results)}")
            for i, r in enumerate(results):
                print(f"  [{i}] ID: {r.get('id')}, Score: {r.get('score')}")
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_vector_search()
