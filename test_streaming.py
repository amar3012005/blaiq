
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "your_secure_api_key_here")
URL = "http://localhost:8020/query/graphrag/stream"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
PAYLOAD = {
    "query": "Who are you?",
    "collection_name": os.getenv("QDRANT_COLLECTION", "test"),
    "debug": True
}

def test_stream():
    print(f"Connecting to {URL}...")
    try:
        with requests.post(URL, json=PAYLOAD, headers=HEADERS, stream=True) as r:
            r.raise_for_status()
            print("Connected! Streaming events:")
            for line in r.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    print(decoded)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_stream()
