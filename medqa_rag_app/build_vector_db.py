
import json
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
QDRANT_URL = "https://b4883482-d96f-4d99-a4f2-3b13b99268b9.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4uXSe-hl2I_oqdjZ3cb6u7mOTh2v9mfDP99cLyEx_vI"
COLLECTION_NAME = "medqa_textbooks"

# Get absolute path to the data file relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "textbook_chunks.json")

def build_index():
    print("🚀 Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60) # Increased timeout

    print("🧠 Loading Embedding Model...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2') # Creates 384-dim vectors

    # 1. Recreate Collection (Schema)
    # WARNING: This deletes the existing collection!
    print(f"📦 Creating collection '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,  # Dimension for all-MiniLM-L6-v2
            distance=models.Distance.COSINE
        )
    )

    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found!")
        print(f"   Looking in: {os.path.dirname(DATA_PATH)}")
        return

    print("📖 Loading chunks from JSON...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # 3. Embed and Upload in Batches
    # LOWERED BATCH SIZE TO PREVENT TIMEOUTS
    batch_size = 64 
    total_docs = len(chunks)
    print(f"📥 uploading {total_docs} chunks in batches of {batch_size}...")

    # Prepare batches
    for i in range(0, total_docs, batch_size):
        batch = chunks[i : i + batch_size]
        
        # Extract Texts & Generate Embeddings
        texts = [doc['text'] for doc in batch]
        embeddings = encoder.encode(texts).tolist()
        
        # Prepare Payloads
        payloads = [
            {"source": doc['source'], "text": doc['text'], "chunk_index": doc['chunk_index']} 
            for doc in batch
        ]
        
        # Upload
        try:
            client.upload_collection(
                collection_name=COLLECTION_NAME,
                vectors=embeddings,
                payload=payloads,
                ids=None, # Auto-generate IDs
                batch_size=batch_size
            )
            print(f"   Processed {min(i + batch_size, total_docs)}/{total_docs}", end="\r")
        except Exception as e:
            print(f"\n❌ Error uploading batch {i}: {e}")

    print("\n✅ Indexing complete! Data is now in Qdrant Cloud.")

if __name__ == "__main__":
    build_index()
