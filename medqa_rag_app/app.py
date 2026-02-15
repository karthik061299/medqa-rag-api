
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import uvicorn
import os

# --- CONFIGURATION (Same as before) ---
QDRANT_URL = "https://b4883482-d96f-4d99-a4f2-3b13b99268b9.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4uXSe-hl2I_oqdjZ3cb6u7mOTh2v9mfDP99cLyEx_vI"
COLLECTION_NAME = "medqa_textbooks"
OLLAMA_MODEL_NAME = "medqa-local"

# Initialize App
app = FastAPI(title="MedQA RAG API", version="1.0")

# Initialize Clients (Global)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
# We load the encoder once at startup to save time per request
encoder = SentenceTransformer('all-MiniLM-L6-v2') 

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "healthy", "model": OLLAMA_MODEL_NAME}

@app.post("/query")
def query_rag(request: QueryRequest):
    query = request.question
    
    try:
        # 1. Retrieval
        print(f"🔎 Processing query: {query}")
        query_vector = encoder.encode(query).tolist()
        
        hits = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3 
        )
        
        context_list = [hit.payload['text'] for hit in hits]
        context_text = "\n\n".join(context_list)
        
        
        # 2. Augmentation & Generation
        full_prompt = f"""Context:
{context_text}

Question:
{query}
"""
        # Call Ollama
        response = ollama.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=full_prompt,
            stream=False
        )
        
        return {
            "answer": response['response'],
            "sources": [hit.payload['source'] for hit in hits]
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        # Return generic error for now, ideally shouldn't expose internal details
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
