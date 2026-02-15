
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import os

# --- CONFIGURATION ---
QDRANT_URL = "https://b4883482-d96f-4d99-a4f2-3b13b99268b9.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4uXSe-hl2I_oqdjZ3cb6u7mOTh2v9mfDP99cLyEx_vI"
COLLECTION_NAME = "medqa_textbooks"
OLLAMA_MODEL_NAME = "medqa-local"

def get_rag_response(query):
    # 1. Initialize DB Clients
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Search Qdrant
    print(f"🔎 Searching Qdrant for: {query}")
    query_vector = encoder.encode(query).tolist()
    
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3 
    )
    
    # 3. Build Context
    context_list = [hit.payload['text'] for hit in hits]
    context_text = "\n\n".join(context_list)
    print(f"📄 Found {len(context_list)} relevant chunks.")

    # 4. Construct Prompt
    # We pass the full formatted prompt to Ollama, or let the Modelfile handle the template.
    # Since we defined a template in Modelfile, we just pass the input.
    
    full_prompt = f"""Context:
{context_text}

Question:
{query}
"""

    print("💬 Generating answer with Ollama...")
    
    try:
        response = ollama.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=full_prompt,
            stream=False
        )
        return response['response']
    except Exception as e:
        return f"❌ Error invoking Ollama: {e}\nMake sure Ollama is running!"

if __name__ == "__main__":
    q = """Answer the following multiple-choice question about medicine.

Question:
A 40-year-old zookeeper presents to the emergency department complaining of severe abdominal pain that radiates to her back, and nausea. The pain started 2 days ago and slowly increased until she could not tolerate it any longer. Past medical history is significant for hypertension and hypothyroidism. Additionally, she reports that she was recently stung by one of the zoo’s smaller scorpions, but did not seek medical treatment. She takes aspirin, levothyroxine, oral contraceptive pills, and a multivitamin daily. Family history is noncontributory. Today, her blood pressure is 108/58 mm Hg, heart rate is 99/min, respiratory rate is 21/min, and temperature is 37.0°C (98.6°F). On physical exam, she is a well-developed, obese female that looks unwell. Her heart has a regular rate and rhythm. Radial pulses are weak but symmetric. Her lungs are clear to auscultation bilaterally. Her lateral left ankle is swollen, erythematous, and painful to palpate. An abdominal CT is consistent with acute pancreatitis. Which of the following is the most likely etiology for this patient’s disease?

Options:
A: Aspirin
B: Oral contraceptive pills
C: Scorpion sting
D: Hypothyroidism
E: Obesity"""

    answer = get_rag_response(q)
    print("\n" + "="*50)
    print(f"🤖 ANSWER:\n{answer}")
    print("="*50)
