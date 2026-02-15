
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import os

# --- CONFIGURATION ---
QDRANT_URL = "https://b4883482-d96f-4d99-a4f2-3b13b99268b9.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4uXSe-hl2I_oqdjZ3cb6u7mOTh2v9mfDP99cLyEx_vI"
COLLECTION_NAME = "medqa_textbooks"
OLLAMA_MODEL_NAME = "medqa-local"

# Initialize Clients (Global)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# ========== Core RAG Logic ==========
def rag_query(question: str):
    """Core RAG function used by both Gradio UI and API."""
    # 1. Retrieval
    query_vector = encoder.encode(question).tolist()
    
    hits = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3
    ).points
    
    context_list = [hit.payload['text'] for hit in hits]
    context_text = "\n\n".join(context_list)
    sources = list(set([hit.payload['source'] for hit in hits]))
    
    # 2. Augmentation & Generation
    full_prompt = f"""Context:
{context_text}

Question:
{question}
"""
    response = ollama.generate(
        model=OLLAMA_MODEL_NAME,
        prompt=full_prompt,
        stream=False
    )
    
    return response['response'], sources

# ========== Gradio UI ==========
def gradio_query(question):
    """Wrapper for Gradio interface."""
    if not question.strip():
        return "Please enter a question.", ""
    
    try:
        answer, sources = rag_query(question)
        sources_text = "\n".join([f"📚 {s}" for s in sources])
        return answer, sources_text
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

# Sample questions for easy testing
EXAMPLES = [
    ["What are the clinical manifestations of Type 1 Diabetes?"],
    ["What is the mechanism of action of aspirin?"],
    ["Answer the following multiple-choice question about medicine.\n\nQuestion:\nA 40-year-old zookeeper presents to the emergency department complaining of severe abdominal pain that radiates to her back, and nausea. The pain started 2 days ago and slowly increased until she could not tolerate it any longer. Past medical history is significant for hypertension and hypothyroidism. Additionally, she reports that she was recently stung by one of the zoo\u2019s smaller scorpions, but did not seek medical treatment. She takes aspirin, levothyroxine, oral contraceptive pills, and a multivitamin daily. Family history is noncontributory. An abdominal CT is consistent with acute pancreatitis. Which of the following is the most likely etiology for this patient\u2019s disease?\n\nOptions:\nA: Aspirin\nB: Oral contraceptive pills\nC: Scorpion sting\nD: Hypothyroidism\nE: Obesity"],
]

with gr.Blocks(
    title="MedQA RAG",
    theme=gr.themes.Soft(primary_hue="teal"),
    css="""
    .gradio-container { max-width: 900px !important; }
    .header { text-align: center; margin-bottom: 20px; }
    """
) as demo:
    
    gr.HTML("""
    <div class="header">
        <h1>🏥 MedQA RAG System</h1>
        <p>Ask medical questions powered by fine-tuned DeepSeek + Qdrant retrieval</p>
        <p><em>⏱️ Responses may take 1-3 minutes on free CPU</em></p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Your Medical Question",
                placeholder="Type your medical question here...",
                lines=5
            )
            submit_btn = gr.Button("🔍 Ask Question", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=3):
            answer_output = gr.Textbox(label="🤖 Answer", lines=10, interactive=False)
        with gr.Column(scale=1):
            sources_output = gr.Textbox(label="📚 Sources", lines=10, interactive=False)
    
    gr.Examples(
        examples=EXAMPLES,
        inputs=question_input,
        label="💡 Try these examples"
    )
    
    submit_btn.click(
        fn=gradio_query,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

# ========== FastAPI API (still available at /api/) ==========
app = FastAPI(title="MedQA RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.get("/api/health")
def read_root():
    return {"status": "healthy", "model": OLLAMA_MODEL_NAME}

@app.post("/api/query")
def api_query_rag(request: QueryRequest):
    try:
        answer, sources = rag_query(request.question)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount Gradio on the FastAPI app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
