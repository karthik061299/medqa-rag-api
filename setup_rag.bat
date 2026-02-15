
@echo off
echo ===================================================
echo Setting up RAG with Ollama (Bypassing compiler issues)
echo ===================================================

echo 1. Installing python dependencies...
pip install qdrant-client sentence-transformers ollama

echo 2. Building Vector Database (Qdrant)...
python medqa_rag_app/build_vector_db.py

echo 3. Creating Local Model in Ollama...
echo (Make sure Ollama app is running in the background!)
ollama create medqa-local -f medqa_rag_app/Modelfile

echo 4. Running RAG Inference Test...
python medqa_rag_app/rag_inference.py

pause
 