
#!/usr/bin/env bash

# 1. Start Ollama in the background
ollama serve &

# 2. Wait for Ollama to wake up
echo "Waiting for Ollama..."
sleep 5

# 3. Create the model from your GGUF
echo "Creating model medqa-local..."
ollama create medqa-local -f Modelfile

# 4. Start FastAPI
echo "Starting FastAPI..."
uvicorn app:app --host 0.0.0.0 --port 7860
