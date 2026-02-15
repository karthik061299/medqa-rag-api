
import os
import time
import threading
import subprocess

def run_ollama():
    print("Include: Starting Ollama serve...")
    # Run ollama serve in the background
    subprocess.run(["ollama", "serve"], check=False)

def wait_for_ollama(timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Check if ollama is running by listing models
            result = subprocess.run(["ollama", "list"], capture_output=True, check=True)
            print("Ollama is ready!")
            return True
        except subprocess.CalledProcessError:
            time.sleep(1)
        except FileNotFoundError:
            print("Ollama binary not found yet...")
            time.sleep(1)
    return False

def main():
    # 1. Start Ollama in a separate thread
    print("Starting Ollama serve...")
    thread = threading.Thread(target=run_ollama, daemon=True)
    thread.start()
    

    # 2. Wait for Ollama to be ready
    if not wait_for_ollama(timeout=60):
        print("Error: Ollama failed to start within timeout.")
        # Proceed anyway? No, it will fail.
    

    # 3. Pull the model from Hugging Face (Ollama Native)
    # The user's model tag: hf.co/karthik6129/medqa-finetuned-gguf:Q4_K_M
    hf_model_tag = "hf.co/karthik6129/medqa-finetuned-gguf:Q4_K_M"
    
    print(f"Pulling model {hf_model_tag}...")
    pull_result = subprocess.run(["ollama", "pull", hf_model_tag], check=True)
    
    # 4. Create the custom model (with System Prompt)
    # We need to update Modelfile dynamically or just rely on the pulled model if Modelfile is updated.
    # Let's assume Modelfile is updated to FROM <hf_model_tag>
    
    print("Creating custom model medqa-local...")
    create_result = subprocess.run(["ollama", "create", "medqa-local", "-f", "Modelfile"])
    
    if create_result.returncode != 0:
        print("❌ Failed to create model 'medqa-local'. Check Modelfile.")
    else:
        print("✅ Model 'medqa-local' created successfully.")
    
    # 5. Start FastAPI
    print("Starting FastAPI app...")
    subprocess.run(["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"])

if __name__ == "__main__":
    main()
