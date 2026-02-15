# 🏥 MedQA RAG System — End-to-End Medical Question Answering

A complete **Retrieval-Augmented Generation (RAG)** pipeline for answering medical multiple-choice questions. The system combines a **fine-tuned DeepSeek-R1-Distill-Llama-8B** language model with a **Qdrant** vector database containing 126,000+ medical textbook chunks, deployed as a web application on **Hugging Face Spaces**.

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Dataset & Source Information](#3-dataset--source-information)
4. [Project Structure](#4-project-structure)
5. [Phase 1 — Data Preparation](#5-phase-1--data-preparation)
6. [Phase 2 — Fine-Tuning DeepSeek](#6-phase-2--fine-tuning-deepseek)
7. [Phase 3 — Vector Database (Qdrant Cloud)](#7-phase-3--vector-database-qdrant-cloud)
8. [Phase 4 — RAG Application (Local)](#8-phase-4--rag-application-local)
9. [Phase 5 — Cloud Deployment (Hugging Face Spaces)](#9-phase-5--cloud-deployment-hugging-face-spaces)
10. [How to Test the Live API](#10-how-to-test-the-live-api)
11. [Technical Details & Configuration](#11-technical-details--configuration)
12. [Troubleshooting](#12-troubleshooting)
13. [Future Improvements](#13-future-improvements)

---

## 1. Project Overview

### Problem Statement
Medical licensing exams require deep reasoning over vast medical knowledge. General-purpose LLMs often lack domain-specific accuracy.

### Solution
I built a **RAG pipeline** that:
1. **Fine-tunes** a state-of-the-art LLM (DeepSeek-R1-Distill-Llama-8B) on 12,723 medical QA pairs
2. **Retrieves** relevant context from 18 medical textbooks (126,803 chunks) using semantic search
3. **Generates** grounded, context-aware answers by combining retrieval with the fine-tuned model
4. **Deploys** the entire system as a live web application on Hugging Face Spaces

### Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Base Model | DeepSeek-R1-Distill-Llama-8B | Medical reasoning LLM |
| Fine-Tuning | Unsloth + QLoRA (4-bit) | Efficient fine-tuning on Google Colab T4 GPU |
| Vector Database | Qdrant Cloud | Semantic search over textbook chunks |
| Embeddings | all-MiniLM-L6-v2 (384-dim) | Convert text to vectors for similarity search |
| Model Serving | Ollama (GGUF Q4_K_M) | Local/cloud inference with quantized model |
| API Framework | FastAPI + Gradio | REST API + Web UI |
| Deployment | Docker + Hugging Face Spaces | Cloud hosting |
| Platform | Google Colab (T4 GPU) | Training environment |

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                   │
│    "What is the most likely etiology of acute pancreatitis           │
│     in a patient stung by a scorpion?"                              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GRADIO WEB UI / FastAPI                          │
│                 (Hugging Face Spaces)                                │
│                 https://karthik6129-medqa-rag.hf.space              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌──────────────────────┐    ┌─────────────────────────────┐
│   1. RETRIEVAL       │    │   2. GENERATION              │
│                      │    │                              │
│  all-MiniLM-L6-v2    │    │   Ollama (medqa-local)       │
│  Encode query →      │    │   Fine-tuned DeepSeek-R1     │
│  384-dim vector      │    │   Distill-Llama-8B           │
│         │            │    │   (Q4_K_M quantized)         │
│         ▼            │    │         ▲                    │
│  Qdrant Cloud        │    │         │                    │
│  Top-3 similar ──────┼────┼─► Context + Query            │
│  textbook chunks     │    │   → Prompt → Answer          │
│                      │    │                              │
└──────────────────────┘    └─────────────────────────────┘
        │                               │
        │  Sources:                     │  Answer:
        │  Harrison's Internal Med      │  "The correct answer
        │  Robbins Pathology            │   is C. Scorpion sting"
        │  Schwartz Surgery             │
        └───────────────────────────────┘
```

---

## 3. Dataset & Source Information

### MedQA Dataset
- **Source**: [MedQA (US Medical Licensing Exam Questions)](https://github.com/jndf0/MedQA)
- **Files**: `data_clean/data_clean/questions/US/` containing `train.jsonl`, `dev.jsonl`, `test.jsonl`
- **Total QA Pairs**: **12,723** multiple-choice questions
- **Format**: Each record contains a question, 5 options (A–E), and the correct answer

### Medical Textbooks (for RAG Knowledge Base)
- **Source**: `data_clean/data_clean/textbooks/en/`
- **Total Textbooks**: **18 books** covering all major medical disciplines:

| # | Textbook | Discipline |
|---|----------|-----------|
| 1 | Harrison's Internal Medicine | Internal Medicine |
| 2 | Robbins Pathology | Pathology |
| 3 | Schwartz Surgery | Surgery |
| 4 | Gray's Anatomy | Anatomy |
| 5 | Katzung Pharmacology | Pharmacology |
| 6 | Janeway Immunology | Immunology |
| 7 | Levy Physiology | Physiology |
| 8 | Lippincott Biochemistry | Biochemistry |
| 9 | Ross Histology | Histology |
| 10 | Alberts Cell Biology | Cell Biology |
| 11 | Nelson Pediatrics | Pediatrics |
| 12 | Novak Gynecology | Gynecology |
| 13 | Williams Obstetrics | Obstetrics |
| 14 | Adams Neurology | Neurology |
| 15 | DSM-5 Psychiatry | Psychiatry |
| 16 | Pathoma (Husain) | Pathology (Supplement) |
| 17 | First Aid Step 1 | USMLE Review |
| 18 | First Aid Step 2 | USMLE Review |

- **Total Chunks**: **126,803** (after splitting with chunk_size=1000, overlap=200)

---

## 4. Project Structure

```
data_clean/
│
├── README.md                         # ← This file
│
├── Data_Preparation.ipynb            # Phase 1: Data processing notebook (Google Colab)
├── Fine_Tuning_DeepSeek.ipynb        # Phase 2: Model fine-tuning notebook (Google Colab)
├── medqa_finetune_data.jsonl         # Output: 12,723 formatted QA pairs for training
│
├── data_clean/                       # Raw dataset
│   ├── questions/US/                 # MedQA question files (train/dev/test.jsonl)
│   └── textbooks/en/                 # 18 medical textbooks (.txt files)
│
├── setup_rag.bat                     # One-click local setup script (Windows)
│
└── medqa_rag_app/                    # Phase 3-5: RAG Application
    ├── data/
    │   └── textbook_chunks.json      # 126,803 textbook chunks for vector DB
    ├── models/
    │   └── deepseek-r1-distill-llama-8b.Q4_K_M.gguf  # Fine-tuned model (4.6 GB)
    │
    ├── build_vector_db.py            # Script to upload chunks to Qdrant Cloud
    ├── rag_inference.py              # Standalone RAG inference test script
    ├── app.py                        # FastAPI + Gradio web application
    ├── Modelfile                     # Ollama model configuration (system prompt + template)
    ├── start.py                      # Cloud startup script (starts Ollama → pulls model → starts API)
    ├── Dockerfile                    # Docker configuration for Hugging Face Spaces
    └── requirements.txt              # Python dependencies
```

---

## 5. Phase 1 — Data Preparation

> **Notebook**: `Data_Preparation.ipynb`
> **Platform**: Google Colab (no GPU required)

### What It Does
1. **Reads** MedQA question files (`train.jsonl`, `dev.jsonl`, `test.jsonl`) from Google Drive
2. **Formats** each QA pair into an instruction-tuning format (Alpaca style)
3. **Chunks** 18 medical textbooks into 126,803 passages for the vector database
4. **Downloads** the processed files locally

### Step-by-Step Instructions

1. **Upload the `data_clean` folder** to your Google Drive (under `My Drive/data_clean/`)
2. **Open** `Data_Preparation.ipynb` in Google Colab
3. **Run all cells** sequentially:
   - Cell 1: Install dependencies (`pandas`, `pyarrow`, `langchain`, `tqdm`)
   - Cell 2: Mount Google Drive
   - Cell 3: Locate the `questions/US/` and `textbooks/en/` directories
   - Cell 4: Process QA files → Outputs `medqa_finetune_data.jsonl` (12,723 records)
   - Cell 5: Chunk textbooks → Outputs `textbook_chunks.json` (126,803 chunks)
   - Cell 6: Download both files to local machine

### Output Files

| File | Size | Description |
|------|------|-------------|
| `medqa_finetune_data.jsonl` | ~13 MB | 12,723 QA pairs in Alpaca instruction format |
| `textbook_chunks.json` | ~112 MB | 126,803 textbook chunks with source metadata |

### Sample Data Point (QA)
```json
{
  "instruction": "Answer the following multiple-choice question about medicine.\n\nQuestion:\nA 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination...\n\nOptions:\nA: Ampicillin\nB: Ceftriaxone\nC: Ciprofloxacin\nD: Doxycycline\nE: Nitrofurantoin",
  "input": "",
  "output": "The correct answer is E. Nitrofurantoin"
}
```

### Sample Textbook Chunk
```json
{
  "id": "InternalMed_Harrison_42",
  "text": "Acute pancreatitis is a common clinical condition... Scorpion stings, particularly from Tityus species, are a well-recognized cause of acute pancreatitis...",
  "source": "InternalMed_Harrison",
  "chunk_index": 42
}
```

### Chunking Strategy
- **Method**: `RecursiveCharacterTextSplitter` from LangChain
- **Chunk Size**: 1,000 characters
- **Overlap**: 200 characters (to preserve context at boundaries)
- **Separators**: `\n\n`, `\n`, `. `, ` `, `` (in order of priority)

---

## 6. Phase 2 — Fine-Tuning DeepSeek

> **Notebook**: `Fine_Tuning_DeepSeek.ipynb`
> **Platform**: Google Colab with **T4 GPU** (free tier)

### What It Does
1. **Loads** the base model `unsloth/DeepSeek-R1-Distill-Llama-8B` with 4-bit quantization
2. **Applies** LoRA adapters for efficient fine-tuning (only 0.52% of parameters trained)
3. **Trains** on 12,723 MedQA pairs using the Alpaca prompt format
4. **Exports** the fine-tuned model in **GGUF Q4_K_M** format (~4.6 GB) for use with Ollama

### Step-by-Step Instructions

1. **Upload** `medqa_finetune_data.jsonl` to Google Drive (e.g., `My Drive/Text_Book_Chunk/`)
2. **Open** `Fine_Tuning_DeepSeek.ipynb` in Google Colab
3. **Set Runtime** → Change runtime type → **T4 GPU**
4. **Run all cells** sequentially:
   - Cell 1: Install Unsloth (`pip install unsloth`)
   - Cell 2: Load base model with 4-bit quantization
   - Cell 3: Apply LoRA adapters
   - Cell 4: Load and format dataset (Alpaca template)
   - Cell 5: Train the model (30 steps, ~4 minutes)
   - Cell 6: Test inference on a sample question
   - Cell 7: Export to GGUF Q4_K_M format
   - Cell 8: Download the `.gguf` file

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `unsloth/DeepSeek-R1-Distill-Llama-8B` |
| Quantization | 4-bit (QLoRA) |
| Max Sequence Length | 2,048 tokens |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 16 |
| LoRA Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable Parameters | 41,943,040 / 8,072,204,288 (**0.52%**) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 1 per device |
| Gradient Accumulation Steps | 8 |
| Effective Batch Size | 8 |
| Learning Rate | 2e-4 |
| LR Scheduler | Linear |
| Warmup Steps | 5 |
| Max Steps | 30 |
| Optimizer | AdamW 8-bit |
| Weight Decay | 0.01 |
| Precision | FP16 (T4 GPU) |

### Training Loss Curve

The model shows a clear learning trajectory over 30 steps:

| Step | Loss | Step | Loss | Step | Loss |
|------|------|------|------|------|------|
| 1 | 2.3676 | 11 | 1.4087 | 21 | 1.2690 |
| 2 | 2.3181 | 12 | 1.3605 | 22 | 1.3206 |
| 3 | 2.5121 | 13 | 1.4690 | 23 | 1.2753 |
| 4 | 2.3195 | 14 | 1.2856 | 24 | 1.2798 |
| 5 | 2.1235 | 15 | 1.2613 | 25 | 1.3808 |
| 6 | 2.0409 | 16 | 1.2387 | 26 | 1.2154 |
| 7 | 1.7604 | 17 | 1.2641 | 27 | **1.1861** |
| 8 | 1.7588 | 18 | 1.2576 | 28 | 1.2476 |
| 9 | 1.6494 | 19 | 1.3227 | 29 | 1.2390 |
| 10 | 1.3839 | 20 | 1.4267 | 30 | 1.4123 |

- **Starting Loss**: 2.37 → **Final Loss**: 1.19 (best at step 27)
- **Training Time**: ~4 minutes on T4 GPU

### Test Inference Result

**Question**: A 40-year-old zookeeper with acute pancreatitis after a scorpion sting...

**Model Output**: `The correct answer is C. Scorpion sting` ✅

### Output

| File | Size | Description |
|------|------|-------------|
| `deepseek-r1-distill-llama-8b.Q4_K_M.gguf` | ~4.6 GB | Quantized fine-tuned model |

The GGUF file was also uploaded to **Hugging Face Hub**: [`karthik6129/medqa-finetuned-gguf`](https://huggingface.co/karthik6129/medqa-finetuned-gguf)

---

## 7. Phase 3 — Vector Database (Qdrant Cloud)

### What It Does
Uploads 126,803 textbook chunks to **Qdrant Cloud** as a searchable vector database for the Retrieval step.

### Prerequisites
- A free Qdrant Cloud account at [cloud.qdrant.io](https://cloud.qdrant.io)
- The `textbook_chunks.json` file from Phase 1

### Setup Steps

1. **Create a Qdrant Cloud cluster** (free tier available)
2. **Get your credentials**:
   - **URL**: `https://<your-cluster-id>.aws.cloud.qdrant.io:6333`
   - **API Key**: From the Qdrant Cloud dashboard
3. **Place** `textbook_chunks.json` in `medqa_rag_app/data/`
4. **Update** the credentials in `build_vector_db.py` (lines 9-10):
   ```python
   QDRANT_URL = "https://<your-cluster-id>.aws.cloud.qdrant.io:6333"
   QDRANT_API_KEY = "<your-api-key>"
   ```
5. **Run** the indexing script:
   ```bash
   python medqa_rag_app/build_vector_db.py
   ```

### Script: `build_vector_db.py`

This script:
1. Connects to Qdrant Cloud
2. Creates (or recreates) a collection called `medqa_textbooks`
3. Loads the `textbook_chunks.json` file
4. Encodes each chunk using `all-MiniLM-L6-v2` (384 dimensions)
5. Uploads vectors + payloads in batches of 64
6. Stores metadata: `source` (textbook name), `text` (content), `chunk_index`

### Collection Configuration

| Setting | Value |
|---------|-------|
| Collection Name | `medqa_textbooks` |
| Vector Size | 384 dimensions |
| Distance Metric | Cosine Similarity |
| Total Vectors | 126,803 |
| Embedding Model | `all-MiniLM-L6-v2` |

### Estimated Time
- Indexing takes approximately **30-60 minutes** (depending on your network speed)

---

## 8. Phase 4 — RAG Application (Local)

### What It Does
Runs the complete RAG pipeline on your local machine using **Ollama** for model inference.

### Prerequisites
- **Python 3.9+**
- **Ollama** installed: [Download from ollama.com](https://ollama.com)
- The `.gguf` model file in `medqa_rag_app/models/`

### Local Setup (Windows)

#### Option A: Automated Setup
```batch
cd data_clean
.\setup_rag.bat
```

This script will:
1. Install Python dependencies (`qdrant-client`, `sentence-transformers`, `ollama`)
2. Build the vector database (upload chunks to Qdrant)
3. Create the local Ollama model (`medqa-local`)
4. Run a test inference

#### Option B: Manual Setup

**Step 1**: Install dependencies
```bash
pip install qdrant-client sentence-transformers ollama fastapi uvicorn gradio
```

**Step 2**: Start Ollama (keep running in background)
```bash
ollama serve
```

**Step 3**: Create the custom model
```bash
cd medqa_rag_app
ollama create medqa-local -f Modelfile
```

**Step 4**: Test the RAG pipeline
```bash
python rag_inference.py
```

**Step 5**: Start the web application
```bash
python app.py
```
Open `http://localhost:7860` in your browser.

### Modelfile Configuration

The `Modelfile` defines the custom Ollama model:
```
FROM hf.co/karthik6129/medqa-finetuned-gguf:Q4_K_M

TEMPLATE """Below is an instruction that describes a task...
### Instruction:
{{ .System }}
{{ .Prompt }}

### Response:
"""

SYSTEM """Answer the question based strictly on the provided context. If the answer is not in the context, say "I don't know"."""
```

### RAG Pipeline Flow

```
User Question
     │
     ▼
┌────────────────────┐
│ Encode with         │
│ all-MiniLM-L6-v2    │  → 384-dim vector
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Qdrant Cloud        │
│ Cosine similarity   │  → Top 3 relevant chunks
│ search (limit=3)    │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Construct Prompt    │
│                     │
│ Context:            │
│ [chunk1]            │
│ [chunk2]            │
│ [chunk3]            │
│                     │
│ Question:           │
│ [user question]     │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Ollama              │
│ medqa-local model   │  → Generated Answer
│ (DeepSeek 8B GGUF)  │
└────────────────────┘
```

---

## 9. Phase 5 — Cloud Deployment (Hugging Face Spaces)

### What It Does
Deploys the RAG application as a Docker-based Hugging Face Space with a **Gradio web UI**.

### 🌐 Live URL: [https://karthik6129-medqa-rag.hf.space](https://karthik6129-medqa-rag.hf.space)

### Deployment Steps

1. **Create a new Space** on [huggingface.co/spaces](https://huggingface.co/spaces)
   - SDK: **Docker**
   - Hardware: Free CPU (Basic)
2. **Upload these files** to the Space root (drag and drop into "Files" tab):

   | File | Purpose |
   |------|---------|
   | `Dockerfile` | Builds the container (Python 3.11 + Ollama) |
   | `requirements.txt` | Python dependencies |
   | `app.py` | FastAPI + Gradio application |
   | `start.py` | Startup orchestrator (Ollama → Model → API) |
   | `Modelfile` | Ollama model configuration |

3. **Commit** the files — the Space will auto-build

### Dockerfile Explained

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl zstd dos2unix wget && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "start.py"]
```

### Startup Sequence (`start.py`)

1. **Starts** Ollama server in a background thread
2. **Waits** for Ollama to be ready (up to 60 seconds)
3. **Pulls** the model from Hugging Face Hub: `hf.co/karthik6129/medqa-finetuned-gguf:Q4_K_M`
4. **Creates** the custom model `medqa-local` with the Modelfile
5. **Launches** the Uvicorn server on port 7860

### Web UI Features

The Gradio interface provides:
- 📝 Text input box for medical questions
- 🔍 "Ask Question" button
- ⏳ Loading spinner while the model generates (1-3 minutes on free CPU)
- 🤖 Answer display area
- 📚 Sources panel showing which textbooks were retrieved
- 💡 Pre-loaded example questions for quick testing

---

## 10. How to Test the Live API

### Method 1: Web UI (Recommended)
1. Go to [https://karthik6129-medqa-rag.hf.space](https://karthik6129-medqa-rag.hf.space)
2. Type or select an example question
3. Click **"🔍 Ask Question"**
4. Wait 1-3 minutes for the response

### Method 2: PowerShell (Terminal)
```powershell
$body = '{"question": "What is the mechanism of action of aspirin?"}'
Invoke-RestMethod -Uri "https://karthik6129-medqa-rag.hf.space/api/query" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
```

### Method 3: Python Script
```python
import requests

response = requests.post(
    "https://karthik6129-medqa-rag.hf.space/api/query",
    json={"question": "What are the clinical features of Type 1 Diabetes?"},
    timeout=300  # 5 minute timeout for free CPU
)
print(response.json())
```

### Method 4: cURL (Linux/Mac)
```bash
curl -X POST "https://karthik6129-medqa-rag.hf.space/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is diabetes?"}' \
  --max-time 300
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Gradio Web UI |
| `/api/health` | GET | Health check (returns `{"status": "healthy"}`) |
| `/api/query` | POST | Submit a question (JSON body: `{"question": "..."}`) |

### Sample API Response
```json
{
  "answer": "The correct answer is C. Scorpion sting. Scorpion stings, particularly from Tityus species, are a well-recognized cause of acute pancreatitis...",
  "sources": ["InternalMed_Harrison", "Surgery_Schwartz"]
}
```

> **⚠️ Note**: The Swagger UI at `/docs` may show "Failed to fetch" because the browser times out after ~60 seconds. The model takes 1-3 minutes on the free CPU tier. Use the Gradio UI or terminal commands instead.

---

## 11. Technical Details & Configuration

### Embedding Model
| Property | Value |
|----------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Max Sequence Length | 256 tokens |
| Speed | ~14,000 sentences/sec on GPU |

### Fine-Tuned LLM
| Property | Value |
|----------|-------|
| Base Model | DeepSeek-R1-Distill-Llama-8B |
| Parameters | 8.07 billion |
| Quantization | Q4_K_M (4-bit, ~4.6 GB) |
| Context Length | 2,048 tokens |
| Fine-Tuning Method | QLoRA (LoRA rank=16) |
| Training Data | 12,723 MedQA pairs |
| Training Steps | 30 |
| HF Hub | `karthik6129/medqa-finetuned-gguf` |

### Qdrant Vector Database
| Property | Value |
|----------|-------|
| Provider | Qdrant Cloud (Free Tier) |
| Collection | `medqa_textbooks` |
| Vectors | 126,803 |
| Vector Size | 384 |
| Distance | Cosine |
| Retrieval Limit | Top 3 |

---

## 12. Troubleshooting

### Issue: "Failed to fetch" in Swagger UI
- **Cause**: Browser timeout (~60 seconds). The model takes 1-3 minutes on free CPU.
- **Solution**: Use the Gradio UI at the root URL, or use terminal/Python with a longer timeout.

### Issue: Ollama model not found
- **Cause**: Ollama server not running, or model not created.
- **Solution**:
  ```bash
  ollama serve          # Start Ollama server
  ollama create medqa-local -f Modelfile  # Create the model
  ollama list           # Verify model exists
  ```

### Issue: Qdrant connection error
- **Cause**: Invalid API key or cluster URL.
- **Solution**: Verify credentials in `build_vector_db.py` and `app.py`. Ensure your Qdrant Cloud cluster is active.

### Issue: Docker build fails on Hugging Face (missing zstd)
- **Cause**: Ollama installer requires `zstd` for extraction.
- **Solution**: Already fixed in the Dockerfile (`apt-get install -y curl zstd`).

### Issue: "exec format error" on HF Spaces
- **Cause**: Windows line endings (\r\n) in shell scripts.
- **Solution**: Use `start.py` (Python) instead of `start.sh` (bash). Already configured.

### Issue: Memory issues during fine-tuning
- **Cause**: T4 GPU has limited VRAM (16 GB).
- **Solution**: Use `load_in_4bit=True`, `per_device_train_batch_size=1`, and `gradient_accumulation_steps=8`.

---

## 13. Future Improvements

1. **More Training Steps**: Increase from 30 to a full epoch (1,590 steps) for better accuracy
2. **Evaluation (RAGAs)**: Implement automated evaluation using RAGAs framework (Faithfulness, Relevance, Context Recall)
3. **GPU Deployment**: Use HF Spaces with GPU hardware for faster inference (<10 seconds vs 2 minutes)
4. **Streaming Responses**: Add token-by-token streaming for better user experience
5. **Larger Context Window**: Retrieve 5-10 chunks instead of 3 for more comprehensive answers
6. **Hybrid Search**: Combine vector search with BM25 keyword search for better retrieval
7. **Chat History**: Add multi-turn conversation support
8. **Model Comparison**: Benchmark against base DeepSeek (no fine-tuning) to measure improvement

---

## 📝 Summary

| Phase | What | Where | Output |
|-------|------|-------|--------|
| 1. Data Preparation | Process QA + Chunk Textbooks | Google Colab | `medqa_finetune_data.jsonl`, `textbook_chunks.json` |
| 2. Fine-Tuning | Train DeepSeek with QLoRA | Google Colab (T4) | `deepseek-r1-distill-llama-8b.Q4_K_M.gguf` |
| 3. Vector DB | Upload chunks to Qdrant | Local Python | Qdrant Cloud collection (126,803 vectors) |
| 4. RAG App | Build & test locally | Local (Ollama + Python) | Working RAG pipeline |
| 5. Deployment | Deploy to HF Spaces | Docker + Hugging Face | [Live URL](https://karthik6129-medqa-rag.hf.space) |

---

*Built by Karthikeyan Iyappan • February 2026*
