
# Use Python base image
FROM python:3.11-slim

# Install system dependencies (curl for installing ollama, dos2unix for line endings, wget for model download)
RUN apt-get update && apt-get install -y curl zstd dos2unix wget && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port HF Spaces uses
EXPOSE 7860

# Run the python start script (avoids Windows line ending issues in bash)
CMD ["python", "start.py"]
