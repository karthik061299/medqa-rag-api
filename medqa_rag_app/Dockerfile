
# Use Python base image
FROM python:3.11-slim

# Install system dependencies (curl for installing ollama)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make the start script executable
RUN chmod +x start.sh

# Expose the port HF Spaces uses
EXPOSE 7860

# Run the start script
CMD ["./start.sh"]
