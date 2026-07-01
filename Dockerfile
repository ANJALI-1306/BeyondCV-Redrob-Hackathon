FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create embeddings cache directory
RUN mkdir -p embeddings_cache

# Default command
CMD ["python", "rank.py", "--candidates", "candidates.jsonl", "--out", "submission.csv"]
