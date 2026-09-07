FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-mlm.txt .
RUN pip install --no-cache-dir -r requirements-mlm.txt flask

# Copy application code
COPY caveman_compress_mlm.py .
COPY utils.py .
COPY server.py .

# Download spaCy models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download fr_core_news_sm

# Expose port
EXPOSE 3000

# Run server
CMD ["python", "server.py"]
