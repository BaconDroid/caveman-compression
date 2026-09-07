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
COPY caveman_compress_nlp.py .
COPY utils.py .
COPY server.py .
COPY mcp_server.py .
COPY download_models.py .

# Default languages (can be overridden at runtime)
ENV LANGUAGES="en,fr"

# Download models on build (for default languages)
RUN python download_models.py

# Expose port
EXPOSE 3000

# Entry point: download models for custom languages, then start server
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "server.py"]
