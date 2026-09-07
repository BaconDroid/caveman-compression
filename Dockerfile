FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY caveman_compress_mlm.py .
COPY caveman_compress_nlp.py .
COPY server.py .
COPY mcp_server.py .
COPY download_models.py .

# Create models directory
RUN mkdir -p /app/models

# Download models during build
ARG LANGUAGES=en,fr
ENV LANGUAGES=${LANGUAGES}
RUN python download_models.py

# Expose port
EXPOSE 3000

# Run server
CMD ["python", "server.py"]
