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

# Download spaCy models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download fr_core_news_sm

# Download MLM models (RoBERTa and CamemBERT)
RUN python -c "from transformers import RobertaForMaskedLM, RobertaTokenizer; RobertaTokenizer.from_pretrained('roberta-base'); RobertaForMaskedLM.from_pretrained('roberta-base')"
RUN python -c "from transformers import CamembertForMaskedLM, CamembertTokenizer; CamembertTokenizer.from_pretrained('camembert-base'); CamembertForMaskedLM.from_pretrained('camembert-base')"

# Expose port
EXPOSE 3000

# Run server
CMD ["python", "server.py"]
