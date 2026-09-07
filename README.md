# Caveman Compression - Forked

Forked from [wilpel/caveman-compression](https://github.com/wilpel/caveman-compression) with additions for OpenCode integration.

## Additions

- **CamemBERT support** for French MLM compression
- **MCP server** for OpenCode integration
- **HTTP API server** for remote compression
- **Docker support** for Unraid deployment
- **Auto language detection** (en, fr, es, de, etc.)

## Models

| Language | Model | Type | Notes |
|----------|-------|------|-------|
| English | RoBERTa | MLM | roberta-base |
| French | CamemBERT | MLM | camembert-base |
| Other | spaCy | NLP | Language-specific models |

## Installation

### Local (Laptop)

```bash
# Clone the repo
git clone https://github.com/BaconDroid/caveman-compression.git
cd caveman-compression

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-mlm.txt flask

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm

# Test
./venv/bin/python caveman_compress_mlm.py compress "Test text"
```

### Docker (Unraid)

```bash
# Build and run
docker-compose up -d

# Or build manually
docker build -t caveman-compression .
docker run -p 3000:3000 caveman-compression
```

## Usage

### CLI

```bash
# Compress English text (MLM)
./venv/bin/python caveman_compress_mlm.py compress "Your text here"

# Compress French text (MLM)
./venv/bin/python caveman_compress_mlm.py compress -l fr "Votre texte ici"

# Compress with NLP (lighter, less aggressive)
./venv/bin/python caveman_compress_nlp.py compress "Your text here"
./venv/bin/python caveman_compress_nlp.py compress -l fr "Votre texte ici"
```

### HTTP API

```bash
# Start server
./venv/bin/python server.py

# Compress text
curl -X POST http://localhost:3000/compress \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "language": "en"}'
```

### MCP (OpenCode)

The MCP server is configured in `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "caveman": {
      "type": "local",
      "command": [
        "/var/home/bacon/Projects/caveman-compression/venv/bin/python",
        "/var/home/bacon/Projects/caveman-compression/mcp_server.py"
      ],
      "enabled": true
    }
  }
}
```

## Adding More Languages

To add support for another language:

1. Add the model config to `SUPPORTED_LANGUAGES` in `caveman_compress_mlm.py`:
   ```python
   SUPPORTED_LANGUAGES = {
       "en": {"model": "roberta-base", "spacy": "en_core_web_sm"},
       "fr": {"model": "camembert-base", "spacy": "fr_core_news_sm"},
       "de": {"model": "bert-base-german-cased", "spacy": "de_core_news_sm"},
   }
   ```

2. Add the model imports in `caveman_compress_mlm.py`:
   ```python
   from transformers import BertForMaskedLM, BertTokenizer
   ```

3. Update the `get_mlm_model` function to handle the new model type.

4. Download the spaCy model:
   ```bash
   python -m spacy download de_core_news_sm
   ```

## License

MIT (same as original)
