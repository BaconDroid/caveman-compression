# opencode-caveman-compress

MLM-based text compression for LLM context optimization. Fork of [wilpel/caveman-compression](https://github.com/wilpel/caveman-compression) with additional features.

**Why use many token when few token do trick?**

## Features

- **MLM compression** using masked language models (RoBERTa for English, CamemBERT for French)
- **NLP fallback** for languages without MLM models (spaCy-based)
- **Auto language detection** via fastText (170+ languages)
- **Dynamic presets** calibrated from NLP compression ratio
- **Sentence or text mode** for different compression contexts
- **Custom MLM models** via environment variable
- **HTTP API** and **MCP server** for integration
- **Docker support** for easy deployment

## Quick Start

### Docker

```bash
# Build (models are provisioned during the build)
docker build -t caveman-compression .

# Run (models are baked in; the runtime stays offline)
docker run -p 3000:3000 caveman-compression

# Customize the languages baked into the image
docker build --build-arg LANGUAGES=en,fr,de -t caveman-compression .

# Or with docker-compose
docker-compose up -d
```

### API Usage

```bash
# Compress text (default: preset=lite, mode=sentence)
curl -X POST http://localhost:3000/compress \
  -H "Content-Type: application/json" \
  -d '{"text": "The reason your React component is re-rendering is likely because you are creating a new object reference on each render cycle."}'

# Response
{
  "compressed": "reason your React is re-rendering is likely creating a object reference on each render cycle.",
  "language": "en",
  "model": "caveman-mlm-en",
  "preset": "lite",
  "mode": "sentence",
  "original_size": 127,
  "compressed_size": 93,
  "compression_ratio": 0.73
}
```

## Presets

Presets are dynamically calibrated from NLP compression ratio. The NLP compressor removes stop words, determiners, and auxiliaries. MLM presets are calculated relative to NLP:

| Preset | Formula | Effect |
|--------|---------|--------|
| `lite` | = NLP | Similar to NLP compression (~30% removed) |
| `full` | = NLP × 1.5 | More aggressive (~45% removed) |
| `ultra` | = NLP × 2 | Maximum compression (~60% removed) |

**Default:** `lite` (fast, simple, effective)

## Modes

| Mode | NLP Context | MLM Context | Performance | Use Case |
|------|-------------|-------------|-------------|----------|
| `sentence` | Per sentence | Word in sentence | Fast | Default, preserves structure |
| `text` | Full text | Word in text | Slower | Long texts, global context |

**Default:** `sentence`

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to compress |
| `language` | string | auto | Language code (en, fr, de, etc.) |
| `preset` | string | `lite` | Compression preset: `lite`, `full`, `ultra` |
| `mode` | string | `sentence` | Compression mode: `sentence`, `text` |
| `method` | string | `mlm` | Compression method: `mlm`, `nlp` |

### Method

- `mlm` (default): Uses MLM when available (en, fr, de, zh, pt, it), NLP fallback for others
- `nlp`: Force NLP mode (spaCy-based, no MLM)

## Examples

### French

```bash
curl -X POST http://localhost:3000/compress \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour, je travaille sur un projet complexe qui nécessite une attention particulière aux détails et une compréhension approfondie des exigences du client.", "preset": "lite"}'

# Response
{
  "compressed": "Bonjour, je travaille sur un projet complexe nécessite particulière détails et compréhension approfondie exigences du client.",
  "language": "fr",
  "model": "caveman-mlm-fr",
  "preset": "lite",
  "mode": "sentence",
  "original_size": 155,
  "compressed_size": 126,
  "compression_ratio": 0.81
}
```

### English

```bash
curl -X POST http://localhost:3000/compress \
  -H "Content-Type: application/json" \
  -d '{"text": "The reason your React component is re-rendering is likely because you are creating a new object reference on each render cycle.", "preset": "full"}'

# Response
{
  "compressed": "reason React is re-rendering is likely creating a reference on cycle.",
  "language": "en",
  "model": "caveman-mlm-en",
  "preset": "full",
  "mode": "sentence",
  "original_size": 127,
  "compressed_size": 69,
  "compression_ratio": 0.54
}
```

## Custom Models

Add custom MLM models via the `CUSTOM_MLM_MODELS` build argument (models are
provisioned during the build):

```bash
docker build \
  --build-arg LANGUAGES=en,fr,sv \
  --build-arg CUSTOM_MLM_MODELS='{"sv": {"model": "KB/bert-base-swedish-cased", "spacy": "sv_core_news_sm"}}' \
  -t caveman-compression .
```

Each custom entry must provide both a `model` and a `spacy` pipeline. Entries
missing either are skipped with a warning, and any language listed in
`LANGUAGES` without an available spaCy pipeline fails the build instead of
baking an image that cannot run.

To provision custom models at container startup instead, set
`DOWNLOAD_MODELS_ON_STARTUP=1` (the runtime is offline by default).

## MCP Server

The MCP server provides the same functionality via the Model Context Protocol:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "caveman_compress",
    "arguments": {
      "text": "Your text here",
      "preset": "lite",
      "mode": "sentence"
    }
  }
}
```

## Language Detection

Language detection uses fastText as the primary method, with a word-list
heuristic fallback (English vs French) when fastText is unavailable or the
model fails to load:

- **fastText**: Fast, accurate, 170+ languages
- **Word-list heuristic**: Last resort for unknown text (en/fr only)

## Performance

| Method | 100 words | 1000 words |
|--------|-----------|------------|
| NLP | ~100ms | ~1s |
| MLM (sentence mode) | ~10s | ~100s |
| MLM (text mode) | ~10s | ~100s |

MLM is ~100x slower than NLP due to neural network inference per word.

## Supported Languages

### MLM Models (default)

| Language | Model | spaCy |
|----------|-------|-------|
| English | roberta-base | en_core_web_sm |
| French | camembert-base | fr_core_news_sm |
| German | bert-base-german-cased | de_core_news_sm |
| Italian | dbmdz/bert-base-italian-cased | it_core_news_sm |
| Portuguese | neuralmind/bert-base-portuguese-cased | pt_core_news_sm |
| Chinese | bert-base-chinese | zh_core_web_sm |

Turkish is not a default language: spaCy ships no Turkish pipeline, and every
language needs a spaCy pipeline for tokenization and NER.

### NLP Models (fallback)

spaCy models for 15+ languages with multilingual fallback.

## Environment Variables

`LANGUAGES` and `CUSTOM_MLM_MODELS` are build arguments (see above). Runtime
variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | HTTP server port (honored by the gunicorn bind and the health check) |
| `DOWNLOAD_MODELS_ON_STARTUP` | `0` | Set to `1` to provision models at container startup (runtime is offline by default) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/compress` | POST | Compress text |

## License

Same as upstream [wilpel/caveman-compression](https://github.com/wilpel/caveman-compression).
