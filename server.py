#!/usr/bin/env python3
"""
Caveman Compression HTTP Server
Runs on Unraid, provides MLM-based text compression via HTTP API.
Uses fastText for language detection, MLM for en/fr, NLP fallback for others.
"""

from flask import Flask, request, jsonify
import sys
import os

# Add the caveman-compression directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caveman_compress_mlm import compress_text, detect_language, SUPPORTED_LANGUAGES
from caveman_compress_nlp import compress_text as compress_text_nlp

app = Flask(__name__)

# Compression presets: name -> prob_threshold
COMPRESSION_PRESETS = {
    "lite": 0.95,       # Keep articles, full sentences (~30% dropped)
    "full": 0.5,        # Drop articles, filler, fragments OK (default)
    "ultra": 0.01,      # Maximum compression
}

# Languages with MLM models available (derived from SUPPORTED_LANGUAGES)
# These are models that could be loaded; actual loading happens lazily
MLM_LANGUAGES = set(SUPPORTED_LANGUAGES.keys())

# Cache models on startup
print("Loading models...", file=sys.stderr)
try:
    from caveman_compress_mlm import get_mlm_model
    get_mlm_model("en")
    get_mlm_model("fr")
    print("MLM models loaded (en, fr).", file=sys.stderr)
except Exception as e:
    print(f"Warning: MLM models failed to load: {e}", file=sys.stderr)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'caveman-compression'})

@app.route('/compress', methods=['POST'])
def compress():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'text is required'}), 400
    
    text = data['text']
    language = data.get('language')
    method = data.get('method', 'auto')  # auto, mlm, nlp
    preset = data.get('preset', 'full')  # lite, full, ultra
    threshold = data.get('threshold')  # Override preset if specified
    
    # Get threshold from preset or use custom value
    if threshold is not None:
        # Custom threshold (must be between 0 and 1)
        threshold = max(0.0, min(1.0, float(threshold)))
    else:
        # Use preset
        threshold = COMPRESSION_PRESETS.get(preset, COMPRESSION_PRESETS["full"])
    
    try:
        # Auto-detect language if not specified
        if language is None:
            language = detect_language(text)
        
        if method == 'nlp':
            # Force NLP mode
            compressed = compress_text_nlp(text, lang=language)
            model = f"caveman-nlp-{language}"
        elif method == 'mlm':
            # Force MLM mode with fallback to NLP
            try:
                compressed = compress_text(text, language=language, prob_threshold=threshold)
                model = f"caveman-mlm-{language}"
            except Exception:
                compressed = compress_text_nlp(text, lang=language)
                model = f"caveman-nlp-{language}"
        else:
            # Auto mode: try MLM first for supported langs, fallback to NLP
            if language in MLM_LANGUAGES:
                try:
                    compressed = compress_text(text, language=language, prob_threshold=threshold)
                    model = f"caveman-mlm-{language}"
                except Exception:
                    compressed = compress_text_nlp(text, lang=language)
                    model = f"caveman-nlp-{language}"
            else:
                compressed = compress_text_nlp(text, lang=language)
                model = f"caveman-nlp-{language}"
        
        return jsonify({
            'compressed': compressed,
            'language': language,
            'model': model,
            'preset': preset,
            'threshold': threshold,
            'original_size': len(text),
            'compressed_size': len(compressed),
            'compression_ratio': len(compressed) / len(text) if text else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
