#!/usr/bin/env python3
"""
Caveman Compression HTTP Server
Runs on Unraid, provides MLM-based text compression via HTTP API.
Uses RoBERTa for English, CamemBERT for French, NLP for other languages.
"""

from flask import Flask, request, jsonify
import sys
import os

# Add the caveman-compression directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caveman_compress_mlm import compress_text, detect_language
from caveman_compress_nlp import compress_text as compress_text_nlp

app = Flask(__name__)

# Compression presets: name -> prob_threshold
# Higher threshold = less aggressive (keeps more words)
# Lower threshold = more aggressive (removes more words)
COMPRESSION_PRESETS = {
    "light": 1e-2,      # Light compression, keeps most words
    "medium": 1e-3,     # Medium compression
    "aggressive": 1e-5, # Aggressive compression (old default)
}

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
    preset = data.get('preset', 'light')  # light, medium, aggressive
    threshold = data.get('threshold')  # Override preset if specified
    
    # Get threshold from preset or use custom value
    if threshold is not None:
        # Custom threshold (must be between 1e-7 and 1e-1)
        threshold = max(1e-7, min(1e-1, float(threshold)))
    else:
        # Use preset
        threshold = COMPRESSION_PRESETS.get(preset, COMPRESSION_PRESETS["light"])
    
    try:
        # Auto-detect language if not specified
        if language is None:
            language = detect_language(text)
        
        if method == 'nlp':
            # Force NLP mode
            compressed = compress_text_nlp(text, lang=language)
            model = f"caveman-nlp-{language}"
        elif method == 'mlm':
            # Force MLM mode
            compressed = compress_text(text, language=language, prob_threshold=threshold)
            model = f"caveman-mlm-{language}"
        else:
            # Auto mode: MLM for en/fr/de/zh/pt/tr/it, NLP for others
            if language in ["en", "fr", "de", "zh", "pt", "tr", "it"]:
                compressed = compress_text(text, language=language, prob_threshold=threshold)
                model = f"caveman-mlm-{language}"
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
