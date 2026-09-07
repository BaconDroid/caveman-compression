#!/usr/bin/env python3
"""
Caveman Compression HTTP Server
Runs on Unraid, provides MLM-based text compression via HTTP API.
<<<<<<< HEAD
Uses RoBERTa for English, CamemBERT for French, NLP for other languages.
=======
Uses RoBERTa for English, spaCy for other languages (NLP fallback).
>>>>>>> main
"""

from flask import Flask, request, jsonify
import sys
import os

# Add the caveman-compression directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

<<<<<<< HEAD
from caveman_compress_mlm import compress_text, detect_language
=======
from caveman_compress_mlm import compress_text
>>>>>>> main
from caveman_compress_nlp import compress_text as compress_text_nlp

app = Flask(__name__)

# Cache models on startup
print("Loading models...", file=sys.stderr)
try:
<<<<<<< HEAD
    from caveman_compress_mlm import get_mlm_model
    get_mlm_model("en")
    get_mlm_model("fr")
    print("MLM models loaded (en, fr).", file=sys.stderr)
=======
    from caveman_compress_mlm import get_models
    get_models()
    print("MLM models loaded.", file=sys.stderr)
>>>>>>> main
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
<<<<<<< HEAD
    language = data.get('language')
    method = data.get('method', 'auto')  # auto, mlm, nlp
    
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
            compressed = compress_text(text, language=language)
            model = f"caveman-mlm-{language}"
        else:
            # Auto mode: MLM for en/fr/de/zh/pt/tr/it, NLP for others
            if language in ["en", "fr", "de", "zh", "pt", "tr", "it"]:
                compressed = compress_text(text, language=language)
                model = f"caveman-mlm-{language}"
            else:
                compressed = compress_text_nlp(text, lang=language)
                model = f"caveman-nlp-{language}"
=======
    language = data.get('language', 'en')
    method = data.get('method', 'auto')  # auto, mlm, nlp
    prob_threshold = data.get('prob_threshold', 1e-5)
    
    try:
        if method == 'mlm' or (method == 'auto' and language == 'en'):
            # Use MLM (RoBERTa) for English
            compressed = compress_text(text, prob_threshold=prob_threshold)
            model = 'caveman-mlm-roberta'
        else:
            # Use NLP for other languages
            compressed = compress_text_nlp(text, language=language)
            model = 'caveman-nlp'
>>>>>>> main
        
        return jsonify({
            'compressed': compressed,
            'language': language,
            'model': model,
<<<<<<< HEAD
            'original_size': len(text),
            'compressed_size': len(compressed),
            'compression_ratio': len(compressed) / len(text) if text else 0
=======
            'compressionRatio': len(compressed) / len(text) if text else 0
>>>>>>> main
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
