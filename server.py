#!/usr/bin/env python3
"""
Caveman Compression HTTP Server
Runs on Unraid, provides MLM-based text compression via HTTP API.
Uses RoBERTa for English, spaCy for other languages (NLP fallback).
"""

from flask import Flask, request, jsonify
import sys
import os

# Add the caveman-compression directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caveman_compress_mlm import compress_text
from caveman_compress_nlp import compress_text as compress_text_nlp

app = Flask(__name__)

# Cache models on startup
print("Loading models...", file=sys.stderr)
try:
    from caveman_compress_mlm import get_models
    get_models()
    print("MLM models loaded.", file=sys.stderr)
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
        
        return jsonify({
            'compressed': compressed,
            'language': language,
            'model': model,
            'compressionRatio': len(compressed) / len(text) if text else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
