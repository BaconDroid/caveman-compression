#!/usr/bin/env python3
"""
Caveman Compression HTTP Server
Runs on Unraid, provides MLM-based text compression via HTTP API.
Uses fastText for language detection; MLM for active languages (en/fr by
default), returns the input unchanged for all others.
"""

from flask import Flask, request, jsonify
import sys
import os

# Add the caveman-compression directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caveman_compress_mlm import compress_text, detect_language, SUPPORTED_LANGUAGES, InputTooLongError
from caveman_compress_nlp import compress_text as compress_text_nlp

app = Flask(__name__)

# Languages with MLM models available (derived from SUPPORTED_LANGUAGES)
MLM_LANGUAGES = set(SUPPORTED_LANGUAGES.keys())

# Input constraints
MAX_INPUT_LENGTH = 4096  # characters

# Transport-boundary limit (bytes). Generous headroom over the semantic
# MAX_INPUT_LENGTH so the character-level check below remains the precise
# contract; this only rejects oversized request bodies before they are read or
# parsed.
MAX_CONTENT_LENGTH = 64 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

PRESETS = ("lite", "full", "ultra")
MODES = ("sentence", "paragraph", "text")
METHODS = ("mlm", "nlp")


@app.errorhandler(413)
def request_too_large(e):
    """Return a stable JSON body for transport-boundary rejections."""
    return jsonify({'error': 'Request body too large'}), 413

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
    # get_json(silent=True) returns None for missing/malformed JSON instead of
    # raising, so we can produce a deterministic JSON 400 rather than Flask's
    # default HTML error page.
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid request body: expected a JSON object'}), 400

    text = data.get('text')
    if not isinstance(text, str):
        return jsonify({'error': "'text' must be a non-empty string"}), 400
    if not text.strip():
        return jsonify({'error': "'text' must not be empty"}), 400
    if len(text) > MAX_INPUT_LENGTH:
        return jsonify({'error': f"'text' exceeds maximum length of {MAX_INPUT_LENGTH} characters"}), 413

    language = data.get('language')
    if language is not None and not isinstance(language, str):
        return jsonify({'error': "'language' must be a string"}), 400

    preset = data.get('preset', 'lite')  # lite, full, ultra
    if not isinstance(preset, str) or preset not in PRESETS:
        return jsonify({'error': f"'preset' must be one of {list(PRESETS)}"}), 400

    mode = data.get('mode', 'sentence')  # sentence, text
    if not isinstance(mode, str) or mode not in MODES:
        return jsonify({'error': f"'mode' must be one of {list(MODES)}"}), 400

    method = data.get('method', 'mlm')  # "mlm" (MLM if available, NLP fallback) or "nlp" (force NLP)
    if not isinstance(method, str) or method not in METHODS:
        return jsonify({'error': f"'method' must be one of {list(METHODS)}"}), 400

    if method == 'nlp' and (preset != 'lite' or mode != 'sentence'):
        return jsonify({'error': "'method' 'nlp' only supports preset='lite' and mode='sentence'"}), 400

    try:
        # Auto-detect language if not specified. A detector that cannot
        # classify the input returns None or "unknown"; normalize to "unknown"
        # so the response never leaks a None model tag.
        if language is None:
            language = detect_language(text) or "unknown"

        fallback = False
        uncompressed = False

        if language not in MLM_LANGUAGES:
            # Non-active language: return the input unchanged. There is no
            # provisioned model for this language, so no compression (and no
            # NLP fallback) is attempted.
            compressed = text
            model = f"caveman-uncompressed-{language}"
            uncompressed = True
        else:
            # Reject incompatible language/mode combinations before inference
            if language == 'zh' and mode == 'text':
                return jsonify({'error': "'mode' 'text' is not supported for Chinese (zh); use mode='sentence'"}), 400

            if method == 'nlp':
                # Force NLP mode
                compressed = compress_text_nlp(text, lang=language)
                model = f"caveman-nlp-{language}"
            else:
                # method="mlm" (default): try MLM, fallback to NLP
                try:
                    compressed = compress_text(text, language=language, preset=preset, mode=mode)
                    model = f"caveman-mlm-{language}"
                except OSError as e:
                    # MLM model unavailable for this language: expected fallback.
                    print(f"Warning: MLM unavailable for {language}, falling back to NLP: {e}", file=sys.stderr)
                    compressed = compress_text_nlp(text, lang=language)
                    model = f"caveman-nlp-{language}"
                    fallback = True
                    # NLP fallback is always lite/sentence; report the actual
                    # effective preset/mode rather than the requested MLM values.
                    preset = 'lite'
                    mode = 'sentence'
                except InputTooLongError as e:
                    return jsonify({'error': str(e)}), 413

        return jsonify({
            'compressed': compressed,
            'language': language,
            'model': model,
            'preset': preset,
            'mode': mode,
            'fallback': fallback,
            'uncompressed': uncompressed,
            'original_size': len(text),
            'compressed_size': len(compressed),
            'compression_ratio': len(compressed) / len(text) if text else 0
        })
    except Exception as e:
        # Log the real error, return a generic body so internals are not leaked.
        app.logger.error("Compression failed: %s", e, exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
