#!/usr/bin/env python3
"""
Download ML models for Caveman Compression.
Runs during Docker build to pre-download models.
"""

import os
import sys
import subprocess

def download_spacy_model(model_name):
    """Download spaCy model"""
    print(f"Downloading spaCy model '{model_name}'...", file=sys.stderr)
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        print(f"  ✓ spaCy model '{model_name}' downloaded", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to download spaCy model '{model_name}': {e}", file=sys.stderr)

def download_mlm_model(model_name, lang_code):
    """Download MLM model from HuggingFace"""
    print(f"Downloading MLM model '{model_name}' for '{lang_code}'...", file=sys.stderr)
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForMaskedLM.from_pretrained(model_name)
        print(f"  ✓ MLM model '{model_name}' downloaded", file=sys.stderr)
    except Exception as e:
        print(f"  ✗ Failed to download MLM model '{model_name}': {e}", file=sys.stderr)

def download_fasttext_model():
    """Download fastText language detection model"""
    print("Downloading fastText language detection model...", file=sys.stderr)
    try:
        import fasttext
        # fasttext.load_model will download if not found
        # But we need to explicitly download lid.176.bin
        model_path = "/app/models/lid.176.bin"
        os.makedirs("/app/models", exist_ok=True)
        
        if not os.path.exists(model_path):
            # Download from official fastText repository
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            print(f"  Downloading from {url}...", file=sys.stderr)
            urllib.request.urlretrieve(url, model_path)
            print(f"  ✓ fastText model downloaded to {model_path}", file=sys.stderr)
        else:
            print(f"  ✓ fastText model already exists at {model_path}", file=sys.stderr)
    except Exception as e:
        print(f"  ✗ Failed to download fastText model: {e}", file=sys.stderr)
        print("  Language detection will fall back to spaCy", file=sys.stderr)

def main():
    # Get languages from environment variable
    languages_env = os.environ.get("LANGUAGES", "en,fr")
    languages = [lang.strip() for lang in languages_env.split(",")]
    
    print(f"Downloading models for languages: {', '.join(languages)}", file=sys.stderr)
    
    # spaCy model mappings
    spacy_models = {
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "es": "es_core_news_sm",
        "it": "it_core_news_sm",
        "pt": "pt_core_news_sm",
        "nl": "nl_core_news_sm",
        "zh": "zh_core_web_sm",
    }
    
    # MLM model mappings
    mlm_models = {
        "en": "roberta-base",
        "fr": "camembert-base",
        "de": "bert-base-german-cased",
        "it": "dbmdz/bert-base-italian-cased",
        "pt": "neuralmind/bert-base-portuguese-cased",
        "tr": "dbmdz/bert-base-turkish-cased",
        "zh": "bert-base-chinese",
    }
    
    # Load custom models from environment
    custom_json = os.environ.get("CUSTOM_MLM_MODELS")
    if custom_json:
        try:
            import json
            custom = json.loads(custom_json)
            for lang, config in custom.items():
                if "model" in config:
                    mlm_models[lang] = config["model"]
                if "spacy" in config:
                    spacy_models[lang] = config["spacy"]
            print(f"Added custom models: {list(custom.keys())}", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid CUSTOM_MLM_MODELS JSON: {e}", file=sys.stderr)
    
    # Download spaCy models
    for lang in languages:
        if lang in spacy_models:
            download_spacy_model(spacy_models[lang])
    
    # Download MLM models
    for lang in languages:
        if lang in mlm_models:
            download_mlm_model(mlm_models[lang], lang)
    
    # Download fastText model
    download_fasttext_model()
    
    print("\nModel downloads complete!", file=sys.stderr)

if __name__ == "__main__":
    main()
