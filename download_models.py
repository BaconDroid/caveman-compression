#!/usr/bin/env python3
"""
Startup script for Caveman Compression Docker container.
Downloads MLM and spaCy models based on LANGUAGES environment variable.
Supports custom models via CUSTOM_MLM_MODELS and CUSTOM_SPACY_MODELS.
"""

import os
import sys
import subprocess

# Default languages (only official/source models)
DEFAULT_LANGUAGES = "en,fr"

# Official/source MLM models (highest quality)
MLM_MODELS = {
    "de": "bert-base-german-cased",              # Google's German BERT
    "en": "roberta-base",                         # Meta's RoBERTa (English)
    "fr": "camembert-base",                       # CamemBERT (French RoBERTa)
    "it": "dbmdz/bert-base-italian-cased",        # MDZ's Italian BERT
    "pt": "neuralmind/bert-base-portuguese-cased", # NeuralMind's Portuguese BERT
    "tr": "dbmdz/bert-base-turkish-cased",        # MDZ's Turkish BERT
    "zh": "bert-base-chinese",                    # Google's Chinese BERT
}

# Official spaCy models
SPACY_MODELS = {
    "de": "de_core_news_sm",
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "tr": "tr_core_news_sm",
    "zh": "zh_core_web_sm",
}

def parse_custom_models(env_var, default_models):
    """Parse custom models from environment variable.
    
    Format: "lang1:model1,lang2:model2"
    Example: "de:bert-base-german-cased,es:dccuchile/bert-base-spanish-wwm-cased"
    """
    custom = {}
    value = os.environ.get(env_var, "")
    if value:
        for pair in value.split(","):
            pair = pair.strip()
            if ":" in pair:
                lang, model = pair.split(":", 1)
                custom[lang.strip()] = model.strip()
    return {**default_models, **custom}

def download_models(languages, mlm_models, spacy_models):
    """Download MLM and spaCy models for specified languages"""
    lang_list = [lang.strip() for lang in languages.split(",")]
    
    print(f"Downloading models for languages: {', '.join(lang_list)}")
    
    errors = []
    
    for lang in lang_list:
        if lang not in mlm_models:
            print(f"Warning: No MLM model configured for language '{lang}', skipping")
            continue
        
        # Download MLM model
        mlm_model = mlm_models[lang]
        print(f"Downloading MLM model '{mlm_model}' for '{lang}'...")
        try:
            if lang == "fr":
                from transformers import CamembertTokenizer, CamembertForMaskedLM
                CamembertTokenizer.from_pretrained(mlm_model)
                CamembertForMaskedLM.from_pretrained(mlm_model)
            else:
                from transformers import AutoTokenizer, AutoModelForMaskedLM
                AutoTokenizer.from_pretrained(mlm_model)
                AutoModelForMaskedLM.from_pretrained(mlm_model)
            print(f"  ✓ MLM model '{mlm_model}' downloaded")
        except Exception as e:
            print(f"  ✗ Failed to download MLM model '{mlm_model}': {e}")
            errors.append(f"MLM model '{mlm_model}' for '{lang}'")
        
        # Download spaCy model
        spacy_model = spacy_models.get(lang)
        if spacy_model:
            print(f"Downloading spaCy model '{spacy_model}' for '{lang}'...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", spacy_model],
                    check=True,
                    capture_output=True
                )
                print(f"  ✓ spaCy model '{spacy_model}' downloaded")
            except Exception as e:
                print(f"  ✗ Failed to download spaCy model '{spacy_model}': {e}")
                errors.append(f"spaCy model '{spacy_model}' for '{lang}'")
    
    if errors:
        print(f"\nFailed to download {len(errors)} model(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    
    print("Model downloads complete!")

if __name__ == "__main__":
    languages = os.environ.get("LANGUAGES", DEFAULT_LANGUAGES)
    print(f"LANGUAGES environment variable: {languages}")
    
    # Parse custom models
    mlm_models = parse_custom_models("CUSTOM_MLM_MODELS", MLM_MODELS)
    spacy_models = parse_custom_models("CUSTOM_SPACY_MODELS", SPACY_MODELS)
    
    if os.environ.get("CUSTOM_MLM_MODELS"):
        print(f"Custom MLM models: {os.environ.get('CUSTOM_MLM_MODELS')}")
    if os.environ.get("CUSTOM_SPACY_MODELS"):
        print(f"Custom spaCy models: {os.environ.get('CUSTOM_SPACY_MODELS')}")
    
    download_models(languages, mlm_models, spacy_models)
