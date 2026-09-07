#!/usr/bin/env python3
"""
Startup script for Caveman Compression Docker container.
Downloads MLM and spaCy models based on LANGUAGES environment variable.
"""

import os
import sys
import subprocess

# Default languages
DEFAULT_LANGUAGES = "en,fr"

# Model mappings
MLM_MODELS = {
    "en": "roberta-base",
    "fr": "camembert-base",
    "de": "bert-base-german-cased",
    "es": "dccuchile/bert-base-spanish-wwm-cased",
    "it": "dbmdz/bert-base-italian-cased",
    "pt": "neuralmind/bert-base-portuguese-cased",
    "nl": "wietsedv/bert-base-dutch-cased",
    "zh": "bert-base-chinese",
    "ja": "cl-tohoku/bert-base-japanese",
    "ko": "snunlp/KR-Finetuned-ColBERT-Ko",
    "ru": "DeepPavlov/rubert-base-cased",
    "ar": "asafaya/bert-base-arabic",
}

SPACY_MODELS = {
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "ru": "ru_core_news_sm",
    "ar": "ar_core_news_sm",
}

def download_models(languages):
    """Download MLM and spaCy models for specified languages"""
    lang_list = [lang.strip() for lang in languages.split(",")]
    
    print(f"Downloading models for languages: {', '.join(lang_list)}")
    
    for lang in lang_list:
        if lang not in MLM_MODELS:
            print(f"Warning: No MLM model configured for language '{lang}', skipping")
            continue
        
        # Download MLM model
        mlm_model = MLM_MODELS[lang]
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
        
        # Download spaCy model
        spacy_model = SPACY_MODELS.get(lang)
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
    
    print("Model downloads complete!")

if __name__ == "__main__":
    languages = os.environ.get("LANGUAGES", DEFAULT_LANGUAGES)
    print(f"LANGUAGES environment variable: {languages}")
    download_models(languages)
