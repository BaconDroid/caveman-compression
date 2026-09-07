#!/usr/bin/env python3
"""
MLM-based caveman compression using RoBERTa (English) and CamemBERT (French).
Removes highly predictable tokens based on masked language model probabilities.
No LLM API required - uses local models for deterministic compression.

Probability thresholds:
  P >= 1e-3:  Conservative (16% reduction, 98% accuracy)
  P >= 1e-4:  Moderate (19% reduction, 97% accuracy)
  P >= 1e-5:  Balanced (32% reduction, 92% accuracy) [DEFAULT]
  P >= 1e-6:  Aggressive (54% reduction, 83% accuracy)
"""

import sys
import os
import json
import argparse
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from transformers import RobertaForMaskedLM, RobertaTokenizer, CamembertForMaskedLM, CamembertTokenizer
    import spacy
except ImportError:
    print("Error: Required packages not installed. Install with:", file=sys.stderr)
    print("  pip install torch transformers spacy", file=sys.stderr)
    print("  python -m spacy download en_core_web_sm", file=sys.stderr)
    print("  python -m spacy download fr_core_news_sm", file=sys.stderr)
    sys.exit(1)

# Model cache
_models = {}
_device = None
_nlp_models = {}
_fasttext_model = None

# Default supported languages
DEFAULT_LANGUAGES = {
    "de": {"model": "bert-base-german-cased", "spacy": "de_core_news_sm"},
    "en": {"model": "roberta-base", "spacy": "en_core_web_sm"},
    "fr": {"model": "camembert-base", "spacy": "fr_core_news_sm"},
    "it": {"model": "dbmdz/bert-base-italian-cased", "spacy": "it_core_news_sm"},
    "pt": {"model": "neuralmind/bert-base-portuguese-cased", "spacy": "pt_core_news_sm"},
    "tr": {"model": "dbmdz/bert-base-turkish-cased", "spacy": "tr_core_news_sm"},
    "zh": {"model": "bert-base-chinese", "spacy": "zh_core_web_sm"},
}

# Load custom models from environment variable
# Format: CUSTOM_MLM_MODELS='{"sv": {"model": "KB/bert-base-swedish-cased", "spacy": "sv_core_news_sm"}}'
def _load_custom_models():
    """Load custom MLM models from CUSTOM_MLM_MODELS environment variable"""
    custom_json = os.environ.get("CUSTOM_MLM_MODELS")
    if custom_json:
        try:
            custom = json.loads(custom_json)
            print(f"Loading custom MLM models: {list(custom.keys())}", file=sys.stderr)
            return custom
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid CUSTOM_MLM_MODELS JSON: {e}", file=sys.stderr)
    return {}

# Merge default and custom languages
SUPPORTED_LANGUAGES = {**DEFAULT_LANGUAGES, **_load_custom_models()}

def get_fasttext_model():
    """Get or load fastText language detection model"""
    global _fasttext_model
    if _fasttext_model is None:
        try:
            import fasttext
            # Try to find the model in common locations
            model_paths = [
                "/app/models/lid.176.bin",
                os.path.expanduser("~/.cache/fasttext/lid.176.bin"),
                "lid.176.bin",
            ]
            for path in model_paths:
                if os.path.exists(path):
                    _fasttext_model = fasttext.load_model(path)
                    print(f"fastText model loaded from {path}", file=sys.stderr)
                    return _fasttext_model
            print("Warning: fastText model not found, falling back to spaCy", file=sys.stderr)
        except ImportError:
            print("Warning: fasttext not installed, falling back to spaCy", file=sys.stderr)
    return _fasttext_model

def detect_language(text):
    """Detect language using fastText (primary) or spaCy (fallback)"""
    # Try fastText first (faster and more accurate)
    ft_model = get_fasttext_model()
    if ft_model is not None:
        try:
            # fastText expects single line text
            clean_text = text[:1000].replace('\n', ' ')
            predictions = ft_model.predict(clean_text)
            lang_code = predictions[0][0].replace('__label__', '')
            confidence = predictions[1][0]
            if confidence > 0.3:  # Minimum confidence threshold
                return lang_code
        except Exception:
            pass
    
    # Fallback to spaCy
    for lang_code in ["fr", "en"]:
        try:
            nlp = get_nlp_model(lang_code)
            doc = nlp(text[:1000])
            if doc.lang_ == lang_code:
                return lang_code
        except:
            pass
    
    # Simple heuristic fallback
    french_words = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "est", "sont", "avoir", "être", "faire"}
    english_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does"}
    
    words = set(text.lower().split())
    french_count = len(words & french_words)
    english_count = len(words & english_words)
    
    return "fr" if french_count > english_count else "en"

def get_nlp_model(lang_code):
    """Get or load spaCy model for language"""
    if lang_code not in _nlp_models:
        model_name = SUPPORTED_LANGUAGES.get(lang_code, {}).get("spacy", "en_core_web_sm")
        try:
            _nlp_models[lang_code] = spacy.load(model_name)
        except OSError:
            print(f"Warning: spaCy model '{model_name}' not found. Using en_core_web_sm", file=sys.stderr)
            _nlp_models[lang_code] = spacy.load("en_core_web_sm")
    return _nlp_models[lang_code]

def get_mlm_model(lang_code):
    """Get or load MLM model for language using Auto classes"""
    if lang_code not in _models:
        config = SUPPORTED_LANGUAGES.get(lang_code, SUPPORTED_LANGUAGES["en"])
        model_name = config["model"]
        
        print(f"Loading MLM model '{model_name}' for language '{lang_code}'...", file=sys.stderr)
        
        # Use Auto classes for generic model loading
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        _models[lang_code] = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
        }
        print(f"Model '{model_name}' loaded.", file=sys.stderr)
    
    return _models[lang_code]

def get_mlm_probability(lang_code, sentence, word_idx):
    """Get MLM probability for a specific word in a sentence"""
    model_data = get_mlm_model(lang_code)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    device = model_data["device"]
    
    words = sentence.split()
    if word_idx >= len(words):
        return 0.0

    target_word = words[word_idx]
    masked_words = words.copy()
    masked_words[word_idx] = tokenizer.mask_token
    masked_sentence = " ".join(masked_words)

    try:
        inputs = tokenizer(masked_sentence, return_tensors="pt", max_length=512, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if len(mask_token_index) == 0:
            return 0.0

        mask_token_logits = logits[0, mask_token_index[0], :]
        probs = F.softmax(mask_token_logits, dim=0)

        target_tokens = tokenizer.encode((" " if word_idx > 0 else "") + target_word, add_special_tokens=False)
        if len(target_tokens) == 0:
            return 0.0

        target_token_id = target_tokens[0]
        prob = float(probs[target_token_id].item())

        return prob

    except Exception:
        return 0.0

def compress_text(text, language=None, drop_ratio=None, preset="lite", mode="sentence", calibrate_from_nlp=True, no_adjacent_removal=False, protect_ner=True):
    """
    Apply MLM-based compression by removing words whose predictability exceeds threshold.
    
    Args:
        text: Input text to compress
        language: Language code (auto-detect if None)
        prob_threshold: Absolute probability threshold for removal (legacy)
        drop_ratio: Fraction of words to drop (0.0-1.0). If set, uses adaptive threshold.
        preset: Preset name (lite/full/ultra) used with calibrate_from_nlp
        mode: "sentence" (NLP/MLM per sentence) or "text" (NLP/MLM on full text)
        calibrate_from_nlp: If True, calibrate drop_ratio from NLP compression
        no_adjacent_removal: If True, don't remove adjacent words
        protect_ner: If True, don't remove named entities
    """
    if not text or not text.strip():
        return text
    
    # Auto-detect language if not specified
    if language is None:
        language = detect_language(text)
    
    # Get NLP model for tokenization and NER
    nlp = get_nlp_model(language)
    doc = nlp(text)
    
    # Get MLM model
    get_mlm_model(language)
    
    # Calculate drop_ratio from NLP if needed
    if drop_ratio is None and calibrate_from_nlp:
        import math
        try:
            from caveman_compress_nlp import compress_text as compress_text_nlp
            
            if mode == "text":
                # NLP on full text
                nlp_result = compress_text_nlp(text, lang=language)
                nlp_kept_ratio = len(nlp_result) / len(text) if text else 1.0
                nlp_drop = 1.0 - nlp_kept_ratio
                nlp_drop = round(nlp_drop * 10) / 10
                
                if preset == "lite":
                    drop_ratio = nlp_drop
                elif preset == "full":
                    drop_ratio = min(0.9, math.ceil(nlp_drop * 1.5 * 10) / 10)
                elif preset == "ultra":
                    drop_ratio = min(0.9, math.ceil(nlp_drop * 2 * 10) / 10)
                else:
                    drop_ratio = nlp_drop
        except Exception:
            drop_ratio = 0.3  # fallback
    
    if drop_ratio is None:
        drop_ratio = 0.3  # default fallback
    
    # Process based on mode
    if mode == "text":
        # Text mode: NLP on full text, MLM on full text context
        return _compress_text_mode(text, doc, language, drop_ratio, no_adjacent_removal, protect_ner)
    else:
        # Sentence mode: NLP per sentence, MLM per sentence context (default)
        return _compress_sentence_mode(text, doc, language, preset, drop_ratio, no_adjacent_removal, protect_ner)

def _compress_text_mode(text, doc, language, drop_ratio, no_adjacent_removal, protect_ner):
    """Text mode: compress using full text context"""
    # Use spaCy tokens for consistent indexing
    tokens = [token.text for token in doc if not token.is_space]
    if len(tokens) < 3:
        return text
    
    # Get NER spans (spaCy token indices)
    ner_spans = set()
    if protect_ner:
        for ent in doc.ents:
            for token in ent:
                ner_spans.add(token.i)
    
    # Calculate probabilities for all tokens
    word_probs = []
    for i, token in enumerate(tokens):
        if len(token) <= 2 or not token.isalpha():
            word_probs.append((i, 0.0))
        elif protect_ner and i in ner_spans:
            word_probs.append((i, 0.0))
        else:
            prob = get_mlm_probability(language, text, i)
            word_probs.append((i, prob))
    
    # Calculate how many words to drop
    num_to_drop = int(len(words) * drop_ratio)
    
    # Sort by probability (highest first = most predictable)
    sortable_probs = [(i, p) for i, p in word_probs if p > 0]
    sortable_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Drop the most predictable words
    to_remove = set()
    for i, p in sortable_probs:
        if len(to_remove) >= num_to_drop:
            break
        to_remove.add(i)
    
    # Apply no_adjacent_removal constraint
    if no_adjacent_removal:
        filtered_remove = set()
        prev_removed = False
        for i in range(len(words)):
            if i in to_remove:
                if not prev_removed:
                    filtered_remove.add(i)
                    prev_removed = True
            else:
                prev_removed = False
        to_remove = filtered_remove
    
    # Reconstruct
    compressed_words = [w for i, w in enumerate(words) if i not in to_remove]
    return " ".join(compressed_words)

def _compress_sentence_mode(text, doc, language, preset, drop_ratio, no_adjacent_removal, protect_ner):
    """Sentence mode: compress per sentence (default)"""
    import math
    from caveman_compress_nlp import compress_text as compress_text_nlp
    
    result_parts = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        
        words = sent_text.split()
        if len(words) < 3:
            result_parts.append(sent_text)
            continue
        
        # Calculate drop_ratio for this sentence
        sent_drop_ratio = drop_ratio
        try:
            nlp_result = compress_text_nlp(sent_text, lang=language)
            nlp_kept_ratio = len(nlp_result) / len(sent_text) if sent_text else 1.0
            nlp_drop = 1.0 - nlp_kept_ratio
            nlp_drop = round(nlp_drop * 10) / 10
            
            if preset == "lite":
                sent_drop_ratio = nlp_drop
            elif preset == "full":
                sent_drop_ratio = min(0.9, math.ceil(nlp_drop * 1.5 * 10) / 10)
            elif preset == "ultra":
                sent_drop_ratio = min(0.9, math.ceil(nlp_drop * 2 * 10) / 10)
            else:
                sent_drop_ratio = nlp_drop
        except Exception:
            sent_drop_ratio = drop_ratio  # fallback to global
        
        # Get NER spans
        ner_spans = set()
        if protect_ner:
            sent_start = sent.start
            for ent in sent.ents:
                for token in ent:
                    ner_spans.add(token.i - sent_start)
        
        # Calculate probabilities
        word_probs = []
        for i, word in enumerate(words):
            if len(word) <= 2 or not word.isalpha():
                word_probs.append((i, 0.0))
            elif protect_ner and i in ner_spans:
                word_probs.append((i, 0.0))
            else:
                prob = get_mlm_probability(language, sent_text, i)
                word_probs.append((i, prob))
        
        # Calculate how many words to drop
        num_to_drop = int(len(words) * sent_drop_ratio)
        
        # Sort by probability
        sortable_probs = [(i, p) for i, p in word_probs if p > 0]
        sortable_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Drop the most predictable words
        to_remove = set()
        for i, p in sortable_probs:
            if len(to_remove) >= num_to_drop:
                break
            to_remove.add(i)
        
        # Apply no_adjacent_removal
        if no_adjacent_removal:
            filtered_remove = set()
            prev_removed = False
            for i in range(len(words)):
                if i in to_remove:
                    if not prev_removed:
                        filtered_remove.add(i)
                        prev_removed = True
                else:
                    prev_removed = False
            to_remove = filtered_remove
        
        # Reconstruct
        compressed_words = [w for i, w in enumerate(words) if i not in to_remove]
        result_parts.append(" ".join(compressed_words))
    
    return " ".join(result_parts)

def count_tokens(text):
    """Estimate token count: characters / 4"""
    return len(text.strip()) // 4

def main():
    parser = argparse.ArgumentParser(description="MLM-based caveman compression")
    parser.add_argument("action", choices=["compress"], help="Action to perform")
    parser.add_argument("text", nargs="?", help="Text to compress")
    parser.add_argument("-f", "--file", help="Input file")
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("-l", "--language", help="Language code (en, fr)")
    parser.add_argument("-k", "--threshold", type=float, default=1e-5, help="Probability threshold")
    parser.add_argument("--no-adjacent", action="store_true", help="Don't remove adjacent words")
    parser.add_argument("--no-ner", action="store_true", help="Don't protect named entities")
    
    args = parser.parse_args()
    
    # Get input text
    if args.file:
        with open(args.file, "r") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Error: Provide text or use -f for file", file=sys.stderr)
        sys.exit(1)
    
    if args.action == "compress":
        compressed = compress_text(
            text,
            language=args.language,
            prob_threshold=args.threshold,
            no_adjacent_removal=args.no_adjacent,
            protect_ner=not args.no_ner
        )
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(compressed)
            print(f"Compressed text written to {args.output}", file=sys.stderr)
        else:
            print(compressed)

if __name__ == "__main__":
    main()
