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

SUPPORTED_LANGUAGES = {
    "en": {"model": "roberta-base", "spacy": "en_core_web_sm"},
    "fr": {"model": "camembert-base", "spacy": "fr_core_news_sm"},
}

def detect_language(text):
    """Detect language using spaCy or simple heuristics"""
    # Try spaCy language detection
    for lang_code in ["fr", "en"]:
        try:
            nlp = get_nlp_model(lang_code)
            doc = nlp(text[:1000])  # Limit text length for detection
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
    """Get or load MLM model for language"""
    if lang_code not in _models:
        config = SUPPORTED_LANGUAGES.get(lang_code, SUPPORTED_LANGUAGES["en"])
        model_name = config["model"]
        
        print(f"Loading MLM model '{model_name}' for language '{lang_code}'...", file=sys.stderr)
        
        if lang_code == "fr":
            tokenizer = CamembertTokenizer.from_pretrained(model_name)
            model = CamembertForMaskedLM.from_pretrained(model_name)
        else:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaForMaskedLM.from_pretrained(model_name)
        
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

        target_tokens = tokenizer.encode(target_word, add_special_tokens=False)
        if len(target_tokens) == 0:
            return 0.0

        target_token_id = target_tokens[0]
        prob = float(probs[target_token_id].item())

        return prob

    except Exception:
        return 0.0

def compress_text(text, language=None, prob_threshold=1e-5, no_adjacent_removal=False, protect_ner=True):
    """
    Apply MLM-based compression by removing words whose predictability exceeds threshold.
    
    Args:
        text: Input text to compress
        language: Language code (auto-detect if None)
        prob_threshold: Probability threshold for removal (lower = more aggressive)
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
    
    # Process sentence by sentence
    result_parts = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        
        words = sent_text.split()
        if len(words) < 3:  # Don't compress very short sentences
            result_parts.append(sent_text)
            continue
        
        # Get NER spans to protect
        ner_spans = set()
        if protect_ner:
            for ent in sent.ents:
                for token in ent:
                    ner_spans.add(token.i)
        
        # Calculate probabilities and mark words for removal
        to_remove = set()
        prev_removed = False
        
        for i, word in enumerate(words):
            # Skip short words and punctuation
            if len(word) <= 2 or not word.isalpha():
                prev_removed = False
                continue
            
            # Skip NER tokens
            if protect_ner and i in ner_spans:
                prev_removed = False
                continue
            
            # Get MLM probability
            prob = get_mlm_probability(language, sent_text, i)
            
            # Mark for removal if probability exceeds threshold
            if prob >= prob_threshold:
                if no_adjacent_removal and prev_removed:
                    continue
                to_remove.add(i)
                prev_removed = True
            else:
                prev_removed = False
        
        # Reconstruct sentence without removed words
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
