"""Static catalogue of LT n-gram MLM languages.

Single source of truth for the language -> MLM model / spaCy pipeline mapping.
Imported by both the compressor (``caveman_compress_mlm.py``) and the model
downloader (``download_models.py``) so the registry is defined once and never
duplicated.

This module also owns the *provisioned language manifest*: the downloader
writes it into the image at build time and the compressor reads it back at
runtime. The runtime never activates languages from mutable environment
variables (``LANGUAGES`` / ``CUSTOM_MLM_MODELS``), so a running container's
active-language set is frozen to whatever was baked into the image.
"""

import json
import os
import sys

# Full catalogue of languages with LT n-gram MLM mappings. Every entry has a
# known MLM model and a known spaCy pipeline. Which languages are actually
# provisioned (build time) and activated (runtime) is controlled by the
# LANGUAGES build argument, whose result is recorded in the manifest below.
MLM_LANGUAGE_CATALOGUE = {
    "en": {"model": "roberta-base", "spacy": "en_core_web_sm"},
    "fr": {"model": "camembert-base", "spacy": "fr_core_news_sm"},
    "de": {"model": "bert-base-german-cased", "spacy": "de_core_news_sm"},
    "es": {"model": "dccuchile/bert-base-spanish-wwm-cased", "spacy": "es_core_news_sm"},
}

# Languages provisioned at build time by default. This is the local-development
# fallback when no manifest exists (no Docker build has produced one).
DEFAULT_LANGUAGES = "en,fr"

# Location of the checked-in-image manifest. The downloader writes it here at
# build time; the compressor reads it back at runtime. Overridable via
# LANGUAGE_MANIFEST_PATH for local development or non-standard layouts.
MANIFEST_PATH = os.environ.get("LANGUAGE_MANIFEST_PATH", "/app/models/languages.json")


def write_language_manifest(path, languages):
    """Write the provisioned active-language configs to a manifest file.

    Called by ``download_models.py`` at build time once provisioning succeeds.
    ``languages`` is a mapping of language code -> ``{"model": ..., "spacy": ...}``.
    The compressor reads the file back via :func:`load_active_languages`.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(languages, fh, ensure_ascii=False, indent=2)


def _default_languages():
    """Local-dev fallback: the default en/fr pair, resolved from the catalogue."""
    return {
        code: MLM_LANGUAGE_CATALOGUE[code]
        for code in DEFAULT_LANGUAGES.split(",")
        if code in MLM_LANGUAGE_CATALOGUE
    }


def _valid_entry(config):
    """Return True when a manifest entry is a usable {model, spacy} config."""
    return (
        isinstance(config, dict)
        and isinstance(config.get("model"), str)
        and config["model"].strip()
        and isinstance(config.get("spacy"), str)
        and config["spacy"].strip()
    )


def load_active_languages(manifest_path=None):
    """Return the active language -> config mapping from the build manifest.

    The runtime reads the manifest written into the image at build time; it
    never activates languages from mutable environment variables. When the
    manifest is absent (local development without a build), fall back to the
    default en/fr pair so the core still has a valid, minimal active set.

    A present-but-unreadable or malformed manifest falls back to the defaults
    with a warning rather than failing startup; invalid entries are skipped.
    """
    path = manifest_path or MANIFEST_PATH
    if not os.path.exists(path):
        return _default_languages()

    try:
        with open(path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: language manifest unreadable at {path}: {e}", file=sys.stderr)
        return _default_languages()

    if not isinstance(manifest, dict):
        print(f"Warning: language manifest at {path} is not an object; ignoring", file=sys.stderr)
        return _default_languages()

    return {
        lang: {"model": config["model"].strip(), "spacy": config["spacy"].strip()}
        for lang, config in manifest.items()
        if isinstance(lang, str) and _valid_entry(config)
    }
