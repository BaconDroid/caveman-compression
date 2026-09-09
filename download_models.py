#!/usr/bin/env python3
"""
Download ML models for Caveman Compression.

Runs during the Docker build (as root) to provision models into the immutable
image. The runtime is offline and never downloads models.

Exit codes:
    0 - success (spaCy/MLM models downloaded and the language manifest written;
        fastText failure is non-fatal)
    1 - invalid custom model configuration, or a requested spaCy/MLM model
        failed to download
"""

import json
import os
import shutil
import sys
import tempfile
import urllib.request

from language_catalog import (
    DEFAULT_LANGUAGES,
    MANIFEST_PATH,
    MLM_LANGUAGE_CATALOGUE,
    write_language_manifest,
)

# spaCy pipeline names and HuggingFace MLM model ids by language code, derived
# from the shared catalogue in language_catalog.py. The catalogue is the single
# source of truth; these flat projections exist only for the downloader's
# convenience and cover the full catalogue (en/de/fr/es). Additional languages
# can be supplied via CUSTOM_MLM_MODELS at build time.
SPACY_MODELS = {lang: cfg["spacy"] for lang, cfg in MLM_LANGUAGE_CATALOGUE.items()}
MLM_MODELS = {lang: cfg["model"] for lang, cfg in MLM_LANGUAGE_CATALOGUE.items()}

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_PATH = "/app/models/lid.176.bin"
DOWNLOAD_TIMEOUT = 600
FASTTEXT_MAX_ATTEMPTS = 2


def download_spacy_model(model_name):
    """Download a spaCy pipeline via spaCy's resolver. Returns True on success.

    spaCy's resolver (``spacy.cli.download``) selects the pipeline version
    compatible with the installed spaCy release before installing it. This
    replaces the former ``pip install <pipeline-name>`` approach and runs
    during the image build, where the process runs as root.
    """
    print(f"Downloading spaCy model '{model_name}'...", file=sys.stderr)
    try:
        from spacy.cli import download as spacy_download

        spacy_download(model_name)
        print(f"  OK spaCy model '{model_name}' downloaded", file=sys.stderr)
        return True
    except SystemExit as e:
        # spaCy's downloader calls sys.exit on resolution or install failure.
        print(
            f"  FAILED to download spaCy model '{model_name}': "
            f"spaCy resolver exited with code {e.code}",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        print(f"  FAILED to download spaCy model '{model_name}': {e}", file=sys.stderr)
        return False


def download_mlm_model(model_name, lang_code):
    """Download a HuggingFace MLM model. Returns True on success."""
    print(f"Downloading MLM model '{model_name}' for '{lang_code}'...", file=sys.stderr)
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        AutoTokenizer.from_pretrained(model_name)
        AutoModelForMaskedLM.from_pretrained(model_name)
        print(f"  OK MLM model '{model_name}' downloaded", file=sys.stderr)
        return True
    except Exception as e:
        print(f"  FAILED to download MLM model '{model_name}': {e}", file=sys.stderr)
        return False


def download_to(url, dest, timeout=DOWNLOAD_TIMEOUT):
    """
    Download url to dest via a temporary file in the same directory, then
    atomically rename it into place. Raises on error and cleans up the
    temporary file.
    """
    dest_dir = os.path.dirname(dest) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmp_file, urllib.request.urlopen(
            url, timeout=timeout
        ) as response:
            shutil.copyfileobj(response, tmp_file)
        os.replace(tmp_path, dest)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def validate_fasttext_model(path):
    """
    Validate a fastText artifact by attempting to load it.

    A load attempt is the only reliable validity check: fastText publishes no
    official checksum for lid.176.bin, so we must not fabricate one. Returns
    True when the file is a usable fastText model, False otherwise. Never
    raises.
    """
    try:
        import fasttext
    except ImportError:
        print("  WARN fasttext not installed; cannot validate model", file=sys.stderr)
        return False
    try:
        fasttext.load_model(path)
        return True
    except Exception as e:
        print(f"  WARN fastText model failed to load: {e}", file=sys.stderr)
        return False


def download_fasttext_model():
    """Download the fastText language detection model. Non-fatal on failure.

    Existing files are validated by a load attempt before being accepted, and
    freshly downloaded files are validated the same way; a bad file is
    discarded and re-downloaded before giving up.
    """
    print("Downloading fastText language detection model...", file=sys.stderr)
    try:
        if os.path.exists(FASTTEXT_PATH):
            if validate_fasttext_model(FASTTEXT_PATH):
                print(f"  OK fastText model already exists at {FASTTEXT_PATH}", file=sys.stderr)
                return
            print("  WARN existing fastText model is invalid; discarding", file=sys.stderr)
            try:
                os.unlink(FASTTEXT_PATH)
            except OSError:
                pass

        print(f"  Downloading from {FASTTEXT_URL}...", file=sys.stderr)
        for attempt in range(1, FASTTEXT_MAX_ATTEMPTS + 1):
            try:
                download_to(FASTTEXT_URL, FASTTEXT_PATH)
            except Exception as e:
                print(f"  WARN fastText download attempt {attempt} failed: {e}", file=sys.stderr)
                continue
            if validate_fasttext_model(FASTTEXT_PATH):
                print(f"  OK fastText model downloaded to {FASTTEXT_PATH}", file=sys.stderr)
                return
            print(f"  WARN fastText download attempt {attempt} is invalid; discarding", file=sys.stderr)
            try:
                os.unlink(FASTTEXT_PATH)
            except OSError:
                pass
        print("  WARN fastText model unavailable after retries", file=sys.stderr)
    except Exception as e:
        print(f"  WARN fastText model download failed: {e}", file=sys.stderr)
    print("  Language detection will fall back to the word-list heuristic", file=sys.stderr)


def parse_custom_models(custom_json):
    """
    Parse and validate the CUSTOM_MLM_MODELS JSON payload.

    Expected shape:
        {"<lang>": {"model": "<hf model id>", "spacy": "<spacy pipeline>"}}

    Returns (mlm_updates, spacy_updates, error):
      - mlm_updates / spacy_updates are dicts of validated entries.
      - error is a non-empty string when the payload itself is malformed
        (invalid JSON or not a JSON object); individual invalid entries are
        skipped with a warning rather than treated as fatal.
    """
    mlm_updates = {}
    spacy_updates = {}
    if not custom_json:
        return mlm_updates, spacy_updates, None

    try:
        custom = json.loads(custom_json)
    except json.JSONDecodeError as e:
        return mlm_updates, spacy_updates, f"invalid JSON: {e}"

    if not isinstance(custom, dict):
        return mlm_updates, spacy_updates, "must be a JSON object"

    for lang, config in custom.items():
        if not isinstance(lang, str) or not isinstance(config, dict):
            print(
                f"  WARN skipping invalid custom model config for '{lang}'",
                file=sys.stderr,
            )
            continue
        model = config.get("model")
        spacy = config.get("spacy")
        # A usable custom entry needs both a HuggingFace MLM model and a spaCy
        # pipeline. Without a spaCy pipeline the entry would provision but fail
        # at runtime, so require both and skip entries missing either.
        if not (isinstance(model, str) and model.strip()) or not (
            isinstance(spacy, str) and spacy.strip()
        ):
            print(
                f"  WARN skipping custom model config for '{lang}': "
                "both 'model' and 'spacy' must be non-empty",
                file=sys.stderr,
            )
            continue
        mlm_updates[lang] = model.strip()
        spacy_updates[lang] = spacy.strip()

    return mlm_updates, spacy_updates, None


def main():
    languages_env = os.environ.get("LANGUAGES", DEFAULT_LANGUAGES)
    languages = [lang.strip() for lang in languages_env.split(",") if lang.strip()]
    print(f"Downloading models for languages: {', '.join(languages)}", file=sys.stderr)

    mlm_models = dict(MLM_MODELS)
    spacy_models = dict(SPACY_MODELS)

    mlm_updates, spacy_updates, error = parse_custom_models(
        os.environ.get("CUSTOM_MLM_MODELS", "")
    )
    if error is not None:
        print(f"Error: CUSTOM_MLM_MODELS {error}", file=sys.stderr)
        return 1

    mlm_models.update(mlm_updates)
    spacy_models.update(spacy_updates)
    if mlm_updates or spacy_updates:
        custom_langs = sorted(set(mlm_updates) | set(spacy_updates))
        print(f"Added custom models: {custom_langs}", file=sys.stderr)

    # Every requested language must have a usable spaCy pipeline. A language
    # without one would provision "successfully" but fail at runtime, so reject
    # the configuration up front instead of baking a broken image.
    missing_spacy = [lang for lang in languages if lang not in spacy_models]
    if missing_spacy:
        print(
            f"Error: no spaCy pipeline available for: {', '.join(missing_spacy)}. "
            "Provide a spaCy pipeline via CUSTOM_MLM_MODELS or remove the "
            "language from LANGUAGES.",
            file=sys.stderr,
        )
        return 1

    failed = []

    # Requested spaCy models: failures are fatal.
    for lang in languages:
        if lang in spacy_models:
            if not download_spacy_model(spacy_models[lang]):
                failed.append(f"spaCy:{spacy_models[lang]}")

    # Requested MLM models: failures are fatal.
    for lang in languages:
        if lang in mlm_models:
            if not download_mlm_model(mlm_models[lang], lang):
                failed.append(f"MLM:{mlm_models[lang]}")

    # fastText is best-effort; failures do not affect the exit code.
    download_fasttext_model()

    if failed:
        print(
            f"Error: failed to download {len(failed)} model(s): {', '.join(failed)}",
            file=sys.stderr,
        )
        return 1

    # Write the checked-in-image manifest of successfully provisioned active
    # languages (including any custom build models). The runtime core reads this
    # manifest and never consults LANGUAGES / CUSTOM_MLM_MODELS environment
    # variables, so the active-language set is frozen to this build.
    manifest = {
        lang: {"model": mlm_models[lang], "spacy": spacy_models[lang]}
        for lang in languages
    }
    write_language_manifest(MANIFEST_PATH, manifest)
    print(f"Wrote language manifest to {MANIFEST_PATH}", file=sys.stderr)

    print("\nModel downloads complete!", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
