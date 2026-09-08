"""Focused regression tests for caveman_compress_mlm fixes.

Covers:
- ModelUnavailableError contract (OSError subclass, shared between modules).
- CUSTOM_MLM_MODELS structure validation.
- Unsupported language no longer silently mapped to English.
- Punctuation-before-NER char-offset protection (index alignment).
- Core input-length bound (ValueError, no silent truncation).
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The server/MCP boundary tests stub the backend modules in sys.modules with
# bare ModuleType objects. Drop such stubs so this module exercises the real
# backend implementations instead of the stubs.
for _name in ("caveman_compress_mlm", "caveman_compress_nlp"):
    _mod = sys.modules.get(_name)
    if _mod is not None and getattr(_mod, "__file__", None) is None:
        del sys.modules[_name]

import spacy  # noqa: E402

import caveman_compress_mlm as mlm  # noqa: E402
from caveman_compress_nlp import ModelUnavailableError  # noqa: E402

HAS_EN_MODEL = spacy.util.is_package("en_core_web_sm")


class TestModelUnavailableError(unittest.TestCase):
    def test_subclasses_oserror(self):
        self.assertTrue(issubclass(ModelUnavailableError, OSError))

    def test_shared_between_modules(self):
        self.assertIs(mlm.ModelUnavailableError, ModelUnavailableError)


class TestCustomModelValidation(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("CUSTOM_MLM_MODELS")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("CUSTOM_MLM_MODELS", None)
        else:
            os.environ["CUSTOM_MLM_MODELS"] = self._old

    def test_valid_entry(self):
        os.environ["CUSTOM_MLM_MODELS"] = (
            '{"sv": {"model": "KB/bert-base-swedish-cased", "spacy": "sv_core_news_sm"}}'
        )
        self.assertEqual(
            mlm._load_custom_models(),
            {"sv": {"model": "KB/bert-base-swedish-cased", "spacy": "sv_core_news_sm"}},
        )

    def test_missing_model_rejected(self):
        os.environ["CUSTOM_MLM_MODELS"] = '{"sv": {"spacy": "sv_core_news_sm"}}'
        self.assertEqual(mlm._load_custom_models(), {})

    def test_missing_spacy_rejected(self):
        # A model without a spaCy pipeline would provision but fail at runtime.
        os.environ["CUSTOM_MLM_MODELS"] = '{"sv": {"model": "KB/bert-base-swedish-cased"}}'
        self.assertEqual(mlm._load_custom_models(), {})

    def test_empty_model_rejected(self):
        os.environ["CUSTOM_MLM_MODELS"] = '{"sv": {"model": ""}}'
        self.assertEqual(mlm._load_custom_models(), {})

    def test_non_mapping_value_rejected(self):
        os.environ["CUSTOM_MLM_MODELS"] = '{"sv": "KB/bert-base-swedish-cased"}'
        self.assertEqual(mlm._load_custom_models(), {})

    def test_invalid_json_rejected(self):
        os.environ["CUSTOM_MLM_MODELS"] = "{invalid"
        self.assertEqual(mlm._load_custom_models(), {})

    def test_mixed_entries_keep_only_valid(self):
        os.environ["CUSTOM_MLM_MODELS"] = (
            '{"sv": {"model": "KB/x", "spacy": "sv_core_news_sm"}, '
            '"nospacy": {"model": "KB/y"}, "bad": {"model": ""}, "alsobad": 5}'
        )
        self.assertEqual(
            mlm._load_custom_models(),
            {"sv": {"model": "KB/x", "spacy": "sv_core_news_sm"}},
        )


class TestUnsupportedLanguage(unittest.TestCase):
    def test_get_nlp_model_raises(self):
        with self.assertRaises(ModelUnavailableError):
            mlm.get_nlp_model("xx_unsupported")

    def test_get_mlm_model_raises(self):
        with self.assertRaises(ModelUnavailableError):
            mlm.get_mlm_model("xx_unsupported")


class TestInputBound(unittest.TestCase):
    def test_too_long_raises_valueerror(self):
        text = "word " * (mlm.MAX_INPUT_LENGTH + 1)
        with self.assertRaises(ValueError):
            mlm.compress_text(text, language="en")

    def test_empty_text_passthrough(self):
        self.assertEqual(mlm.compress_text(""), "")

    def test_short_text_does_not_hit_length_bound(self):
        # An unsupported language raises ModelUnavailableError, not the length
        # ValueError, proving the bound is enforced on length alone.
        with self.assertRaises(ModelUnavailableError):
            mlm.compress_text("short text here", language="xx_unsupported")


@unittest.skipUnless(HAS_EN_MODEL, "en_core_web_sm not installed")
class TestNerProtectionWithPunctuation(unittest.TestCase):
    def _doc(self, text):
        return spacy.load("en_core_web_sm")(text)

    def test_punctuation_before_entity_aligns_char_offsets(self):
        text = "Hello, John Smith lives in Paris."
        doc = self._doc(text)
        sent = next(doc.sents)
        raw = sent.text
        leading = len(raw) - len(raw.lstrip())
        sent_text = raw.strip()
        words = sent_text.split()
        # "John" is whitespace word index 1 despite the preceding comma token.
        self.assertEqual(words, ["Hello,", "John", "Smith", "lives", "in", "Paris."])

        sent_start_char = sent.start_char + leading
        word_offsets = mlm._word_char_offsets(sent_text, words)
        ner_ranges = [
            (ent.start_char - sent_start_char, ent.end_char - sent_start_char)
            for ent in doc.ents
        ]
        protected = mlm._ner_covered_words(word_offsets, ner_ranges)
        self.assertIn(1, protected)  # "John"
        self.assertIn(2, protected)  # "Smith"
        self.assertIn(5, protected)  # "Paris."

    def test_compress_text_mode_keeps_entities_and_no_nameerror(self):
        text = "Hello, John Smith lives in Paris. He likes New York."
        doc = self._doc(text)

        original = mlm.get_mlm_probability
        mlm.get_mlm_probability = lambda lang, sentence, word_idx: 0.9
        try:
            out = mlm._compress_text_mode(
                text, doc, "en", drop_ratio=0.5,
                no_adjacent_removal=False, protect_ner=True,
            )
        finally:
            mlm.get_mlm_probability = original

        # Named entities must survive even at a high drop ratio.
        for word in ("John", "Smith", "Paris.", "New", "York."):
            self.assertIn(word, out)


class _FakeTokenizer:
    """Returns a fixed token count regardless of input, for token-bound tests."""

    def __init__(self, n_tokens):
        self.n_tokens = n_tokens

    def encode(self, text, truncation=False):
        return list(range(self.n_tokens))


class TestTokenBound(unittest.TestCase):
    def test_input_too_long_error_is_exported_valueerror(self):
        self.assertTrue(issubclass(mlm.InputTooLongError, ValueError))

    def test_over_512_tokens_under_4096_chars_raises(self):
        # Short in characters (far below MAX_INPUT_LENGTH) but the tokenizer
        # output exceeds MAX_SEQ_LENGTH: the token bound, not the char bound,
        # must reject the input.
        text = "word " * 200  # 1000 chars, well under 4096
        fake = _FakeTokenizer(mlm.MAX_SEQ_LENGTH + 1)
        with mock.patch.object(mlm, "get_mlm_model", return_value={"tokenizer": fake}):
            with self.assertRaises(mlm.InputTooLongError):
                mlm.compress_text(text, language="en")

    def test_at_limit_does_not_raise_token_error(self):
        # Exactly MAX_SEQ_LENGTH tokens must pass the token bound and continue
        # into the pipeline (reaching the NLP stage here).
        fake = _FakeTokenizer(mlm.MAX_SEQ_LENGTH)

        def _reach_nlp(lang):
            raise RuntimeError("reached nlp stage")

        with mock.patch.object(mlm, "get_mlm_model", return_value={"tokenizer": fake}), \
                mock.patch.object(mlm, "get_nlp_model", side_effect=_reach_nlp):
            with self.assertRaises(RuntimeError):
                mlm.compress_text("test", language="en")


class TestZhTextModeRejected(unittest.TestCase):
    def test_text_mode_rejected(self):
        with self.assertRaises(ValueError):
            mlm.compress_text("你好世界 hello world", language="zh", mode="text")


if __name__ == "__main__":
    unittest.main()
