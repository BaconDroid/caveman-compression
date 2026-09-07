"""Focused tests for download_models.py provisioning logic (no network)."""

import os
import sys
import tempfile
import unittest
from unittest import mock

import download_models


class CustomModelConfigTests(unittest.TestCase):
    def test_no_turkish_spacy_pipeline(self):
        # spaCy ships no Turkish pipeline, so Turkish must not be a default
        # language in either the spaCy or the MLM tables. A default MLM model
        # without a matching spaCy pipeline would provision but fail at runtime.
        self.assertNotIn("tr", download_models.SPACY_MODELS)
        self.assertNotIn("tr", download_models.MLM_MODELS)

    def test_valid_config(self):
        mlm, spacy, error = download_models.parse_custom_models(
            '{"sv": {"model": "KB/bert-base-swedish-cased", "spacy": "sv_core_news_sm"}}'
        )
        self.assertIsNone(error)
        self.assertEqual(mlm, {"sv": "KB/bert-base-swedish-cased"})
        self.assertEqual(spacy, {"sv": "sv_core_news_sm"})

    def test_invalid_json(self):
        _, _, error = download_models.parse_custom_models("{not json")
        self.assertIsNotNone(error)

    def test_non_dict_payload(self):
        _, _, error = download_models.parse_custom_models('["en"]')
        self.assertIsNotNone(error)

    def test_invalid_entry_skipped(self):
        mlm, spacy, error = download_models.parse_custom_models(
            '{"sv": "not-a-dict"}'
        )
        self.assertIsNone(error)
        self.assertEqual(mlm, {})
        self.assertEqual(spacy, {})

    def test_blank_values_ignored(self):
        mlm, spacy, error = download_models.parse_custom_models(
            '{"sv": {"model": "   ", "spacy": ""}}'
        )
        self.assertIsNone(error)
        self.assertEqual(mlm, {})
        self.assertEqual(spacy, {})

    def test_custom_entry_without_spacy_rejected(self):
        # A custom entry with only a model (no spaCy pipeline) must be skipped:
        # it would provision but fail at runtime.
        mlm, spacy, error = download_models.parse_custom_models(
            '{"sv": {"model": "KB/bert-base-swedish-cased"}}'
        )
        self.assertIsNone(error)
        self.assertEqual(mlm, {})
        self.assertEqual(spacy, {})


class ExitCodeTests(unittest.TestCase):
    def test_success_returns_zero(self):
        env = {"LANGUAGES": "en"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(
                download_models, "download_spacy_model", return_value=True
            ), mock.patch.object(
                download_models, "download_mlm_model", return_value=True
            ), mock.patch.object(
                download_models, "download_fasttext_model", return_value=None
            ):
                self.assertEqual(download_models.main(), 0)

    def test_failed_mlm_download_exits_nonzero(self):
        env = {"LANGUAGES": "en"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(
                download_models, "download_spacy_model", return_value=True
            ), mock.patch.object(
                download_models, "download_mlm_model", return_value=False
            ), mock.patch.object(
                download_models, "download_fasttext_model", return_value=None
            ):
                self.assertEqual(download_models.main(), 1)

    def test_failed_spacy_download_exits_nonzero(self):
        env = {"LANGUAGES": "fr"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(
                download_models, "download_spacy_model", return_value=False
            ), mock.patch.object(
                download_models, "download_mlm_model", return_value=True
            ), mock.patch.object(
                download_models, "download_fasttext_model", return_value=None
            ):
                self.assertEqual(download_models.main(), 1)

    def test_fasttext_failure_is_non_fatal(self):
        with mock.patch.object(
            download_models, "download_to", side_effect=OSError("network down")
        ):
            # Must not raise: fastText failures are swallowed internally.
            download_models.download_fasttext_model()

    def test_invalid_custom_json_exits_nonzero(self):
        env = {"LANGUAGES": "", "CUSTOM_MLM_MODELS": "{bad"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(
                download_models, "download_fasttext_model", return_value=None
            ):
                self.assertEqual(download_models.main(), 1)

    def test_language_without_spacy_exits_nonzero(self):
        # "tr" has an MLM model but no spaCy pipeline; provisioning must fail
        # rather than bake an image that cannot run Turkish.
        env = {"LANGUAGES": "tr"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(
                download_models, "download_fasttext_model", return_value=None
            ):
                self.assertEqual(download_models.main(), 1)


class DownloadToTests(unittest.TestCase):
    def test_failed_download_cleans_up_temp_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "model.bin")
            with mock.patch.object(
                download_models.urllib.request,
                "urlopen",
                side_effect=OSError("boom"),
            ):
                with self.assertRaises(OSError):
                    download_models.download_to("http://example.invalid/x", dest)
            leftovers = [f for f in os.listdir(tmpdir) if f.endswith(".tmp")]
            self.assertEqual(leftovers, [])
            self.assertFalse(os.path.exists(dest))


class FastTextValidationTests(unittest.TestCase):
    def test_validate_load_attempt_rejects_bad_file(self):
        fake_fasttext = mock.Mock()
        fake_fasttext.load_model.side_effect = ValueError("bad model file")
        with mock.patch.dict(sys.modules, {"fasttext": fake_fasttext}):
            with tempfile.NamedTemporaryFile() as f:
                self.assertFalse(download_models.validate_fasttext_model(f.name))

    def test_validate_load_attempt_accepts_good_file(self):
        fake_fasttext = mock.Mock()
        fake_fasttext.load_model.return_value = object()
        with mock.patch.dict(sys.modules, {"fasttext": fake_fasttext}):
            with tempfile.NamedTemporaryFile() as f:
                self.assertTrue(download_models.validate_fasttext_model(f.name))

    def test_existing_invalid_model_is_discarded_and_redownloaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = os.path.join(tmpdir, "lid.176.bin")
            with open(bad_path, "wb") as f:
                f.write(b"not a real fasttext model")
            with mock.patch.object(download_models, "FASTTEXT_PATH", bad_path):
                with mock.patch.object(
                    download_models,
                    "validate_fasttext_model",
                    side_effect=[False, True],
                ):
                    with mock.patch.object(download_models, "download_to") as dl:
                        download_models.download_fasttext_model()
            dl.assert_called_once()

    def test_downloaded_bad_model_is_retried_then_gives_up(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "lid.176.bin")
            with mock.patch.object(download_models, "FASTTEXT_PATH", dest):
                with mock.patch.object(
                    download_models, "validate_fasttext_model", return_value=False
                ):
                    with mock.patch.object(download_models, "download_to") as dl:
                        download_models.download_fasttext_model()
            self.assertEqual(dl.call_count, download_models.FASTTEXT_MAX_ATTEMPTS)
            self.assertFalse(os.path.exists(dest))


if __name__ == "__main__":
    unittest.main()
