"""Focused tests for download_models.py provisioning logic (no network)."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import download_models
import language_catalog


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
            ), mock.patch.object(
                download_models, "write_language_manifest", return_value=None
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


class CatalogueProvisioningTests(unittest.TestCase):
    """The downloader provisions from the shared catalogue; de/es need no CUSTOM."""

    def test_catalogue_is_single_source(self):
        self.assertIs(
            download_models.MLM_LANGUAGE_CATALOGUE,
            language_catalog.MLM_LANGUAGE_CATALOGUE,
        )
        self.assertEqual(
            set(language_catalog.MLM_LANGUAGE_CATALOGUE), {"en", "de", "fr", "es"}
        )

    def test_default_languages_provision_only_en_fr(self):
        # With the default LANGUAGES (en,fr), de/es must not be provisioned.
        spacy_calls = []
        mlm_calls = []
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(
                download_models, "download_spacy_model",
                side_effect=lambda name: spacy_calls.append(name) or True,
            ), mock.patch.object(
                download_models, "download_mlm_model",
                side_effect=lambda name, lang: mlm_calls.append(name) or True,
            ), mock.patch.object(
                download_models, "download_fasttext_model", return_value=None,
            ), mock.patch.object(
                download_models, "write_language_manifest", return_value=None,
            ):
                self.assertEqual(download_models.main(), 0)
        self.assertEqual(set(spacy_calls), {"en_core_web_sm", "fr_core_news_sm"})
        self.assertEqual(set(mlm_calls), {"roberta-base", "camembert-base"})

    def test_languages_en_fr_de_provisions_de_without_custom(self):
        # de is in the catalogue, so LANGUAGES=en,fr,de provisions it with no
        # CUSTOM_MLM_MODELS.
        spacy_calls = []
        mlm_calls = []
        env = {"LANGUAGES": "en,fr,de"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(
                download_models, "download_spacy_model",
                side_effect=lambda name: spacy_calls.append(name) or True,
            ), mock.patch.object(
                download_models, "download_mlm_model",
                side_effect=lambda name, lang: mlm_calls.append(name) or True,
            ), mock.patch.object(
                download_models, "download_fasttext_model", return_value=None,
            ), mock.patch.object(
                download_models, "write_language_manifest", return_value=None,
            ):
                self.assertEqual(download_models.main(), 0)
        self.assertIn("de_core_news_sm", spacy_calls)
        self.assertIn("bert-base-german-cased", mlm_calls)

    def test_languages_es_provisions_es_without_custom(self):
        spacy_calls = []
        mlm_calls = []
        env = {"LANGUAGES": "es"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(
                download_models, "download_spacy_model",
                side_effect=lambda name: spacy_calls.append(name) or True,
            ), mock.patch.object(
                download_models, "download_mlm_model",
                side_effect=lambda name, lang: mlm_calls.append(name) or True,
            ), mock.patch.object(
                download_models, "download_fasttext_model", return_value=None,
            ), mock.patch.object(
                download_models, "write_language_manifest", return_value=None,
            ):
                self.assertEqual(download_models.main(), 0)
        self.assertIn("es_core_news_sm", spacy_calls)
        self.assertIn("dccuchile/bert-base-spanish-wwm-cased", mlm_calls)


class LanguageManifestTests(unittest.TestCase):
    """A successful build writes a checked-in-image manifest of active languages."""

    def _run_main(self, languages=None, custom=""):
        env = {}
        if languages is not None:
            env["LANGUAGES"] = languages
        if custom:
            env["CUSTOM_MLM_MODELS"] = custom
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(
                download_models, "download_spacy_model", return_value=True
            ), mock.patch.object(
                download_models, "download_mlm_model", return_value=True
            ), mock.patch.object(
                download_models, "download_fasttext_model", return_value=None
            ):
                self.assertEqual(download_models.main(), 0)

    def _read_manifest(self, tmpdir):
        with open(os.path.join(tmpdir, "languages.json")) as f:
            return json.load(f)

    def test_manifest_records_default_en_fr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "languages.json")
            with mock.patch.object(download_models, "MANIFEST_PATH", manifest_path):
                self._run_main()
            manifest = self._read_manifest(tmpdir)
        self.assertEqual(set(manifest), {"en", "fr"})
        self.assertEqual(manifest["en"], {"model": "roberta-base", "spacy": "en_core_web_sm"})
        self.assertEqual(manifest["fr"], {"model": "camembert-base", "spacy": "fr_core_news_sm"})

    def test_manifest_includes_custom_model(self):
        custom = '{"sv": {"model": "KB/bert-base-swedish-cased", "spacy": "sv_core_news_sm"}}'
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "languages.json")
            with mock.patch.object(download_models, "MANIFEST_PATH", manifest_path):
                self._run_main(languages="en,fr,sv", custom=custom)
            manifest = self._read_manifest(tmpdir)
        self.assertEqual(
            manifest["sv"],
            {"model": "KB/bert-base-swedish-cased", "spacy": "sv_core_news_sm"},
        )

    def test_manifest_excludes_unrequested_catalogue_languages(self):
        # de/es are in the catalogue but not requested; they must not appear in
        # the manifest.
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "languages.json")
            with mock.patch.object(download_models, "MANIFEST_PATH", manifest_path):
                self._run_main(languages="en")
            manifest = self._read_manifest(tmpdir)
        self.assertEqual(set(manifest), {"en"})

    def test_manifest_writes_full_catalogue_config(self):
        # LANGUAGES=en,fr,de provisions de from the catalogue and records its
        # model/spacy mapping in the manifest.
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "languages.json")
            with mock.patch.object(download_models, "MANIFEST_PATH", manifest_path):
                self._run_main(languages="en,fr,de")
            manifest = self._read_manifest(tmpdir)
        self.assertEqual(
            manifest["de"],
            {"model": "bert-base-german-cased", "spacy": "de_core_news_sm"},
        )


def _spacy_cli_available():
    try:
        import spacy.cli  # noqa: F401
        return True
    except ImportError:
        return False


HAS_SPACY_CLI = _spacy_cli_available()


@unittest.skipUnless(HAS_SPACY_CLI, "spacy.cli not installed")
class SpaCyResolverTests(unittest.TestCase):
    """spaCy pipelines are provisioned via spaCy's resolver, not raw pip."""

    def test_download_spacy_model_uses_spacy_resolver(self):
        with mock.patch("spacy.cli.download") as spacy_download:
            self.assertTrue(download_models.download_spacy_model("en_core_web_sm"))
        spacy_download.assert_called_once_with("en_core_web_sm")

    def test_download_spacy_model_resolver_failure_returns_false(self):
        with mock.patch("spacy.cli.download", side_effect=SystemExit(1)):
            self.assertFalse(download_models.download_spacy_model("en_core_web_sm"))


if __name__ == "__main__":
    unittest.main()
