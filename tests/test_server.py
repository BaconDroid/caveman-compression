"""Boundary tests for the Flask HTTP server (server.py).

The compression backend modules are stubbed only inside an isolated import
scope (see _import_target), which restores sys.modules immediately. The stubs
therefore never leak into unittest collection, so the real backend tests never
see them regardless of discovery order.
"""

import importlib
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BACKENDS = ("caveman_compress_mlm", "caveman_compress_nlp")


class _Behavior:
    mlm_error = None
    mlm_result = "mlm-compressed"
    nlp_result = "nlp-compressed"
    detected = "en"


def _fake_mlm_compress(text, language=None, preset="lite", mode="sentence"):
    if _Behavior.mlm_error is not None:
        raise _Behavior.mlm_error
    return _Behavior.mlm_result


_mlm = types.ModuleType("caveman_compress_mlm")
_mlm.SUPPORTED_LANGUAGES = {"en": {}, "fr": {}, "de": {}}
_mlm.detect_language = lambda text: _Behavior.detected
_mlm.compress_text = _fake_mlm_compress
_mlm.get_mlm_model = lambda lang: None
_mlm.InputTooLongError = type("InputTooLongError", (ValueError,), {})

_nlp = types.ModuleType("caveman_compress_nlp")
_nlp.compress_text = lambda text, lang="en": _Behavior.nlp_result


def _import_target(name):
    """Import `name` fresh under the backend stubs, then restore sys.modules.

    The stubs and the target module live only for the duration of this call;
    afterwards sys.modules is returned to its prior state so nothing leaks into
    collection or contaminates the real backend tests.
    """
    prior = {key: sys.modules.pop(key, None) for key in (*_BACKENDS, name)}
    try:
        sys.modules["caveman_compress_mlm"] = _mlm
        sys.modules["caveman_compress_nlp"] = _nlp
        return importlib.import_module(name)
    finally:
        for key, value in prior.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class ServerBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _import_target("server")

    def setUp(self):
        _Behavior.mlm_error = None
        _Behavior.detected = "en"
        self.client = self.server.app.test_client()

    def test_no_stub_leaks_into_sys_modules(self):
        # The target was imported under stubs in an isolated scope and
        # sys.modules was restored immediately, so no backend stub may remain.
        for name in _BACKENDS:
            mod = sys.modules.get(name)
            self.assertFalse(
                mod is not None and getattr(mod, "__file__", None) is None,
                f"{name} leaked a stub into sys.modules",
            )
        self.assertNotIn("server", sys.modules)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_missing_json_body(self):
        r = self.client.post("/compress", data="", content_type="application/json")
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    def test_malformed_json(self):
        r = self.client.post("/compress", data=b"{not json", content_type="application/json")
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    def test_non_object_json_list(self):
        r = self.client.post("/compress", json=[1, 2, 3])
        self.assertEqual(r.status_code, 400)

    def test_non_object_json_string(self):
        r = self.client.post("/compress", json="hello")
        self.assertEqual(r.status_code, 400)

    def test_missing_text(self):
        r = self.client.post("/compress", json={})
        self.assertEqual(r.status_code, 400)

    def test_text_wrong_type(self):
        r = self.client.post("/compress", json={"text": 123})
        self.assertEqual(r.status_code, 400)

    def test_text_empty(self):
        r = self.client.post("/compress", json={"text": "   "})
        self.assertEqual(r.status_code, 400)

    def test_text_too_long(self):
        r = self.client.post("/compress", json={"text": "a" * 4097})
        self.assertEqual(r.status_code, 413)

    def test_request_body_too_large_transport(self):
        # Oversized body is rejected at the transport boundary by Flask's
        # MAX_CONTENT_LENGTH before JSON parsing, returning stable JSON 413.
        r = self.client.post(
            "/compress",
            data=b"x" * (self.server.MAX_CONTENT_LENGTH + 1),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 413)
        self.assertIn("error", r.get_json())

    def test_invalid_preset(self):
        r = self.client.post("/compress", json={"text": "hello", "preset": "extreme"})
        self.assertEqual(r.status_code, 400)

    def test_preset_wrong_type(self):
        r = self.client.post("/compress", json={"text": "hello", "preset": 5})
        self.assertEqual(r.status_code, 400)

    def test_invalid_mode(self):
        r = self.client.post("/compress", json={"text": "hello", "mode": "paragraph"})
        self.assertEqual(r.status_code, 400)

    def test_invalid_method(self):
        r = self.client.post("/compress", json={"text": "hello", "method": "gpt"})
        self.assertEqual(r.status_code, 400)

    def test_language_wrong_type(self):
        r = self.client.post("/compress", json={"text": "hello", "language": ["en"]})
        self.assertEqual(r.status_code, 400)

    def test_success_default(self):
        r = self.client.post("/compress", json={"text": "hello world"})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["model"], "caveman-mlm-en")
        self.assertFalse(body["fallback"])

    def test_success_nlp_method(self):
        r = self.client.post("/compress", json={"text": "hello world", "method": "nlp"})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["model"], "caveman-nlp-en")
        self.assertFalse(body["fallback"])

    def test_mlm_fallback_on_oserror(self):
        _Behavior.mlm_error = OSError("model unavailable")
        r = self.client.post("/compress", json={"text": "hello world"})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["model"], "caveman-nlp-en")
        self.assertTrue(body["fallback"])

    def test_mlm_method_nlp_fallback_for_language_without_mlm(self):
        _Behavior.detected = "es"  # Spanish has no MLM model
        r = self.client.post("/compress", json={"text": "hola mundo"})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["model"], "caveman-nlp-es")
        self.assertTrue(body["fallback"])

    def test_mlm_unexpected_error_is_500(self):
        _Behavior.mlm_error = RuntimeError("boom")
        r = self.client.post("/compress", json={"text": "hello world"})
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.get_json()["error"], "Internal server error")


if __name__ == "__main__":
    unittest.main()
