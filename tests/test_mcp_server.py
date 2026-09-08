"""Protocol boundary tests for the MCP server (mcp_server.py).

The compression backend modules are stubbed only inside an isolated import
scope (see _import_target), which restores sys.modules immediately. The stubs
therefore never leak into unittest collection, so the real backend tests never
see them regardless of discovery order.
"""

import importlib
import json
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


class McpProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mcp_server = _import_target("mcp_server")

    def setUp(self):
        _Behavior.mlm_error = None
        _Behavior.detected = "en"

    def _call(self, arguments, tool="caveman_compress", request_id=3):
        req = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        return json.loads(self.mcp_server.process_line(json.dumps(req)))

    # --- process_line: transport framing ---

    def test_parse_error_invalid_json(self):
        msg = json.loads(self.mcp_server.process_line("{not json"))
        self.assertEqual(msg["error"]["code"], -32700)
        self.assertIsNone(msg["id"])

    def test_invalid_request_non_object(self):
        msg = json.loads(self.mcp_server.process_line("[1, 2, 3]"))
        self.assertEqual(msg["error"]["code"], -32600)
        self.assertIsNone(msg["id"])

    def test_invalid_request_missing_method(self):
        msg = json.loads(self.mcp_server.process_line(json.dumps({"id": 1})))
        self.assertEqual(msg["error"]["code"], -32600)
        self.assertEqual(msg["id"], 1)

    def test_notification_emits_no_response(self):
        out = self.mcp_server.process_line(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        )
        self.assertIsNone(out)

    def test_blank_line_emits_no_response(self):
        self.assertIsNone(self.mcp_server.process_line("   "))

    # --- method dispatch ---

    def test_initialize(self):
        out = self.mcp_server.process_line(
            json.dumps({"jsonrpc": "2.0", "id": 7, "method": "initialize", "params": {}})
        )
        msg = json.loads(out)
        self.assertEqual(msg["id"], 7)
        self.assertEqual(msg["jsonrpc"], "2.0")
        self.assertIn("result", msg)
        self.assertEqual(msg["result"]["protocolVersion"], "2024-11-05")

    def test_tools_list(self):
        out = self.mcp_server.process_line(
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        )
        msg = json.loads(out)
        self.assertEqual(msg["id"], 1)
        self.assertIn("result", msg)
        self.assertIn("tools", msg["result"])

    def test_unknown_method(self):
        out = self.mcp_server.process_line(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "bogus"}))
        msg = json.loads(out)
        self.assertEqual(msg["error"]["code"], -32601)

    # --- tools/call validation ---

    def test_tools_call_success(self):
        msg = self._call({"text": "hello world"})
        self.assertNotIn("error", msg)
        self.assertIn("result", msg)
        self.assertEqual(msg["result"]["content"][0]["type"], "text")
        self.assertEqual(msg["result"]["metadata"]["model"], "caveman-mlm-en")
        self.assertFalse(msg["result"]["metadata"]["fallback"])

    def test_tools_call_missing_text(self):
        self.assertEqual(self._call({})["error"]["code"], -32602)

    def test_tools_call_text_wrong_type(self):
        self.assertEqual(self._call({"text": 99})["error"]["code"], -32602)

    def test_tools_call_text_empty(self):
        self.assertEqual(self._call({"text": "  "})["error"]["code"], -32602)

    def test_tools_call_text_too_long(self):
        self.assertEqual(self._call({"text": "a" * 4097})["error"]["code"], -32602)

    def test_tools_call_invalid_preset(self):
        self.assertEqual(self._call({"text": "hi", "preset": "nope"})["error"]["code"], -32602)

    def test_tools_call_invalid_mode(self):
        self.assertEqual(self._call({"text": "hi", "mode": "nope"})["error"]["code"], -32602)

    def test_tools_call_invalid_method(self):
        self.assertEqual(self._call({"text": "hi", "method": "nope"})["error"]["code"], -32602)

    def test_tools_call_language_wrong_type(self):
        self.assertEqual(self._call({"text": "hi", "language": 5})["error"]["code"], -32602)

    def test_tools_call_arguments_not_object(self):
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "caveman_compress", "arguments": [1, 2]},
        }
        msg = json.loads(self.mcp_server.process_line(json.dumps(req)))
        self.assertEqual(msg["error"]["code"], -32602)

    def test_tools_call_unknown_tool(self):
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        }
        msg = json.loads(self.mcp_server.process_line(json.dumps(req)))
        self.assertEqual(msg["error"]["code"], -32601)

    # --- MLM -> NLP fallback ---

    def test_mlm_fallback_on_oserror(self):
        _Behavior.mlm_error = OSError("model unavailable")
        msg = self._call({"text": "hello world"})
        self.assertNotIn("error", msg)
        self.assertEqual(msg["result"]["metadata"]["model"], "caveman-nlp-en")
        self.assertTrue(msg["result"]["metadata"]["fallback"])

    def test_mlm_method_nlp_fallback_for_language_without_mlm(self):
        _Behavior.detected = "es"  # Spanish has no MLM model
        msg = self._call({"text": "hola mundo"})
        self.assertNotIn("error", msg)
        self.assertEqual(msg["result"]["metadata"]["model"], "caveman-nlp-es")
        self.assertTrue(msg["result"]["metadata"]["fallback"])

    def test_mlm_unexpected_error_is_internal(self):
        _Behavior.mlm_error = RuntimeError("boom")
        msg = self._call({"text": "hello world"})
        self.assertEqual(msg["error"]["code"], -32000)

    # --- caveman_stats ---

    def test_caveman_stats(self):
        msg = self._call({"text": "hello world this is a test"}, tool="caveman_stats")
        self.assertNotIn("error", msg)
        self.assertIn("result", msg)
        self.assertEqual(msg["result"]["metadata"]["language"], "en")

    def test_no_stub_leaks_into_sys_modules(self):
        # The target was imported under stubs in an isolated scope and
        # sys.modules was restored immediately, so no backend stub may remain.
        for name in _BACKENDS:
            mod = sys.modules.get(name)
            self.assertFalse(
                mod is not None and getattr(mod, "__file__", None) is None,
                f"{name} leaked a stub into sys.modules",
            )
        self.assertNotIn("mcp_server", sys.modules)


if __name__ == "__main__":
    unittest.main()
