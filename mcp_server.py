#!/usr/bin/env python3
"""
Caveman Compression MCP Server
Provides MLM-based text compression via MCP protocol.
"""

import json
import sys
import os

# Add the caveman-compression directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caveman_compress_mlm import compress_text, detect_language, SUPPORTED_LANGUAGES
from caveman_compress_nlp import compress_text as compress_text_nlp

# MCP Protocol Constants
MCP_VERSION = "2024-11-05"

# Compression presets: name -> drop_ratio
COMPRESSION_PRESETS = {
    "lite": 0.3,        # Drop 30% most predictable words
    "full": 0.5,        # Drop 50% most predictable words
    "ultra": 0.7,       # Drop 70% most predictable words
}

# Languages with MLM models available (derived from SUPPORTED_LANGUAGES)
MLM_LANGUAGES = set(SUPPORTED_LANGUAGES.keys())

def handle_request(request):
    """Handle MCP request"""
    method = request.get("method")
    params = request.get("params", {})
    
    if method == "initialize":
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "caveman-compression",
                "version": "1.0.0"
            }
        }
    
    elif method == "tools/list":
        return {
            "tools": [
                {
                    "name": "caveman_compress",
                    "description": "Compress text using Caveman compression (MLM for English/French, NLP for other languages)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to compress"
                            },
                            "language": {
                                "type": "string",
                                "description": "Language code (en, fr, es, de, etc.). Auto-detect if not specified."
                            },
                            "method": {
                                "type": "string",
                                "description": "Compression method: auto, mlm, nlp",
                                "default": "auto"
                            },
                            "preset": {
                                "type": "string",
                                "description": "Compression preset: lite, full, ultra",
                                "default": "full",
                                "enum": ["lite", "full", "ultra"]
                            },
                            "drop_ratio": {
                                "type": "number",
                                "description": "Custom drop ratio (overrides preset). Range: 0.0 to 1.0. E.g., 0.3 = drop 30% most predictable words."
                            }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "caveman_stats",
                    "description": "Get compression statistics for text",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to analyze"
                            }
                        },
                        "required": ["text"]
                    }
                }
            ]
        }
    
    elif method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if name == "caveman_compress":
            text = arguments.get("text")
            language = arguments.get("language")
            method = arguments.get("method", "auto")
            preset = arguments.get("preset", "full")
            drop_ratio = arguments.get("drop_ratio")
            
            if not text:
                return {"error": {"code": -32602, "message": "text is required"}}
            
            # Get drop_ratio from preset or use custom value
            if drop_ratio is not None:
                drop_ratio = max(0.0, min(1.0, float(drop_ratio)))
            else:
                drop_ratio = COMPRESSION_PRESETS.get(preset, COMPRESSION_PRESETS["full"])
            
            try:
                lang = language or detect_language(text)
                
                if method == "nlp":
                    compressed = compress_text_nlp(text, lang=lang)
                    model = f"caveman-nlp-{lang}"
                elif method == "mlm":
                    # Force MLM with NLP fallback
                    try:
                        compressed = compress_text(text, language=lang, drop_ratio=drop_ratio)
                        model = f"caveman-mlm-{lang}"
                    except Exception:
                        compressed = compress_text_nlp(text, lang=lang)
                        model = f"caveman-nlp-{lang}"
                else:
                    # Auto: MLM for supported langs, NLP for others
                    if lang in MLM_LANGUAGES:
                        try:
                            compressed = compress_text(text, language=lang, drop_ratio=drop_ratio)
                            model = f"caveman-mlm-{lang}"
                        except Exception:
                            compressed = compress_text_nlp(text, lang=lang)
                            model = f"caveman-nlp-{lang}"
                    else:
                        compressed = compress_text_nlp(text, lang=lang)
                        model = f"caveman-nlp-{lang}"
                
                return {
                    "content": [{"type": "text", "text": compressed}],
                    "metadata": {
                        "language": lang,
                        "model": model,
                        "preset": preset,
                        "drop_ratio": drop_ratio,
                        "original_size": len(text),
                        "compressed_size": len(compressed),
                        "compression_ratio": len(compressed) / len(text) if text else 0
                    }
                }
            except Exception as e:
                return {"error": {"code": -32000, "message": str(e)}}
        
        elif name == "caveman_stats":
            text = arguments.get("text")
            if not text:
                return {"error": {"code": -32602, "message": "text is required"}}
            
            try:
                lang = detect_language(text)
                orig_tokens = len(text.strip()) // 4
                
                return {
                    "content": [{"type": "text", "text": f"Language: {lang}, Tokens: ~{orig_tokens}"}],
                    "metadata": {
                        "language": lang,
                        "original_size": len(text),
                        "estimated_tokens": orig_tokens
                    }
                }
            except Exception as e:
                return {"error": {"code": -32000, "message": str(e)}}
        
        else:
            return {"error": {"code": -32601, "message": f"Unknown tool: {name}"}}
    
    elif method == "notifications/initialized":
        # Client initialized, no response needed
        return None
    
    else:
        return {"error": {"code": -32601, "message": f"Unknown method: {method}"}}

def main():
    """Main MCP server loop"""
    # Pre-load models
    try:
        from caveman_compress_mlm import get_mlm_model
        # Pre-load English and French models
        get_mlm_model("en")
        get_mlm_model("fr")
        print("MLM models loaded (en, fr).", file=sys.stderr)
    except Exception as e:
        print(f"Warning: MLM models failed to load: {e}", file=sys.stderr)
    
    print("Caveman Compression MCP Server ready", file=sys.stderr)
    
    # Read from stdin, write to stdout
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = handle_request(request)
            
            # Add JSON-RPC fields
            response["jsonrpc"] = "2.0"
            if "id" in request:
                response["id"] = request["id"]
            
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            print(json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error"},
                "id": None
            }), flush=True)
        except Exception as e:
            print(json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)},
                "id": request.get("id") if "request" in dir() else None
            }), flush=True)

if __name__ == "__main__":
    main()
