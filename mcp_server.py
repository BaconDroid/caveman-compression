#!/usr/bin/env python3
"""
Caveman Compression MCP Server
Provides text compression tools via MCP protocol.
Supports MLM (RoBERTa for English, CamemBERT for French) and NLP compression.
"""

import json
import sys
import os

# Add the caveman-compression directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caveman_compress_mlm import compress_text, detect_language
from caveman_compress_nlp import compress_text as compress_text_nlp

# MCP Protocol Constants
MCP_VERSION = "2024-11-05"

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
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "caveman_compress":
            text = arguments.get("text", "")
            language = arguments.get("language")
            method = arguments.get("method", "auto")
            
            try:
                if method == "nlp":
                    # Force NLP mode
                    lang = language or detect_language(text)
                    compressed = compress_text_nlp(text, lang=lang)
                    model = f"caveman-nlp-{lang}"
                elif method == "mlm":
                    # Force MLM mode
                    compressed = compress_text(text, language=language)
                    model = f"caveman-mlm"
                else:
                    # Auto mode: MLM for en/fr/de/zh/pt/tr/it, NLP for others
                    lang = language or detect_language(text)
                    if lang in ["en", "fr", "de", "zh", "pt", "tr", "it"]:
                        compressed = compress_text(text, language=lang)
                        model = f"caveman-mlm-{lang}"
                    else:
                        compressed = compress_text_nlp(text, lang=lang)
                        model = f"caveman-nlp-{lang}"
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": compressed
                        }
                    ]
                }
            except Exception as e:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: {str(e)}"
                        }
                    ],
                    "isError": True
                }
        
        elif tool_name == "caveman_stats":
            text = arguments.get("text", "")
            original_size = len(text)
            language = detect_language(text)
            
            try:
                # Get compression for detected language
                if language in ["en", "fr"]:
                    compressed = compress_text(text, language=language)
                    model = f"caveman-mlm-{language}"
                else:
                    compressed = compress_text_nlp(text, language=language)
                    model = f"caveman-nlp-{language}"
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "original_size": original_size,
                                "compressed_size": len(compressed),
                                "language": language,
                                "model": model,
                                "ratio": len(compressed) / original_size if original_size > 0 else 0,
                                "reduction": f"{(1 - len(compressed) / original_size) * 100:.1f}%" if original_size > 0 else "0%"
                            }, indent=2)
                        }
                    ]
                }
            except Exception as e:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: {str(e)}"
                        }
                    ],
                    "isError": True
                }
    
    return {"error": {"code": -32601, "message": "Method not found"}}

def main():
    """Main MCP server loop"""
    # Load models on startup
    print("Loading models...", file=sys.stderr)
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
            response["id"] = request.get("id")
            
            print(json.dumps(response))
            sys.stdout.flush()
        except json.JSONDecodeError:
            print(json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error"},
                "id": None
            }))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)},
                "id": request.get("id") if 'request' in locals() else None
            }))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
