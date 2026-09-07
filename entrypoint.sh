#!/bin/bash
set -e

# Download models for custom languages if LANGUAGES is set
if [ -n "$LANGUAGES" ]; then
    echo "Downloading models for languages: $LANGUAGES"
    python download_models.py
fi

# Start the server
exec "$@"
