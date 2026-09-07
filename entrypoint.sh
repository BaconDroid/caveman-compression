#!/bin/bash
set -e

# The runtime image is immutable and models are provisioned at build time.
# Provision at startup only when explicitly requested, e.g. for custom models
# that were not baked into the image.
if [ "${DOWNLOAD_MODELS_ON_STARTUP:-0}" = "1" ]; then
    echo "Provisioning models at startup (DOWNLOAD_MODELS_ON_STARTUP=1)..." >&2
    # Temporarily allow network access for the download; the server itself
    # keeps running offline afterwards.
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE python download_models.py
fi

# Honor PORT for the default gunicorn command (default 3000). If the command is
# overridden to something other than gunicorn, leave it untouched.
if [ "${1:-}" = "gunicorn" ]; then
    exec gunicorn --bind "0.0.0.0:${PORT:-3000}" "${@:2}"
fi

exec "$@"
