#!/bin/bash
set -e

# Download models on first start if they are not already present.
# Models are cached in mounted volumes so subsequent starts are fast.
# Set DOWNLOAD_MODELS_ON_STARTUP=0 to skip this check entirely.
if [ "${DOWNLOAD_MODELS_ON_STARTUP:-1}" != "0" ]; then
    echo "Checking models..." >&2
    python download_models.py
fi

# Honor PORT for the default gunicorn command (default 3000). If the command is
# overridden to something other than gunicorn, leave it untouched.
if [ "${1:-}" = "gunicorn" ]; then
    exec gunicorn --bind "0.0.0.0:${PORT:-3000}" "${@:2}"
fi

exec "$@"
