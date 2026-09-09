#!/bin/bash
set -e

# The image is immutable: models are provisioned at build time, so the runtime
# is offline and never downloads models. Enforce offline mode now (only after
# build provisioning has completed) so no HuggingFace hub access happens at
# runtime. This script also honors PORT for the default gunicorn command
# (default 3000) and execs the container command.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ "${1:-}" = "gunicorn" ]; then
    exec gunicorn --bind "0.0.0.0:${PORT:-3000}" "${@:2}"
fi

exec "$@"
