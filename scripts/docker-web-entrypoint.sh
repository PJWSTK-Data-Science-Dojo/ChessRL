#!/bin/sh
set -eu

: "${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
: "${MODEL_SHA256:?MODEL_SHA256 is required}"

if [ "${#MODEL_SHA256}" -ne 64 ]; then
    echo "MODEL_SHA256 must contain exactly 64 hexadecimal characters" >&2
    exit 2
fi

case "$MODEL_SHA256" in
    *[!0-9A-Fa-f]*)
        echo "MODEL_SHA256 must contain exactly 64 hexadecimal characters" >&2
        exit 2
        ;;
esac

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint file is unavailable: $CHECKPOINT_PATH" >&2
    exit 2
fi

printf '%s  %s\n' "$MODEL_SHA256" "$CHECKPOINT_PATH" | sha256sum --check --strict
exec "$@"
