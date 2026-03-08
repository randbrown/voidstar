#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/pipeline_interactionism_voidstar_0.sh" \
  --mode preview \
  --skip-reels-overlay \
  --no-pre-reels-glitchfield \
  --skip-title-hook \
  --jobs 1 \
  "$@"
