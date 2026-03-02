#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/pipeline_atomism_voidstar_base.sh" --config "$SCRIPT_DIR/config/pipeline_atomism_voidstar_3.conf.sh" "$@"
