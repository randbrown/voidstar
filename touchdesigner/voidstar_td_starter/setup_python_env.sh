#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv_voidstar_td"
REQ_FILE="${ROOT_DIR}/touchdesigner/voidstar_td_starter/requirements_voidstar_td.txt"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[voidstar-td] creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

echo "[voidstar-td] upgrading pip/setuptools/wheel"
"${VENV_DIR}/bin/python" -m pip install --upgrade --no-input pip setuptools wheel

echo "[voidstar-td] installing requirements from ${REQ_FILE}"
"${VENV_DIR}/bin/python" -m pip install --no-input -r "${REQ_FILE}"

echo "[voidstar-td] done"
echo "[voidstar-td] TD External Python path: ${VENV_DIR}/bin/python"
echo "[voidstar-td] Run bridge with: ${VENV_DIR}/bin/python ${ROOT_DIR}/touchdesigner/voidstar_td_starter/voidstar_post_bridge.py --help"
