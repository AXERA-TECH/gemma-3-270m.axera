#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_INPUT_DIR="${DEFAULT_INPUT_DIR:-${REPO_ROOT}/../models/gemma-3-270m-it}"
INPUT_DIR="${INPUT_DIR:-${DEFAULT_INPUT_DIR}}"
MODEL_BASENAME="${MODEL_BASENAME:-$(basename "${INPUT_DIR%/}")}"
PREFILL_LEN="${PREFILL_LEN:-1024}"
PREFILL_STEP_SIZE="${PREFILL_STEP_SIZE:-128}"
MAX_CONTEXT="${MAX_CONTEXT:-2048}"
CTX_NAME=$((MAX_CONTEXT - 1))
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/${MODEL_BASENAME}-AX650-C${PREFILL_STEP_SIZE}-P${PREFILL_LEN}-CTX${CTX_NAME}}"

cmd=(
  env PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
  python "${SCRIPT_DIR}/prepare_axllm_package.py"
  --input-path "${INPUT_DIR}"
  --output-path "${OUTPUT_DIR}"
  --prefill-step-size "${PREFILL_STEP_SIZE}"
  --model-name "${MODEL_NAME:-AXERA-TECH/${MODEL_BASENAME}}"
  --dynamic-pool "${DYNAMIC_POOL:-2}"
  --server-max-output-tokens "${SERVER_MAX_OUTPUT_TOKENS:-256}"
)

if [[ -n "${STRICT_AXMODELS:-}" ]]; then
  cmd+=(--strict-axmodels)
fi

if [[ -n "${DYNAMIC_LOAD:-}" ]]; then
  cmd+=(--dynamic-load)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
