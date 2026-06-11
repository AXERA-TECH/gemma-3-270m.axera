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
CHIP="${CHIP:-AX650}"
BUILD_CHIP="${CHIP}"
if [[ "${CHIP}" == "AX630C" ]]; then
  BUILD_CHIP="AX620E"
fi
PACKAGE_CHIP="${PACKAGE_CHIP:-${CHIP}}"
PARALLEL="${PARALLEL:-8}"
HIDDEN_STATE_TYPE="${HIDDEN_STATE_TYPE:-bf16}"
WEIGHT_TYPE="${WEIGHT_TYPE:-s8}"
POST_WEIGHT_TYPE="${POST_WEIGHT_TYPE:-s8}"
CHECK_LEVEL="${CHECK_LEVEL:-0}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/axera-matplotlib}"
export MPLCONFIGDIR

CTX_NAME=$((MAX_CONTEXT - 1))
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/${MODEL_BASENAME}-${PACKAGE_CHIP}-C${PREFILL_STEP_SIZE}-P${PREFILL_LEN}-CTX${CTX_NAME}}"

mkdir -p "${MPLCONFIGDIR}"

if ! command -v pulsar2 >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  set +u
  source "${REPO_ROOT}/npu-codebase/script/npu_dev"
  set -u
fi

cmd=(
  pulsar2 llm_build2
  --input_path "${INPUT_DIR}"
  --output_path "${OUTPUT_DIR}"
  --hidden_state_type "${HIDDEN_STATE_TYPE}"
  --weight_type "${WEIGHT_TYPE}"
  --post_weight_type "${POST_WEIGHT_TYPE}"
  --prefill_len "${PREFILL_LEN}"
  --prefill_step_size "${PREFILL_STEP_SIZE}"
  --max_context "${MAX_CONTEXT}"
  --chip "${BUILD_CHIP}"
  --parallel "${PARALLEL}"
  --check_level "${CHECK_LEVEL}"
)

if [[ -n "${DECODE_STEP_SIZE:-}" ]]; then
  cmd+=(--decode_step_size "${DECODE_STEP_SIZE}")
fi

if [[ -n "${NPU_MODE:-}" ]]; then
  cmd+=(--npu_mode "${NPU_MODE}")
fi

printf 'INPUT_DIR=%s\n' "${INPUT_DIR}"
printf 'OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
printf 'CHIP=%s\n' "${CHIP}"
printf 'BUILD_CHIP=%s\n' "${BUILD_CHIP}"
printf 'NPU_MODE=%s\n' "${NPU_MODE:-default}"
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
