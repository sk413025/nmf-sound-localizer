#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_MD="${REPO_ROOT}/paper/manuscript/manuscript.md"
DEFAULTS_YAML="${REPO_ROOT}/paper/pandoc/paper.defaults.yaml"
OUT_DIR="${REPO_ROOT}/paper/out"
OUT_PDF="${OUT_DIR}/manuscript.pdf"
LOG_FILE="${OUT_DIR}/build_pdf.log"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "ERROR: pandoc is required. Install it from https://pandoc.org and re-run." >&2
  exit 1
fi

if ! command -v xelatex >/dev/null 2>&1; then
  echo "ERROR: xelatex is required for PDF export. Install MacTeX/TeX Live and re-run." >&2
  exit 1
fi

if [[ ! -f "${INPUT_MD}" ]]; then
  echo "ERROR: missing manuscript source: ${INPUT_MD}" >&2
  exit 1
fi

if [[ ! -f "${DEFAULTS_YAML}" ]]; then
  echo "ERROR: missing pandoc defaults: ${DEFAULTS_YAML}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

log() {
  echo "$*" | tee -a "${LOG_FILE}" >&2
}

: >"${LOG_FILE}"
log "Build started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
log "Pandoc: $(pandoc --version | head -n 1)"
log "XeLaTeX: $(xelatex --version | head -n 1)"
log "Input:   ${INPUT_MD}"
log "Out:     ${OUT_PDF}"

PANDOC_ARGS=(
  --defaults "${DEFAULTS_YAML}"
  --to pdf
  --pdf-engine xelatex
  -o "${OUT_PDF}"
  "${INPUT_MD}"
)

CSL_FILE="${REPO_ROOT}/paper/csl/style.csl"
if [[ -f "${CSL_FILE}" ]]; then
  PANDOC_ARGS+=(--csl "${CSL_FILE}")
else
  log "NOTE: no CSL file at ${CSL_FILE}; using Pandoc default citation style."
fi

log "Running: pandoc --defaults ${DEFAULTS_YAML} --to pdf --pdf-engine xelatex -o ${OUT_PDF} ${INPUT_MD}"

pandoc "${PANDOC_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"

log "OK: ${OUT_PDF}"
echo "${OUT_PDF}"
