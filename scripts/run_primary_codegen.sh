#!/usr/bin/env bash
set -euo pipefail

: "${HF_HOME:?Set HF_HOME to a cache volume with at least 30 GB free}"
mkdir -p "$HF_HOME"
available_kb="$(df -Pk "$HF_HOME" | awk 'NR==2 {print $4}')"
if (( available_kb < 30 * 1024 * 1024 )); then
  echo "HF_HOME requires at least 30 GB free for the primary checkpoints" >&2
  exit 1
fi

mkdir -p results/evalplus

python -m pip freeze > results/evalplus/python-freeze.txt
nvidia-smi -q > results/evalplus/nvidia-smi.txt
printf '%s\n' "$HF_HOME" > results/evalplus/huggingface-cache-path.txt
git rev-parse HEAD > results/evalplus/repository-commit.txt
git status --porcelain > results/evalplus/repository-status.txt

python scripts/primary_codegen.py

find results/evalplus -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum > results/evalplus/SHA256SUMS
