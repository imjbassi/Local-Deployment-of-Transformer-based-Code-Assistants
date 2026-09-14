#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="$paper_dir/output/pdf"
build_dir="$paper_dir/build"
final_pdf="$output_dir/Do_Published_HumanEval_Rankings_Survive_Local_Deployment.pdf"

mkdir -p "$output_dir" "$build_dir"
(
  cd "$paper_dir"
  tectonic -X compile main.tex --outdir "$build_dir" --keep-logs
)
cp "$build_dir/main.pdf" "$final_pdf"
printf '%s\n' "$final_pdf"
