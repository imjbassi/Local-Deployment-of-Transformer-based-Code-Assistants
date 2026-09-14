#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 PATH_TO_SANITIZED_SAMPLES.jsonl" >&2
  exit 2
fi

samples="$(realpath "$1")"
results_root="$(dirname "$samples")"
sample_name="$(basename "$samples")"
base_image="ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740"
image="local-code-study/evalplus:0.3.1"
cache="$results_root/evalplus-cache"
mount_cache="$cache"
mount_results_root="$results_root"

if [[ "${DOCKER_DESKTOP_WINDOWS_PATHS:-0}" == "1" ]]; then
  command -v wslpath >/dev/null || {
    echo "DOCKER_DESKTOP_WINDOWS_PATHS=1 requires WSL's wslpath" >&2
    exit 2
  }
  mount_cache="$(wslpath -w "$cache")"
  mount_results_root="$(wslpath -w "$results_root")"
fi

docker pull "$base_image"
docker build --pull=false -t "$image" -f containers/evalplus/Dockerfile .
docker image inspect "$image" --format '{{.Id}}' > "$results_root/evaluator-image.txt"
mkdir -p "$cache"

# Fetch the pinned public test data before the untrusted-code phase loses network access.
docker run --rm \
  --mount "type=bind,src=$mount_cache,dst=/cache" \
  --env XDG_CACHE_HOME=/cache \
  "$image" \
  python -c "from evalplus.data import get_human_eval_plus; get_human_eval_plus(version='v0.1.10')"

docker run --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 512 \
  --memory 8g \
  --cpus 8 \
  --tmpfs /tmp:rw,noexec,nosuid,size=2g \
  --mount "type=bind,src=$mount_cache,dst=/cache" \
  --env XDG_CACHE_HOME=/cache \
  --mount "type=bind,src=$mount_results_root,dst=/results" \
  "$image" \
  evalplus.evaluate \
    --dataset humaneval \
    --version v0.1.10 \
    --samples "/results/$sample_name"
