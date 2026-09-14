"""Backward-compatible entry point; prefer ``code-model-benchmark``."""

from local_code_benchmark.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
