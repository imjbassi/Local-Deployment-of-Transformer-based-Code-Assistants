# Contributing

Use Python 3.10-3.13 and create an isolated environment. Install with
`python -m pip install -e ".[dev]"`, then run `ruff check .` and `pytest` before
submitting a change.

Do not commit downloaded model weights, Hugging Face caches, generated benchmark
runs, credentials, or model-generated code. A proposed benchmark result should
include the full command, hardware description, `run_config.json`,
`environment.json`, `metrics.json`, `timings.jsonl`, and checksums or archived
copies of `samples.jsonl`. Run `code-model-validate` before proposing a result.

Generated code is untrusted. Functional evaluation must run in a disposable,
network-isolated container or virtual machine with no credentials or writable
host mounts.
