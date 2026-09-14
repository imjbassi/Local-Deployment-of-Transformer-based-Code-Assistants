# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Evidence-backed technical report for the five-model primary experiment.
- Canonical LaTeX and BibTeX sources, Tectonic build script, rendered PDF, and
  internal claim-to-evidence review.
- Regression checks tying manuscript counts and release boundaries to the
  primary analysis artifacts.

## [1.0.0] - 2026-09-14

### Added

- Installable `local_code_benchmark` package and command-line interface.
- Pinned HumanEval dataset provenance and runtime model revision capture.
- Deterministic sampling, warm-up runs, actual token accounting, and one-model-at-a-time loading.
- Official pass@k estimator and opt-in HumanEval functional evaluation.
- Task-bootstrap confidence intervals and separate accuracy/performance modes.
- Modern Qwen2.5-Coder, DeepSeek-Coder, and Code Llama comparison conditions.
- FP16, 8-bit, 4-bit, and prompt-prefix sensitivity controls.
- JSON/JSONL run artifacts, archive validation, unit tests, linting, packaging, and CI.
- A scoped external-ranking protocol, explicit primary endpoint, trigger-based
  compute budget, EvalPlus converter, and executable analysis command.
- Citation, contribution, security, publication-status, and license documents.

### Changed

- Replaced the divergent legacy scripts with one compatibility entry point.
- Removed the unsupported historical and self-audit manuscripts from the release
  tree; neither is a result of the new study.
