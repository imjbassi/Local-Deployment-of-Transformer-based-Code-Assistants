# Primary artifact bundle

This directory is the checked-in evidence bundle for the 2026-09-14 primary
run. It contains generated code and evaluator records; treat all model output as
untrusted data.

## Provenance

- Generation repository commit: `288b39f93d690817f337acd724efedd340299233`
- Generation working tree: clean, as shown by the empty
  `repository-status.txt`
- Evaluator launcher commit: `548c3c1`
- Evaluator image:
  `sha256:fad31f5a0c46760bfc17ec644ff71622953ee10638243e4ae2eceda298dc3777`
- Evaluator package: EvalPlus 0.3.1
- Dataset: HumanEval/HumanEval+ v0.1.10
- Device summary and runtime packages: `hardware.txt` and `python-freeze.txt`

The launcher commits after the generation commit only corrected Docker 29 bind
syntax and WSL-to-Windows source-path translation. They did not modify model
generation, the evaluator image, task data, or analysis code.

## Layout

- `humaneval/*.raw.jsonl`: prompt plus raw model generation
- `humaneval/*.jsonl`: EvalPlus-sanitized samples
- `humaneval/*_eval_results.json`: task-level base and plus execution results
- `outcomes.jsonl`: normalized 820-row analysis input
- `primary-analysis.json`: preregistered aggregate result
- `SHA256SUMS`: integrity manifest for the evidence and environment files

Verify the bundle from this directory with `sha256sum -c SHA256SUMS`. Never run
the generated Python directly on a host system; the recorded evaluation used the
network-isolated container described in the repository root.
