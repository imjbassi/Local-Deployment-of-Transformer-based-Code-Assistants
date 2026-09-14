# Do Published HumanEval Rankings Survive Local Deployment?

This repository tests whether a published ordering of small, open code models is
preserved on a single consumer GPU. The primary target is Table 5 of the
[Qwen2.5-Coder technical report](https://arxiv.org/abs/2409.12186), not the
unsupported values in this repository's historical manuscript.

## Evidence status

The preregistered primary run is complete for all five models and 164 tasks. The
published ordering failed to reproduce (Kendall's tau-b = 0.8) because
StarCoder2-3B reversed its ordering with Qwen2.5-Coder-0.5B. See
[RESULTS.md](RESULTS.md) for the measured scores, decision rule, interpretation
boundary, and remaining release gates. The exact published targets remain
machine-readable in [protocol/published_targets.json](protocol/published_targets.json).

The evidence-backed technical report, its canonical Markdown source, build
script, and internal claim review are in [paper](paper). The rendered PDF is
[Do Published HumanEval Rankings Survive Local Deployment?](paper/output/pdf/Do_Published_HumanEval_Rankings_Survive_Local_Deployment.pdf).

## Primary comparison

The five exact base checkpoints reported together in Qwen2.5-Coder Table 5 are:

| Checkpoint | Published HumanEval | Published HumanEval+ |
|---|---:|---:|
| Qwen2.5-Coder-0.5B | 28.0 | 23.8 |
| StarCoder2-3B | 31.7 | 27.4 |
| DeepSeek-Coder-1.3B | 34.8 | 26.8 |
| Qwen2.5-Coder-1.5B | 43.9 | 36.6 |
| Qwen2.5-Coder-3B | 52.4 | 42.7 |

The primary run uses greedy decoding, one completion per task, the base-model
prompt, and full-precision weights. EvalPlus scores the same completions against
HumanEval and HumanEval+. Sampling, prompts, and quantization are secondary
sensitivity analyses and cannot replace the primary endpoint.

## Install

Use Python 3.10-3.13 in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,evaluation]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Install the
appropriate CUDA-enabled PyTorch build first for NVIDIA hardware.

The verified WSL/CUDA environment is constrained in
[`protocol/constraints-wsl-cu128.txt`](protocol/constraints-wsl-cu128.txt).
Pass it with pip's `--constraint` option and use the CUDA 12.8 PyTorch index when
reconstructing the primary environment. Runtime `pip freeze`, GPU details, Git
state, and model revisions are captured again by the run script.

## Safe smoke test

Generation-only mode does not execute model output:

```bash
code-model-benchmark \
  --models qwen2.5-coder-0.5b \
  --limit 2 \
  --samples-per-task 1 \
  --max-new-tokens 64 \
  --decoding greedy \
  --output-dir results/smoke
code-model-validate results/smoke
```

`--limit` is only for pipeline checks and never produces a reportable
HumanEval score.

## Instrumented diagnostic runner

The package runner records detailed generation timing and environment artifacts.
It is useful for smoke tests and the separate performance study:

```bash
code-model-benchmark \
  --mode generation \
  --models qwen2.5-coder-0.5b \
  --samples-per-task 1 \
  --max-new-tokens 512 \
  --decoding greedy \
  --device cuda \
  --dtype bfloat16 \
  --output-dir results/primary/qwen2.5-coder-0.5b
```

This command is not the source of the paper's correctness scores. The primary
correctness workflow uses EvalPlus for prompt construction, generation
sanitization, and both HumanEval test suites so it stays aligned with the
published comparison.

## Primary correctness workflow

EvalPlus also warns that base checkpoints with chat templates must use a forced
base prompt. The primary study therefore records this explicitly rather than
relying on automatic prompt detection.

The pinned reference workflow is scripted:

```bash
export HF_HOME=/mnt/d/model-cache/huggingface
bash scripts/run_primary_codegen.sh
bash scripts/evaluate_in_docker.sh \
  results/evalplus/humaneval/qwen2.5-coder-0.5b.jsonl
```

The Docker script downloads the pinned public test data in a preparation step,
then disables networking and drops Linux capabilities for the untrusted-code
phase. When invoking a Windows Docker Desktop client from WSL, set
`DOCKER_DESKTOP_WINDOWS_PATHS=1` so bind sources are translated with `wslpath`.
The released primary artifacts were evaluated through this hardened path.

## Outputs

Every run directory contains:

| File | Purpose |
|---|---|
| `run_config.json` | Complete configuration and pinned dataset revision |
| `environment.json` | Git state, device, runtime, and dependency versions |
| `samples.jsonl` | Completion, task, sample, seed, and token count |
| `timings.jsonl` | Every timed generation call |
| `metrics.json` | Aggregate timing, memory, and confidence intervals |

The validator checks model identity, counts, duplicate records, and required
files. Correctness outcomes come only from EvalPlus and are checked by the
consolidation and analysis commands.

After all five EvalPlus result files exist, convert them to the documented
task-level outcome JSONL with `code-model-consolidate`, then run
`code-model-analyze --output results/primary-analysis.json`. The analysis command
refuses missing, duplicated, non-Boolean, or mismatched task records.

The checked-in primary artifacts are in [artifacts/primary](artifacts/primary).
They include untrusted model-generated Python; inspect them as data and execute
them only inside an appropriate sandbox.

## Development checks

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m build
```

CI performs linting, unit tests, and package builds without downloading weights
or executing generated code. See [CONTRIBUTING.md](CONTRIBUTING.md),
[SECURITY.md](SECURITY.md), and [PUBLICATION_STATUS.md](PUBLICATION_STATUS.md)
before opening a release or paper submission. Current execution caveats are in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Historical material

The unsupported original PDF has been removed from the release tree; Git history
retains the prior file. Its results are not incorporated into the new study. A
short internal provenance statement is in
[notes/PROVENANCE.md](notes/PROVENANCE.md); there is deliberately no publishable
audit manuscript.

## License and citation

Software and documentation are MIT licensed. Models, HumanEval, and EvalPlus
retain their upstream licenses. Citation metadata in [CITATION.cff](CITATION.cff)
describes the software protocol only; update it only after validated result
artifacts and a public archive exist.
