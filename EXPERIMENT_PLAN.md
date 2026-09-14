# Experiment plan

## Question and target

**Primary question:** Does the HumanEval ordering of five small base models in
Table 5 of the Qwen2.5-Coder technical report survive local deployment on one
consumer NVIDIA GPU?

The target is external, widely inspectable, and machine-readable in
`protocol/published_targets.json`. HumanEval+ is a co-reported robustness outcome,
not a replacement for the HumanEval primary endpoint.

## Primary endpoint and decision rule

The primary estimand is Kendall's tau-b between the published HumanEval ordering
and the local greedy pass@1 ordering. Uncertainty is computed by paired bootstrap
resampling of the 164 tasks, preserving every model's outcome for a task in the
same bootstrap draw.

The result has three possible interpretations:

- **Reproduced:** point ordering is identical (tau-b = 1), with no adjacent
  published pair reversed locally.
- **Failed to reproduce:** at least one local pair is reversed and the paired
  95% bootstrap interval for that pass@1 difference excludes zero in the reversed
  direction.
- **Inconclusive:** any other result, including unresolved ties or reversals whose
  paired interval includes zero.

This rule is fixed before correctness results are inspected. Exact recovery of
each published percentage is not required for ranking reproduction; absolute
score deltas and HumanEval+ deltas are reported descriptively.

## Primary condition

- Models: the five exact checkpoint IDs and immutable Hugging Face revisions in
  `published_targets.json`.
- Dataset: all 164 HumanEval tasks, with the EvalPlus HumanEval+ tests applied to
  the same completions.
- Prompt: EvalPlus base-model HumanEval prompt; force base mode even if a
  tokenizer exposes a chat template.
- Decoding: greedy, one completion per task, maximum 512 new tokens.
- Precision: BF16 when supported without fallback; otherwise FP16, disclosed for
  every model and held constant across all five.
- Hardware: one RTX 4070 12 GB GPU; batch size one.
- Evaluator: EvalPlus v0.3.1 (`e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`)
  with HumanEval+ dataset version v0.1.10. The local evaluator image derives from
  official base digest
  `sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740`,
  force-installs released EvalPlus 0.3.1, asserts that version, and records the
  derived image ID.
- Execution: disposable, network-isolated Linux container with no credentials or
  writable host mounts other than the result exchange directory.

Greedy pass@1 is deterministic conditional on the resolved weights and software
stack, so the primary condition has no repeated random seeds. Repeating an
identical greedy run would estimate systems nondeterminism, not pass@1 sampling
variance.

## Secondary conditions and compute budget

The full 20-sample factorial is limited to three resource-matched models:
Qwen2.5-Coder-1.5B, DeepSeek-Coder-1.3B, and StarCoder2-3B. The reference
condition uses 20 samples per task, temperature 0.2, top-p 0.95, seed 11, BF16,
and the forced base prompt. This is 9,840 completions.

Seeds 23 and 37 are run only when the seed-11 analysis produces either a rank
reversal or an adjacent pair with an absolute pass@1 difference below five
percentage points. The trigger adds at most 19,680 completions. Prompt and 8-bit
ablations are run on Qwen2.5-Coder-1.5B only, first at seed 11; the other two
seeds use the same trigger. Sampling analyses report pass@1 and pass@5 with the
Chen estimator and paired task-bootstrap intervals.

Performance is a separate 30-task, batch-one study with five warm-ups and three
timed repetitions. Latency and throughput are never inferred from accuracy-run
wall time.

## Predeclared special cases

- **Vanilla CodeT5:** optional negative-control pilot only. If pass@1 is near
  zero, the interpretation is that a generic pretrained checkpoint is not a
  completion-trained comparator. It does not enter the primary ranking.
- **StarCoder-15B:** run only if an 8-bit 16-token probe fits wholly in 12 GB VRAM,
  uses no CPU/disk offload, and leaves at least 1 GB free. If the trigger fails,
  record the condition as unsupported; never substitute a published number.
- **Out of memory:** reduce no primary scientific parameter. Record the failure
  and use sequential model loading, attention implementation changes, or a
  documented common precision for every primary model.
- **Evaluation failure:** retain the raw completion and evaluator log. Exclusions
  require a benchmark/evaluator defect documented before aggregate analysis.

## Power and precision

With 164 paired binary tasks, the smallest resolvable adjacent differences are
limited by benchmark size. The two closest published HumanEval scores differ by
3.1 percentage points, so an inconclusive adjacent comparison is expected and is
not converted into a positive reproduction claim. The paired bootstrap improves
precision relative to independent intervals but cannot create information not
present in 164 tasks. HumanEval+ adds tests, not independent tasks, so it does not
increase the task-level sample size.

## Analysis order

1. Freeze commits, checkpoint revisions, environment, and raw completion hashes.
2. Evaluate all five primary runs without inspecting partial aggregate scores.
3. Validate artifacts and document failures or exclusions.
4. Compute local pass@1, paired intervals, rank ordering, and tau-b.
5. Apply the three-way primary decision rule.
6. Evaluate only triggered secondary conditions.
7. Generate manuscript tables directly from validated artifacts.
