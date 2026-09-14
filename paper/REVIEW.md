# Manuscript review

Review date: 2026-09-14
Scope: internal evidence, language, citation, and rendering review of version 1.0

## Verdict

The manuscript is suitable as a transparent technical report of the completed
primary experiment. It is not labeled peer reviewed or presented as a completed
archival study. The planned stochastic sensitivity condition and an independent
second-person review remain open publication gates.

## Claim-to-evidence audit

| Claim | Evidence | Status |
|---|---|---|
| Five models completed 164 tasks | Five 164-line sample files and evaluator outputs | Verified |
| Local HumanEval and HumanEval+ counts | `outcomes.jsonl` and five evaluator result files | Verified |
| Kendall's tau-b is 0.8 | `primary-analysis.json` | Verified |
| StarCoder2/Qwen0.5 reversal and interval | `primary-analysis.json`, 10,000 paired replicates, seed 2026 | Verified |
| Checkpoint identities and revisions | `protocol/published_targets.json` | Verified |
| BF16, RTX 4070, package and evaluator versions | hardware, package-freeze, and evaluator-image artifacts | Verified |
| Published scores | Qwen2.5-Coder technical report, Table 5 | Verified against primary source |
| StarCoder2 diagnostic | Retained completion artifacts and documented direct probe | Bounded, exploratory |

## Language and inference controls

- The title and abstract specify a local deployment reproduction, not a general
  model-quality ranking.
- The result is phrased as failure to reproduce under the declared condition;
  it is not framed as proof that a published score is false.
- Exact counts accompany rounded percentages.
- The task bootstrap is not described as generation-seed uncertainty.
- No latency, throughput, CPU, pass@5, quantization, or secondary-condition
  result is claimed.
- The StarCoder2 diagnostic is explicitly post hoc and non-causal.
- The report identifies the 2024 target as historical rather than calling the
  selected models current state of the art.

## Citation review

References distinguish HumanEval, EvalPlus, Qwen2.5-Coder, StarCoder2, and
DeepSeek-Coder. All links resolve to the authors' paper pages or the study
repository. The manuscript has no uncited borrowed result beyond the explicitly
identified Table 5 target.

## Open gates before venue submission

1. Run the predeclared 20-sample sensitivity condition and any triggered seeds.
2. Add controlled prompt and stop-policy ablations for the StarCoder2 finding.
3. Obtain a second-person protocol-to-artifact and manuscript review.
4. Deposit a versioned artifact bundle and report at a persistent identifier.
5. Select a venue and adapt length, anonymization, formatting, and disclosure
   statements to its current author instructions.

These items limit a venue-submission claim, but they do not invalidate the
checked-in primary result or this explicitly scoped technical report.
