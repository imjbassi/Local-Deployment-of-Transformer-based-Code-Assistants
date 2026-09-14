# Primary results

The preregistered five-model primary run completed on 2026-09-14. Each model
generated one greedy completion for all 164 HumanEval tasks in BF16 on an RTX
4070. The same completions were scored against HumanEval and HumanEval+ with
EvalPlus 0.3.1 in a network-isolated container.

| Model | Published HE | Local HE | Published HE+ | Local HE+ |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder-0.5B | 28.0 | 23.8 (39/164) | 23.8 | 20.1 (33/164) |
| StarCoder2-3B | 31.7 | 1.8 (3/164) | 27.4 | 1.8 (3/164) |
| DeepSeek-Coder-1.3B | 34.8 | 34.1 (56/164) | 26.8 | 28.7 (47/164) |
| Qwen2.5-Coder-1.5B | 43.9 | 39.0 (64/164) | 36.6 | 34.1 (56/164) |
| Qwen2.5-Coder-3B | 52.4 | 52.4 (86/164) | 42.7 | 42.7 (70/164) |

## Primary decision

The published HumanEval order was not reproduced. Kendall's tau-b is 0.8. The
local order from low to high is StarCoder2-3B, Qwen2.5-Coder-0.5B,
DeepSeek-Coder-1.3B, Qwen2.5-Coder-1.5B, and Qwen2.5-Coder-3B.

StarCoder2-3B reverses its published ordering with Qwen2.5-Coder-0.5B. The local
paired pass@1 difference (StarCoder2 minus Qwen0.5B) is -21.95 percentage
points, with a preregistered paired-bootstrap 95% interval of [-28.66, -15.85].
Because the interval excludes zero in the reversed direction, the primary
decision is **failed to reproduce**.

## Interpretation boundary

This result applies to the uniform local deployment condition in
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md). It does not establish that the
published StarCoder2 score is wrong. The Qwen2.5-Coder report identifies the
models and benchmark but does not state every decoding, stopping, and software
detail needed to reconstruct Table 5 from the paper alone.

A direct diagnostic showed that the pinned StarCoder2 checkpoint generates
sensible code for its model-card prompt, but begins HumanEval/0 by repeating the
function definition. EvalPlus treats a new top-level `def` as the end of the
requested completion, leaving an empty function body. Most StarCoder2 failures
have this form or continue with repository-path text. This is evidence of
prompt-and-stopping sensitivity under a uniform harness, not evidence of damaged
weights.

## Artifacts and remaining work

[`artifacts/primary`](artifacts/primary) contains raw and sanitized completions,
the five full evaluator result files, 820 task-level outcomes, the analysis JSON,
environment metadata, and checksums. These files include untrusted generated
Python and must not be executed outside a sandbox.

The planned 20-sample sensitivity condition and independent review have not yet
been completed. The primary result is complete and auditable; an archival paper
release remains gated as described in
[`PUBLICATION_STATUS.md`](PUBLICATION_STATUS.md).
