# Known issues

## EvalPlus attention-mask warning

EvalPlus v0.3.1's Hugging Face provider passes token IDs but not an explicit
attention mask. Transformers 4.57.6 therefore warns when a tokenizer uses the
same ID for padding and end-of-sequence tokens. The primary condition uses batch
size one with no padded input tokens, so the warning does not identify an actual
masked token in the verified Qwen2.5-Coder-0.5B probe. It is retained in logs and
must not be suppressed.

Changing the provider would cease to be an exact run of the pinned public
evaluation implementation. If a model produces different output when an
all-ones attention mask is supplied, report that as a predeclared implementation
sensitivity analysis; do not silently replace the primary output.

## EvalPlus stop-criterion performance

The v0.3.1 stop criterion decodes the full completion separately for every stop
string after every token. The primary generation wrapper substitutes an
equivalent batch-one predicate that decodes once and tests all stop strings. A
pinned one-task slow-path output must match the optimized raw and sanitized
outputs byte-for-byte before full generation proceeds. The scientific condition
is unchanged; only redundant decoding is removed.

EvalPlus also leaves each task's wrapped Transformers stopping hook installed,
which otherwise accumulates another criterion on every task. The primary wrapper
restores the original hook in a `finally` block after every generation call.
Raw and sanitized outputs for the first two Qwen0.5B tasks were byte-identical
before and after this change; hashes are recorded in
`protocol/stop_hook_restoration_equivalence.json`.

The provider's stop-text list is derived from mutable shared state and can
contain duplicates when several models are constructed sequentially. The runner
de-duplicates identical strings while preserving first occurrence and order;
this does not change the `any(stop in decoded)` predicate.

## StarCoder2 prompt sensitivity

The pinned StarCoder2-3B checkpoint scored 3/164 in the primary condition. A
direct Transformers probe produced normal code for the model-card prompt but
repeated the complete function definition for HumanEval/0. EvalPlus stops at a
new top-level `def`, yielding an empty body. The result is retained rather than
post-processed because changing the prompt or stop policy after observing scores
would violate the primary protocol. Treat this as a deployment-condition result,
not a claim that the published score is erroneous.

## Host constraints

The Windows system drive had approximately 13 GB free during setup. Primary
weights and the WSL environment are therefore kept on D: through `HF_HOME` and
an explicitly located virtual environment. The run script refuses caches with
less than 30 GB available.

Docker Desktop's Windows client was invoked from WSL through an explicit wrapper
because it was not on the Ubuntu PATH. Set `DOCKER_DESKTOP_WINDOWS_PATHS=1` when
using that arrangement so the script translates bind sources. The official
image tagged v0.3.1 reports package version `0.4.0.dev2`, so the evaluator
Dockerfile derives from its immutable digest and force-installs released
`evalplus==0.3.1`. The build asserts that version and evaluation records the
derived image ID.
