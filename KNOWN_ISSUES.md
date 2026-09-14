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

## Host constraints

The Windows system drive had approximately 13 GB free during setup. Primary
weights and the WSL environment are therefore kept on D: through `HF_HOME` and
an explicitly located virtual environment. The run script refuses caches with
less than 30 GB available.

Docker Desktop's service socket is present, but no Docker client is installed in
the active Windows or Ubuntu PATH; the Windows client is available at its Docker
Desktop installation path. The official image tagged v0.3.1 reports package
version `0.4.0.dev2`, so the evaluator Dockerfile derives from its immutable
digest and force-installs the released `evalplus==0.3.1`. The build asserts that
version and evaluation records the derived image ID.
