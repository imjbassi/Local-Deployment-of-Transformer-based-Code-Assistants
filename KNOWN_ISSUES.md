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

## Host constraints

The Windows system drive had approximately 13 GB free during setup. Primary
weights and the WSL environment are therefore kept on D: through `HF_HOME` and
an explicitly located virtual environment. The run script refuses caches with
less than 30 GB available.

Docker Desktop's service socket is present, but no Docker client is installed in
the active Windows or Ubuntu PATH. The evaluator script is syntax-checked but
cannot be called complete until its digest-pinned container runs successfully.
