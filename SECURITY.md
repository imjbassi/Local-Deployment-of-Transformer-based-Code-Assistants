# Security policy

## Generated-code execution

HumanEval scoring executes model-generated Python. The upstream harness limits
execution time and applies guardrails, but it is not a security sandbox. Never
run `--evaluate` on a workstation containing credentials or sensitive data.
Use a disposable, network-isolated environment with a read-only root filesystem,
strict CPU and memory limits, and only a dedicated writable result-exchange
directory containing no credentials or unrelated files. Treat that directory as
potentially corrupted after evaluation. The package benchmark never executes
generated code; the separate Docker script is the only supported correctness
path.

## Reporting vulnerabilities

Please report security issues privately through GitHub's security advisory
workflow rather than opening a public issue.
