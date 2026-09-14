# Publication status

## Current state

This is an executable study protocol and benchmark implementation, not a
completed empirical paper. No model-accuracy table is publication-ready. The
historical PDF is not evidence for the new study and the removed self-audit
manuscript must not be restored as the contribution.

## Release gate

A paper or archival release requires all of the following:

1. Successful greedy runs for all five primary checkpoints and all 164 tasks.
2. Functional evaluation of the saved completions with the pinned EvalPlus
   release inside a disposable, network-isolated container.
3. Validated per-model archives with resolved model revisions, environment
   metadata, logs, checksums, and an exclusion record.
4. Analysis generated from artifacts, including the preregistered rank endpoint,
   paired task uncertainty, and the three-way decision rule.
5. Completion of only the triggered secondary conditions in the experiment plan.
6. A second-person review of the protocol-to-artifact mapping and manuscript.
7. Deposit of the immutable artifacts at a persistent public identifier.

Until every gate passes, the defensible public claim is limited to: the
repository supplies a tested runner and a preregistered protocol for an external
HumanEval ranking replication.
