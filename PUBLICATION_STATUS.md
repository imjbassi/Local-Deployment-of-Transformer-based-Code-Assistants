# Publication status

## Current state

The five-model primary experiment and preregistered analysis are complete. The
primary decision is `failed_to_reproduce`; measured values and interpretation
limits are in `RESULTS.md`, with auditable artifacts in `artifacts/primary`.

This is not yet a completed archival paper. The historical PDF is not evidence
for the new study and the removed self-audit manuscript must not be restored as
the contribution.

## Release gate

A paper or archival release requires all of the following:

1. **Complete:** greedy runs for all five primary checkpoints and all 164 tasks.
2. **Complete:** functional evaluation with pinned EvalPlus 0.3.1 inside a
   disposable, network-isolated container.
3. **Complete for the primary run:** model revisions, environment metadata,
   raw/sanitized completions, evaluator outputs, and checksums are checked in.
4. **Complete:** the preregistered rank endpoint, paired uncertainty, and
   three-way decision were generated from task-level artifacts.
5. **Open:** run the planned 20-sample sensitivity condition and any conditions
   it triggers under `EXPERIMENT_PLAN.md`.
6. **Open:** obtain a second-person review of protocol-to-artifact mapping and
   the eventual manuscript.
7. **Open:** deposit immutable release artifacts at a persistent public
   identifier.

Until the open gates pass, the defensible claim is limited to the completed
primary condition and its stated interpretation boundary; do not describe the
repository as a finished or peer-reviewed paper.
