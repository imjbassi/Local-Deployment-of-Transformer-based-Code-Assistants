# Paper source and build

The checked-in PDF reports the completed five-model primary experiment. It is a
technical report, not a peer-reviewed publication, and it does not present the
unrun stochastic sensitivity condition as a result.

## Contents

- `manuscript.md`: canonical human-readable paper source.
- `build_paper.py`: deterministic ReportLab builder.
- `REVIEW.md`: claim-to-evidence and presentation review.
- `output/pdf/Do_Published_HumanEval_Rankings_Survive_Local_Deployment.pdf`:
  rendered report.

## Build

From the repository root, with Python and ReportLab available:

```bash
python paper/build_paper.py
```

For visual review, render every page with Poppler:

```bash
mkdir -p tmp/pdfs
pdftoppm -png \
  paper/output/pdf/Do_Published_HumanEval_Rankings_Survive_Local_Deployment.pdf \
  tmp/pdfs/paper
```

The builder contains no benchmark measurements. Values in the manuscript must
remain traceable to `artifacts/primary/primary-analysis.json`,
`artifacts/primary/outcomes.jsonl`, or the cited source table.
