# Paper source and build

The checked-in PDF reports the completed five-model primary experiment. It is a
technical report, not a peer-reviewed publication, and it does not present the
unrun stochastic sensitivity condition as a result.

## Contents

- `main.tex`: canonical LaTeX manuscript.
- `references.bib`: BibTeX bibliography.
- `build.ps1` and `build.sh`: reproducible Tectonic build entry points.
- `REVIEW.md`: claim-to-evidence and presentation review.
- `output/pdf/Do_Published_HumanEval_Rankings_Survive_Local_Deployment.pdf`:
  PDF compiled from `main.tex`.

## Build

Install [Tectonic](https://tectonic-typesetting.github.io/) and run from the
repository root:

```powershell
.\paper\build.ps1
```

On Linux or macOS:

```bash
./paper/build.sh
```

For visual review, render every page with Poppler:

```bash
mkdir -p tmp/pdfs
pdftoppm -png \
  paper/output/pdf/Do_Published_HumanEval_Rankings_Survive_Local_Deployment.pdf \
  tmp/pdfs/paper
```

The LaTeX source contains no derived benchmark logic. Values in the manuscript
must remain traceable to `artifacts/primary/primary-analysis.json`,
`artifacts/primary/outcomes.jsonl`, or the cited source table.
