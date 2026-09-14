import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_primary_table_matches_analysis() -> None:
    manuscript = (ROOT / "paper" / "manuscript.md").read_text(encoding="utf-8")
    analysis = json.loads(
        (ROOT / "artifacts" / "primary" / "primary-analysis.json").read_text(encoding="utf-8")
    )
    counts = {
        "qwen2.5-coder-0.5b": (39, 33),
        "starcoder2-3b": (3, 3),
        "deepseek-coder-1.3b": (56, 47),
        "qwen2.5-coder-1.5b": (64, 56),
        "qwen2.5-coder-3b": (86, 70),
    }

    for model, (humaneval_count, plus_count) in counts.items():
        rates = analysis["local_rates"][model]
        humaneval = f"{100 * rates['humaneval']:.1f}% ({humaneval_count}/164)"
        humaneval_plus = f"{100 * rates['humaneval_plus']:.1f}% ({plus_count}/164)"
        assert humaneval in manuscript
        assert humaneval_plus in manuscript

    assert "Kendall's tau-b was 0.8" in manuscript
    assert "[-28.66, -15.85]" in manuscript
    assert "failed to reproduce" in manuscript.lower()


def test_manuscript_keeps_release_boundaries_explicit() -> None:
    manuscript = (ROOT / "paper" / "manuscript.md").read_text(encoding="utf-8").lower()
    assert "has not been run and is not reported here" in manuscript
    assert "do not show that the published score is erroneous" in manuscript
    assert "no doi or third-party archival identifier is claimed" in manuscript


def test_rendered_pdf_is_checked_in() -> None:
    pdf = (
        ROOT
        / "paper"
        / "output"
        / "pdf"
        / "Do_Published_HumanEval_Rankings_Survive_Local_Deployment.pdf"
    )
    assert pdf.read_bytes().startswith(b"%PDF-")
