import json
from pathlib import Path

from local_code_benchmark.validate import validate_run


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_validate_run_accepts_complete_archive(tmp_path: Path) -> None:
    write_json(
        tmp_path / "run_config.json",
        {
            "models": ["codet5-small"],
            "mode": "accuracy",
            "evaluate": True,
            "timing_repetitions": 3,
        },
    )
    write_json(tmp_path / "environment.json", {"python": "3.12", "packages": {"torch": "2"}})
    write_json(
        tmp_path / "metrics.json",
        {
            "models": [
                {
                    "model": "CodeT5-small",
                    "tasks": 1,
                    "samples_per_task": 1,
                    "pass_at_1": 1.0,
                }
            ]
        },
    )
    write_json(
        tmp_path / "samples.jsonl",
        {"model": "CodeT5-small", "task_id": "HumanEval/0", "sample_index": 0},
    )
    write_json(tmp_path / "timings.jsonl", {"model": "CodeT5-small", "task_id": "HumanEval/0"})

    assert validate_run(tmp_path)["valid"] is True


def test_validate_run_reports_missing_files(tmp_path: Path) -> None:
    report = validate_run(tmp_path)
    assert report["valid"] is False
    assert "missing file: metrics.json" in report["errors"]
