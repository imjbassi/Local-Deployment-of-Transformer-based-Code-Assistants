"""Validate the structural completeness of a benchmark run archive."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .benchmark import MODELS

REQUIRED_FILES = (
    "run_config.json",
    "environment.json",
    "samples.jsonl",
    "timings.jsonl",
    "metrics.json",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_number} is not valid JSON") from exc
    return records


def validate_run(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        return {"valid": False, "errors": [f"missing file: {name}" for name in missing]}

    config = read_json(path / "run_config.json")
    environment = read_json(path / "environment.json")
    metrics = read_json(path / "metrics.json").get("models", [])
    samples = read_jsonl(path / "samples.jsonl")
    timings = read_jsonl(path / "timings.jsonl")

    configured_models = list(config.get("models", []))
    if len(metrics) != len(configured_models):
        errors.append("metrics model count does not match run configuration")
    expected_names = {MODELS[key].name for key in configured_models if key in MODELS}
    metric_names = {item.get("model") for item in metrics}
    if expected_names != metric_names:
        errors.append("metric model identities do not match run configuration")
    if not environment.get("python") or not environment.get("packages"):
        errors.append("environment metadata is incomplete")

    sample_counts = Counter(item.get("model") for item in samples)
    timing_counts = Counter(item.get("model") for item in timings)
    sample_keys = [
        (item.get("model"), item.get("task_id"), item.get("sample_index")) for item in samples
    ]
    if len(sample_keys) != len(set(sample_keys)):
        errors.append("duplicate model/task/sample records found")

    for metric in metrics:
        name = metric.get("model")
        tasks = metric.get("tasks")
        samples_per_task = metric.get("samples_per_task")
        if not isinstance(tasks, int) or not isinstance(samples_per_task, int):
            errors.append(f"{name}: invalid task or sample count")
            continue
        expected_samples = tasks * samples_per_task
        if sample_counts[name] != expected_samples:
            errors.append(
                f"{name}: expected {expected_samples} sample records, found {sample_counts[name]}"
            )
        repetitions = (
            config.get("timing_repetitions", 1)
            if config.get("mode") == "performance"
            else 1
        )
        expected_timings = tasks * repetitions
        if timing_counts[name] != expected_timings:
            errors.append(
                f"{name}: expected {expected_timings} timing records, found {timing_counts[name]}"
            )
        if config.get("evaluate") and metric.get("pass_at_1") is None:
            errors.append(f"{name}: evaluated run is missing pass@1")

    if not metrics and configured_models:
        errors.append("no model metrics found")
    return {
        "valid": not errors,
        "errors": errors,
        "models": len(metrics),
        "sample_records": len(samples),
        "timing_records": len(timings),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a benchmark run directory.")
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args(argv)
    report = validate_run(args.run_directory)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
