"""Preregistered cross-model ranking analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .metrics import bootstrap_mean_ci


def kendall_tau_b(published: list[float], local: list[float]) -> float:
    """Compute Kendall's tau-b, including ties, without a SciPy dependency."""
    if len(published) != len(local) or len(published) < 2:
        raise ValueError("rank vectors must have the same length and at least two values")
    concordant = discordant = ties_published = ties_local = 0
    for left in range(len(published)):
        for right in range(left + 1, len(published)):
            p_delta = published[left] - published[right]
            l_delta = local[left] - local[right]
            if p_delta == 0 and l_delta == 0:
                continue
            if p_delta == 0:
                ties_published += 1
            elif l_delta == 0:
                ties_local += 1
            elif p_delta * l_delta > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_published)
        * (concordant + discordant + ties_local)
    )
    return (concordant - discordant) / denominator if denominator else 0.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
    return records


def analyze(
    targets: dict[str, Any],
    outcomes: list[dict[str, Any]],
    *,
    expected_tasks: int = 164,
    bootstrap_replicates: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Apply the preregistered ranking endpoint to task-level outcomes."""
    models = sorted(targets["models"], key=lambda item: item["humaneval"])
    keys = [item["key"] for item in models]
    by_model: dict[str, dict[str, dict[str, Any]]] = {key: {} for key in keys}
    for record in outcomes:
        key = record.get("model_key")
        task_id = record.get("task_id")
        if key not in by_model:
            raise ValueError(f"unexpected model_key: {key}")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("every outcome requires a non-empty task_id")
        if task_id in by_model[key]:
            raise ValueError(f"duplicate outcome for {key}/{task_id}")
        for field in ("humaneval_passed", "humaneval_plus_passed"):
            if not isinstance(record.get(field), bool):
                raise ValueError(f"{key}/{task_id} requires Boolean {field}")
        by_model[key][task_id] = record

    task_sets = [set(by_model[key]) for key in keys]
    if any(len(tasks) != expected_tasks for tasks in task_sets):
        counts = {key: len(by_model[key]) for key in keys}
        raise ValueError(f"expected {expected_tasks} outcomes per model; found {counts}")
    if any(tasks != task_sets[0] for tasks in task_sets[1:]):
        raise ValueError("all models must contain the identical task IDs")

    rates: dict[str, dict[str, float]] = {}
    for key in keys:
        records = by_model[key].values()
        rates[key] = {
            "humaneval": sum(item["humaneval_passed"] for item in records) / expected_tasks,
            "humaneval_plus": sum(
                item["humaneval_plus_passed"] for item in records
            )
            / expected_tasks,
        }

    published_rates = {
        item["key"]: {
            "humaneval": item["humaneval"] / 100.0,
            "humaneval_plus": item["humaneval_plus"] / 100.0,
        }
        for item in models
    }
    deltas = {
        key: {
            benchmark: rates[key][benchmark] - published_rates[key][benchmark]
            for benchmark in ("humaneval", "humaneval_plus")
        }
        for key in keys
    }

    published = [item["humaneval"] for item in models]
    local = [rates[key]["humaneval"] for key in keys]
    tau = kendall_tau_b(published, local)
    tasks = sorted(task_sets[0])
    adjacent = []
    significant_reversal = False
    for index, (lower, higher) in enumerate(zip(keys, keys[1:], strict=False)):
        differences = [
            float(by_model[higher][task]["humaneval_passed"])
            - float(by_model[lower][task]["humaneval_passed"])
            for task in tasks
        ]
        interval = bootstrap_mean_ci(
            differences, replicates=bootstrap_replicates, seed=seed + index
        )
        difference = sum(differences) / expected_tasks
        reversal = difference < 0
        significant_reversal = significant_reversal or (reversal and interval[1] < 0)
        adjacent.append(
            {
                "published_lower": lower,
                "published_higher": higher,
                "local_difference": difference,
                "paired_bootstrap_ci_95": interval,
                "reversed": reversal,
            }
        )

    if significant_reversal:
        decision = "failed_to_reproduce"
    elif math.isclose(tau, 1.0) and all(item["local_difference"] > 0 for item in adjacent):
        decision = "reproduced"
    else:
        decision = "inconclusive"
    return {
        "primary_endpoint": "kendall_tau_b",
        "kendall_tau_b": tau,
        "decision": decision,
        "local_rates": rates,
        "published_rates": published_rates,
        "local_minus_published": deltas,
        "published_order_low_to_high": keys,
        "local_order_low_to_high": sorted(keys, key=lambda key: rates[key]["humaneval"]),
        "adjacent_comparisons": adjacent,
        "expected_tasks": expected_tasks,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze the preregistered ranking endpoint.")
    parser.add_argument("outcomes", type=Path, help="Task-level consolidated JSONL outcomes.")
    parser.add_argument(
        "--targets", type=Path, default=Path("protocol/published_targets.json")
    )
    parser.add_argument("--expected-tasks", type=int, default=164)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    result = analyze(
        targets,
        load_jsonl(args.outcomes),
        expected_tasks=args.expected_tasks,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
