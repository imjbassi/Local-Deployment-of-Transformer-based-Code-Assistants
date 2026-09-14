"""Convert pinned EvalPlus result files into the analysis outcome contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def convert_evalplus(model_key: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    evaluations = payload.get("eval")
    if not isinstance(evaluations, dict):
        raise ValueError("EvalPlus result has no 'eval' object")
    outcomes = []
    for task_id, attempts in sorted(evaluations.items()):
        if not isinstance(attempts, list) or len(attempts) != 1:
            raise ValueError(f"{model_key}/{task_id} must contain exactly one greedy attempt")
        attempt = attempts[0]
        base_passed = attempt.get("base_status") == "pass"
        plus_passed = base_passed and attempt.get("plus_status") == "pass"
        outcomes.append(
            {
                "model_key": model_key,
                "task_id": task_id,
                "humaneval_passed": base_passed,
                "humaneval_plus_passed": plus_passed,
            }
        )
    return outcomes


def parse_result(value: str) -> tuple[str, Path]:
    key, separator, raw_path = value.partition("=")
    if not separator or not key or not raw_path:
        raise argparse.ArgumentTypeError("results must use MODEL_KEY=PATH")
    return key, Path(raw_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidate greedy EvalPlus results.")
    parser.add_argument(
        "--result", action="append", required=True, type=parse_result, metavar="MODEL_KEY=PATH"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    records = []
    for model_key, path in args.result:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(convert_evalplus(model_key, payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Wrote {len(records)} task-level outcomes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
