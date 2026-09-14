"""Command-line interface for the benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import DEFAULT_MODELS, MODELS, RunConfig, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark local transformer code models on HumanEval."
    )
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(DEFAULT_MODELS))
    parser.add_argument("--limit", type=int, help="Use only the first N tasks (smoke tests only).")
    parser.add_argument("--samples-per-task", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--decoding",
        choices=("greedy", "sample"),
        default="sample",
        help="Greedy is the primary published-ranking replication condition.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--quantization", choices=("none", "8bit", "4bit"), default="none")
    parser.add_argument(
        "--prompt-prefix",
        help="Override the model's default prompt prefix; pass an empty string for no prefix.",
    )
    parser.add_argument("--mode", choices=("generation", "performance"), default="generation")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timing-repetitions", type=int, default=3)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path)
    return parser


def positive(value: int | float, label: str) -> None:
    if value <= 0:
        raise ValueError(f"{label} must be greater than zero")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        positive(args.samples_per_task, "samples-per-task")
        positive(args.max_new_tokens, "max-new-tokens")
        positive(args.bootstrap_replicates, "bootstrap-replicates")
        positive(args.timing_repetitions, "timing-repetitions")
        if args.limit is not None:
            positive(args.limit, "limit")
        if args.warmup_runs < 0:
            raise ValueError("warmup-runs cannot be negative")
        if args.decoding == "sample":
            if not 0 < args.top_p <= 1:
                raise ValueError("top-p must be in (0, 1]")
            positive(args.temperature, "temperature")
        elif args.samples_per_task != 1:
            raise ValueError("greedy decoding requires --samples-per-task 1")
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = args.output_dir or Path("results") / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    config = RunConfig(
        models=tuple(args.models),
        limit=args.limit,
        samples_per_task=args.samples_per_task,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        decoding=args.decoding,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        prompt_prefix=args.prompt_prefix,
        mode=args.mode,
        warmup_runs=args.warmup_runs,
        timing_repetitions=args.timing_repetitions,
        bootstrap_replicates=args.bootstrap_replicates,
        output_dir=output_dir,
    )
    run(config)
    print(f"Artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
