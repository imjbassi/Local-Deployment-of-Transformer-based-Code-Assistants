"""Run the five pinned greedy EvalPlus generation conditions sequentially."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
from typing import Protocol

EVALPLUS_VERSION = "0.3.1"
DATASET_VERSION = "v0.1.10"
MAX_NEW_TOKENS = 512


class SupportsGenerationCap(Protocol):
    max_new_tokens: int


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def enforce_generation_cap(model: SupportsGenerationCap) -> None:
    """Override EvalPlus's version-dependent provider default."""
    model.max_new_tokens = MAX_NEW_TOKENS
    if model.max_new_tokens != MAX_NEW_TOKENS:
        raise RuntimeError("failed to enforce the primary generation token cap")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=Path("protocol/published_targets.json"))
    parser.add_argument("--output-root", type=Path, default=Path("results/evalplus"))
    parser.add_argument("--only", choices=None, help="Run one model key for a smoke test.")
    parser.add_argument("--id-range", nargs=2, type=int, metavar=("LOW", "HIGH"))
    args = parser.parse_args(argv)

    if importlib.metadata.version("evalplus") != EVALPLUS_VERSION:
        raise RuntimeError(f"this protocol requires EvalPlus {EVALPLUS_VERSION}")

    import stop_sequencer.stop_sequencer as stop_module
    import torch
    from evalplus.codegen import codegen
    from evalplus.provider import make_model
    from huggingface_hub import snapshot_download
    from transformers import StoppingCriteria

    class DecodeOnceStopCriteria(StoppingCriteria):
        """Exact batch-one EvalPlus stop predicate with one decode per token."""

        def __init__(self, model_type, tokenizer, stop_texts, input_length, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_type = model_type
            self.tokenizer = tokenizer
            self.stop_texts = stop_texts
            self.input_length = input_length

        def __call__(self, input_ids, scores, **kwargs):
            del scores, kwargs
            if input_ids.shape[0] != 1:
                raise RuntimeError("the optimized primary stop criterion requires batch size one")
            token_ids = input_ids[0].long().tolist()
            if self.model_type == "causal":
                token_ids = token_ids[self.input_length :]
            decoded = self.tokenizer.decode(token_ids)
            return any(text in decoded for text in self.stop_texts)

    stop_module.StopSequenceCriteria = DecodeOnceStopCriteria

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the primary generation run")

    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    selected = [item for item in targets["models"] if not args.only or item["key"] == args.only]
    if not selected:
        raise ValueError(f"unknown model key: {args.only}")

    output_dir = args.output_root / "humaneval"
    output_dir.mkdir(parents=True, exist_ok=True)
    for target in selected:
        output_path = output_dir / f"{target['key']}.jsonl"
        expected = (args.id_range[1] - args.id_range[0]) if args.id_range else 164
        if line_count(output_path) >= expected:
            print(f"Skipping complete run: {output_path}", flush=True)
            continue
        snapshot_path = snapshot_download(
            repo_id=target["model_id"], revision=target["revision"]
        )
        device_name = torch.cuda.get_device_name(0)
        print(
            f"Generating {target['key']} from {target['revision']} on {device_name}",
            flush=True,
        )
        model = make_model(
            model=snapshot_path,
            backend="hf",
            batch_size=1,
            temperature=0.0,
            force_base_prompt=True,
            dataset="humaneval",
            attn_implementation="eager",
            dtype="bfloat16",
        )
        enforce_generation_cap(model)
        codegen(
            target_path=str(output_path),
            model=model,
            dataset="humaneval",
            greedy=True,
            n_samples=1,
            id_range=args.id_range,
            version=DATASET_VERSION,
            resume=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
