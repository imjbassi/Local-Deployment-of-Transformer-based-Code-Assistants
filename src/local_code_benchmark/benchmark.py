"""Core benchmark orchestration and artifact generation."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import bootstrap_mean_ci

DATASET_ID = "openai/openai_humaneval"
DATASET_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
PROMPT_PREFIX = "# Here is the correct implementation of the code exercise\n"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    prompt_prefix: str = ""


MODELS = {
    "codet5-small": ModelSpec("CodeT5-small", "Salesforce/codet5-small"),
    "codet5-base": ModelSpec("CodeT5-base", "Salesforce/codet5-base"),
    "starcoderbase-1b": ModelSpec("StarCoderBase-1B", "bigcode/starcoderbase-1b"),
    "starcoderbase-1b-prompted": ModelSpec(
        "StarCoderBase-1B (prompted)", "bigcode/starcoderbase-1b", PROMPT_PREFIX
    ),
    "starcoder-15b": ModelSpec("StarCoder-15B", "bigcode/starcoder"),
    "qwen2.5-coder-0.5b": ModelSpec("Qwen2.5-Coder-0.5B", "Qwen/Qwen2.5-Coder-0.5B"),
    "qwen2.5-coder-1.5b": ModelSpec("Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-Coder-1.5B"),
    "qwen2.5-coder-3b": ModelSpec("Qwen2.5-Coder-3B", "Qwen/Qwen2.5-Coder-3B"),
    "starcoder2-3b": ModelSpec("StarCoder2-3B", "bigcode/starcoder2-3b"),
    "deepseek-coder-1.3b": ModelSpec(
        "DeepSeek-Coder-1.3B", "deepseek-ai/deepseek-coder-1.3b-base"
    ),
    "codellama-python-7b": ModelSpec(
        "Code Llama Python 7B", "codellama/CodeLlama-7b-Python-hf"
    ),
}
DEFAULT_MODELS = ("qwen2.5-coder-0.5b",)


@dataclass(frozen=True)
class RunConfig:
    models: tuple[str, ...]
    limit: int | None
    samples_per_task: int
    max_new_tokens: int
    temperature: float
    top_p: float
    decoding: str
    seed: int
    device: str
    dtype: str
    quantization: str
    prompt_prefix: str | None
    mode: str
    warmup_runs: int
    timing_repetitions: int
    bootstrap_replicates: int
    output_dir: Path


def select_device(requested: str, torch_module: Any) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str, torch_module: Any) -> Any:
    if name == "auto":
        return torch_module.float16 if device == "cuda" else torch_module.float32
    return {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }[name]


def synchronize(device: str, torch_module: Any) -> None:
    if device == "cuda":
        torch_module.cuda.synchronize()


def normalize_completion(text: str, prompt: str) -> str:
    """Remove an echoed prompt while preserving completion whitespace."""
    return text[len(prompt) :] if text.startswith(prompt) else text


def generated_token_count(token_ids: Any, pad_token_id: int | None) -> int:
    values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    if pad_token_id is None:
        return len(values)
    return sum(token != pad_token_id for token in values)


def generation_options(config: RunConfig, pad_token_id: int | None) -> dict[str, Any]:
    """Build generation arguments without leaking sampling controls into greedy runs."""
    options: dict[str, Any] = {
        "do_sample": config.decoding == "sample",
        "max_new_tokens": config.max_new_tokens,
        "num_return_sequences": config.samples_per_task,
        "pad_token_id": pad_token_id,
    }
    if config.decoding == "sample":
        options.update(temperature=config.temperature, top_p=config.top_p)
    return options


def load_dataset_rows(limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split="test", revision=DATASET_REVISION)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]


def load_model(
    spec: ModelSpec, device: str, dtype: Any, quantization: str
) -> tuple[Any, Any]:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    config = AutoConfig.from_pretrained(spec.model_id)
    model_class = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    load_kwargs: dict[str, Any] = {"dtype": dtype}
    if quantization != "none":
        if device != "cuda":
            raise RuntimeError("4-bit and 8-bit quantization require a CUDA device")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = {"": 0}
    model = model_class.from_pretrained(spec.model_id, **load_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if quantization == "none":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_samples(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: RunConfig,
    device: str,
    torch_module: Any,
) -> tuple[list[str], list[int], float]:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = int(encoded["input_ids"].shape[-1])
    torch_module.manual_seed(config.seed)
    if device == "cuda":
        torch_module.cuda.manual_seed_all(config.seed)

    synchronize(device, torch_module)
    started = time.perf_counter()
    with torch_module.inference_mode():
        sequences = model.generate(
            **encoded,
            **generation_options(config, tokenizer.pad_token_id),
        )
    synchronize(device, torch_module)
    elapsed = time.perf_counter() - started

    is_encoder_decoder = bool(getattr(model.config, "is_encoder_decoder", False))
    continuations = sequences if is_encoder_decoder else sequences[:, input_length:]
    texts = [
        normalize_completion(tokenizer.decode(ids, skip_special_tokens=True), prompt)
        for ids in continuations
    ]
    counts = [generated_token_count(ids, tokenizer.pad_token_id) for ids in continuations]
    return texts, counts, elapsed


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_is_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(torch_module: Any, device: str) -> dict[str, Any]:
    device_details = None
    if device == "cuda":
        index = torch_module.cuda.current_device()
        properties = torch_module.cuda.get_device_properties(index)
        device_details = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "cuda_runtime": torch_module.version.cuda,
            "cudnn_version": torch_module.backends.cudnn.version(),
            "driver_version": nvidia_driver_version(),
        }
    try:
        import psutil

        total_ram_bytes = int(psutil.virtual_memory().total)
    except ImportError:
        total_ram_bytes = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "total_ram_bytes": total_ram_bytes,
        "device": device,
        "device_details": device_details,
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "datasets",
                "evalplus",
                "accelerate",
                "bitsandbytes",
                "psutil",
            )
        },
    }


def nvidia_driver_version() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def benchmark_model(
    spec: ModelSpec,
    rows: list[dict[str, Any]],
    config: RunConfig,
    device: str,
    dtype: Any,
    torch_module: Any,
    samples_file: Any,
    timings_file: Any,
) -> dict[str, Any]:
    load_started = time.perf_counter()
    model, tokenizer = load_model(spec, device, dtype, config.quantization)
    model_load_seconds = time.perf_counter() - load_started
    if device == "cuda":
        torch_module.cuda.reset_peak_memory_stats()

    effective_prefix = (
        config.prompt_prefix if config.prompt_prefix is not None else spec.prompt_prefix
    )
    warmup_prompt = effective_prefix + rows[0]["prompt"]
    warmup_config = RunConfig(**{**asdict(config), "samples_per_task": 1})
    for _ in range(config.warmup_runs):
        generate_samples(model, tokenizer, warmup_prompt, warmup_config, device, torch_module)

    total_tokens = 0
    total_seconds = 0.0
    task_latencies: list[float] = []
    task_latency_means: list[float] = []
    task_throughput_means: list[float] = []

    for task_index, problem in enumerate(rows):
        prompt = effective_prefix + problem["prompt"]
        task_config = RunConfig(**{**asdict(config), "seed": config.seed + task_index})
        repetitions = config.timing_repetitions if config.mode == "performance" else 1
        generated_runs = []
        current_task_latencies: list[float] = []
        current_task_throughputs: list[float] = []
        for repetition in range(repetitions):
            repetition_config = RunConfig(
                **{**asdict(task_config), "seed": task_config.seed + repetition * 1_000_003}
            )
            generated_runs.append(
                generate_samples(model, tokenizer, prompt, repetition_config, device, torch_module)
            )
            _, repetition_counts, repetition_elapsed = generated_runs[-1]
            repetition_tokens = sum(repetition_counts)
            task_latencies.append(repetition_elapsed)
            current_task_latencies.append(repetition_elapsed)
            current_task_throughputs.append(repetition_tokens / repetition_elapsed)
            total_seconds += repetition_elapsed
            total_tokens += repetition_tokens
            timing_record = {
                "model": spec.name,
                "task_id": problem["task_id"],
                "repetition": repetition,
                "batch_size": config.samples_per_task,
                "seed": repetition_config.seed,
                "generated_tokens": repetition_tokens,
                "seconds": repetition_elapsed,
                "tokens_per_second": repetition_tokens / repetition_elapsed,
            }
            timings_file.write(json.dumps(timing_record, sort_keys=True) + "\n")
            timings_file.flush()
        task_latency_means.append(statistics.mean(current_task_latencies))
        task_throughput_means.append(statistics.mean(current_task_throughputs))
        completions, token_counts, elapsed = generated_runs[0]
        pairs = zip(completions, token_counts, strict=True)
        for sample_index, (completion, tokens) in enumerate(pairs):
            record = {
                "model": spec.name,
                "model_id": spec.model_id,
                "task_id": problem["task_id"],
                "sample_index": sample_index,
                "seed": task_config.seed,
                "completion": completion,
                "generated_tokens": tokens,
                "generation_call_seconds": elapsed,
            }
            samples_file.write(json.dumps(record, sort_keys=True) + "\n")
            samples_file.flush()
    quartiles = (
        statistics.quantiles(task_latencies, n=4, method="inclusive")
        if len(task_latencies) > 1
        else [task_latencies[0]] * 3
    )
    metrics: dict[str, Any] = {
        "model": spec.name,
        "model_id": spec.model_id,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "tasks": len(rows),
        "samples_per_task": config.samples_per_task,
        "generated_tokens": total_tokens,
        "generation_seconds": total_seconds,
        "aggregate_tokens_per_second": total_tokens / total_seconds,
        "mean_generation_call_seconds": sum(task_latencies) / len(task_latencies),
        "mean_generation_call_seconds_ci_95": bootstrap_mean_ci(
            task_latency_means, replicates=config.bootstrap_replicates, seed=config.seed + 2
        ),
        "mean_tokens_per_second_ci_95": bootstrap_mean_ci(
            task_throughput_means,
            replicates=config.bootstrap_replicates,
            seed=config.seed + 3,
        ),
        "median_generation_call_seconds": statistics.median(task_latencies),
        "generation_call_seconds_q1": quartiles[0],
        "generation_call_seconds_q3": quartiles[2],
        "model_load_seconds": model_load_seconds,
        "requested_precision": config.dtype,
        "resolved_torch_dtype": str(dtype),
        "quantization": config.quantization,
        "prompt_prefix": effective_prefix,
        "decoding": config.decoding,
        "peak_cuda_memory_bytes": (
            int(torch_module.cuda.max_memory_allocated()) if device == "cuda" else None
        ),
    }
    del model, tokenizer
    if device == "cuda":
        torch_module.cuda.empty_cache()
    return metrics


def run(config: RunConfig) -> dict[str, Any]:
    import torch

    if config.decoding == "greedy" and config.samples_per_task != 1:
        raise ValueError("greedy decoding requires exactly one sample per task")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = select_device(config.device, torch)
    dtype = resolve_dtype(config.dtype, device, torch)
    rows = load_dataset_rows(config.limit)
    if not rows:
        raise ValueError("the selected dataset slice is empty")

    config.output_dir.mkdir(parents=True, exist_ok=False)
    serializable_config = asdict(config)
    serializable_config["output_dir"] = str(config.output_dir)
    serializable_config["dataset_id"] = DATASET_ID
    serializable_config["dataset_revision"] = DATASET_REVISION
    write_json(config.output_dir / "run_config.json", serializable_config)
    write_json(config.output_dir / "environment.json", environment_metadata(torch, device))

    metrics: list[dict[str, Any]] = []
    with (
        (config.output_dir / "samples.jsonl").open("w", encoding="utf-8") as samples_file,
        (config.output_dir / "timings.jsonl").open("w", encoding="utf-8") as timings_file,
    ):
        for model_key in config.models:
            print(f"Benchmarking {MODELS[model_key].name} on {len(rows)} tasks...", flush=True)
            metrics.append(
                benchmark_model(
                    MODELS[model_key],
                    rows,
                    config,
                    device,
                    dtype,
                    torch,
                    samples_file,
                    timings_file,
                )
            )
            write_json(config.output_dir / "metrics.json", {"models": metrics})

    summary = {"models": metrics}
    write_json(config.output_dir / "metrics.json", summary)
    return summary
