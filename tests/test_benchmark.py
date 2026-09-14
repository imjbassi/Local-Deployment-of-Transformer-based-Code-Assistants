from pathlib import Path

from local_code_benchmark.benchmark import (
    RunConfig,
    generated_token_count,
    generation_options,
    normalize_completion,
)


def test_normalize_completion_removes_echoed_prompt() -> None:
    assert normalize_completion("def f():\n    return 1", "def f():\n") == "    return 1"


def test_normalize_completion_preserves_non_echoed_text() -> None:
    assert normalize_completion("    return 1", "def f():\n") == "    return 1"


def test_generated_token_count_excludes_padding() -> None:
    assert generated_token_count([4, 5, 0, 0], pad_token_id=0) == 2


def config(decoding: str) -> RunConfig:
    return RunConfig(
        models=("qwen2.5-coder-0.5b",),
        limit=1,
        samples_per_task=1,
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.95,
        decoding=decoding,
        seed=11,
        device="cpu",
        dtype="float32",
        quantization="none",
        prompt_prefix=None,
        mode="generation",
        warmup_runs=0,
        timing_repetitions=1,
        bootstrap_replicates=100,
        output_dir=Path("results/test"),
    )


def test_greedy_generation_omits_sampling_parameters() -> None:
    options = generation_options(config("greedy"), pad_token_id=0)
    assert options["do_sample"] is False
    assert "temperature" not in options
    assert "top_p" not in options


def test_sample_generation_includes_sampling_parameters() -> None:
    options = generation_options(config("sample"), pad_token_id=0)
    assert options["temperature"] == 0.2
    assert options["top_p"] == 0.95
