import pytest

from local_code_benchmark.analyze import analyze, kendall_tau_b

TARGETS = {
    "models": [
        {"key": "a", "humaneval": 10.0, "humaneval_plus": 8.0},
        {"key": "b", "humaneval": 20.0, "humaneval_plus": 18.0},
        {"key": "c", "humaneval": 30.0, "humaneval_plus": 28.0},
    ]
}


def outcomes(values: dict[str, list[bool]]) -> list[dict[str, object]]:
    return [
        {
            "model_key": key,
            "task_id": f"HumanEval/{index}",
            "humaneval_passed": passed,
            "humaneval_plus_passed": passed,
        }
        for key, passes in values.items()
        for index, passed in enumerate(passes)
    ]


def test_kendall_tau_b_detects_reverse_order() -> None:
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0


def test_analysis_reproduces_strict_order() -> None:
    result = analyze(
        TARGETS,
        outcomes({"a": [False] * 4, "b": [False, True, False, True], "c": [True] * 4}),
        expected_tasks=4,
        bootstrap_replicates=100,
    )
    assert result["decision"] == "reproduced"
    assert result["kendall_tau_b"] == 1.0


def test_analysis_rejects_duplicate_outcome() -> None:
    rows = outcomes({"a": [False] * 2, "b": [True] * 2, "c": [True] * 2})
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate outcome"):
        analyze(TARGETS, rows, expected_tasks=2, bootstrap_replicates=10)


def test_analysis_fails_on_resolved_reversal() -> None:
    result = analyze(
        TARGETS,
        outcomes({"a": [True] * 20, "b": [False] * 20, "c": [True] * 20}),
        expected_tasks=20,
        bootstrap_replicates=100,
    )
    assert result["decision"] == "failed_to_reproduce"
