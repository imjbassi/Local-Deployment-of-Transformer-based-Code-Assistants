import pytest
from local_code_benchmark.metrics import bootstrap_mean_ci, pass_at_k


def test_pass_at_one_uses_unbiased_estimator() -> None:
    assert pass_at_k([[True, False, False, False, False]], 1) == pytest.approx(0.2)


def test_pass_at_five_is_one_when_any_of_five_passes() -> None:
    assert pass_at_k([[False, False, True, False, False]], 5) == 1.0


def test_pass_at_k_averages_tasks() -> None:
    assert pass_at_k([[True, False], [False, False]], 1) == pytest.approx(0.25)


def test_pass_at_k_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="requires at least"):
        pass_at_k([[True]], 2)


def test_bootstrap_interval_is_reproducible_and_bounded() -> None:
    low, high = bootstrap_mean_ci([0.0, 0.5, 1.0], replicates=1000, seed=7)
    assert 0.0 <= low <= high <= 1.0
    assert (low, high) == bootstrap_mean_ci([0.0, 0.5, 1.0], replicates=1000, seed=7)
