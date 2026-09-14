import pytest
from local_code_benchmark.consolidate import convert_evalplus


def test_convert_evalplus_requires_base_and_plus_pass() -> None:
    payload = {
        "eval": {
            "HumanEval/0": [
                {"base_status": "pass", "plus_status": "fail"},
            ]
        }
    }
    record = convert_evalplus("model", payload)[0]
    assert record["humaneval_passed"] is True
    assert record["humaneval_plus_passed"] is False


def test_convert_evalplus_rejects_multiple_greedy_attempts() -> None:
    payload = {"eval": {"HumanEval/0": [{}, {}]}}
    with pytest.raises(ValueError, match="exactly one"):
        convert_evalplus("model", payload)
