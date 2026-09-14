"""Regression checks for the preregistered primary generation settings."""

from types import SimpleNamespace

from scripts.primary_codegen import (
    MAX_NEW_TOKENS,
    deduplicate_stop_texts,
    enforce_generation_cap,
    install_non_accumulating_codegen,
)


def test_evalplus_model_receives_explicit_512_token_cap() -> None:
    """Do not silently inherit EvalPlus's version-dependent token default."""
    model = SimpleNamespace(max_new_tokens=768)

    enforce_generation_cap(model)

    assert MAX_NEW_TOKENS == 512
    assert model.max_new_tokens == 512


def test_repeated_evalplus_stop_texts_are_deduplicated_in_order() -> None:
    model = SimpleNamespace(eos=["</s>", "\ndef ", "</s>", "\ndef ", "\nclass "])

    deduplicate_stop_texts(model)

    assert model.eos == ["</s>", "\ndef ", "\nclass "]


def test_evalplus_stopping_hook_does_not_accumulate_between_tasks() -> None:
    original_hook = object()
    mutated_hook = object()

    class FakeDecoder:
        def __init__(self) -> None:
            self.model = SimpleNamespace(_get_stopping_criteria=original_hook)

        def codegen(self, value: str) -> str:
            self.model._get_stopping_criteria = mutated_hook
            return value

    install_non_accumulating_codegen(FakeDecoder)
    decoder = FakeDecoder()

    assert decoder.codegen("completion") == "completion"
    assert decoder.model._get_stopping_criteria is original_hook
