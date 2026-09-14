"""Regression checks for the preregistered primary generation settings."""

from types import SimpleNamespace

from scripts.primary_codegen import MAX_NEW_TOKENS, enforce_generation_cap


def test_evalplus_model_receives_explicit_512_token_cap() -> None:
    """Do not silently inherit EvalPlus's version-dependent token default."""
    model = SimpleNamespace(max_new_tokens=768)

    enforce_generation_cap(model)

    assert MAX_NEW_TOKENS == 512
    assert model.max_new_tokens == 512
