"""Regression checks for the preregistered primary generation settings."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_evalplus_model_receives_explicit_512_token_cap() -> None:
    """Do not silently inherit EvalPlus's version-dependent token default."""
    source = (ROOT / "scripts" / "primary_codegen.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    make_model_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "make_model"
    ]

    assert assignments["MAX_NEW_TOKENS"] == 512
    assert len(make_model_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in make_model_calls[0].keywords}
    assert isinstance(keywords["max_new_tokens"], ast.Name)
    assert keywords["max_new_tokens"].id == "MAX_NEW_TOKENS"
