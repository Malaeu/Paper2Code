import os
import sys
import importlib
import builtins
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import llm_router.router as router


def test_choose_model_basic():
    assert router.choose_model("chat") == "gemini_flash_25"
    assert router.choose_model("code") == "claude_sonnet_37"


def test_token_threshold():
    assert router.choose_model("long_doc", tokens=400000) == "gpt41"
    with pytest.raises(ValueError):
        router.choose_model("long_doc", tokens=100000)


def test_wrap_call_fallback(monkeypatch):
    calls = []

    def fake_call(mid, prompt, **kw):
        calls.append(mid)
        if len(calls) == 1:
            raise RuntimeError("fail")
        return "ok"

    monkeypatch.setattr(router, "_call", fake_call)
    assert router.wrap_call("chat", "hello") == "ok"
    assert calls == ["gemini_flash_25", "claude_sonnet_35"]


def test_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
 defaults:
   max_retry: 1
 routing_rules:
   foo:
     primary: bar
     fallback: baz
"""
    )
    monkeypatch.setenv("LLM_CFG", str(cfg))
    importlib.reload(router)
    assert router.choose_model("foo") == "bar"


def test_fallback_yaml_loader(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "yaml":
            raise ImportError
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    importlib.reload(router)
    try:
        assert router.choose_model("chat") == "gemini_flash_25"
    finally:
        monkeypatch.setattr(builtins, "__import__", orig_import)
