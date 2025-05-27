import os
import sys
import importlib
from types import SimpleNamespace

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import llm_router.router as router

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "llm_router", "config.yaml")


def reload_router(monkeypatch, cfg_path=CFG_PATH, no_yaml=False):
    monkeypatch.setenv("LLM_CFG", cfg_path)
    if no_yaml:
        monkeypatch.setattr(router, "yaml", None, raising=False)
    importlib.reload(router)


def test_choose_model_patterns(monkeypatch):
    reload_router(monkeypatch)
    assert router.choose_model("unit_tests") == "claude_sonnet_37"
    assert router.choose_model("chat") == "gemini_flash_25"


def test_choose_model_threshold(monkeypatch):
    reload_router(monkeypatch)
    with pytest.raises(ValueError):
        router.choose_model("long_doc", tokens=1000)
    assert router.choose_model("long_doc", tokens=400000) == "gpt41"


def test_wrap_call_uses_fallback(monkeypatch):
    reload_router(monkeypatch)
    calls = []

    def fake_call(model, prompt, **kw):
        calls.append(model)
        if model == "claude_sonnet_37":
            raise RuntimeError
        return "ok"

    monkeypatch.setattr(router, "_call", fake_call)
    out = router.wrap_call("code", "hello")
    assert out == "ok"
    assert calls == ["claude_sonnet_37", "o4mini"]


def test_fallback_yaml_loader(monkeypatch, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("""defaults:\n  max_retry: 1\nrouting_rules:\n  test:\n    primary: A\n    fallback: B\n""")
    reload_router(monkeypatch, str(cfg), no_yaml=True)
    assert router.choose_model("test") == "A"

