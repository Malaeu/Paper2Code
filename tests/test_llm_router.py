import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pytest

import llm_router.router as router

CONFIG_PATH = os.path.join(ROOT, "llm_router", "config.yaml")


def setup_function(function):
    router._load_cfg.cache_clear()
    os.environ.pop("LLM_CFG", None)


def test_choose_model_basic():
    assert router.choose_model("rag") == "gemini_flash_25"


def test_choose_model_long_doc():
    assert router.choose_model("long_doc", tokens=350_000) == "gpt41"


def test_choose_model_no_match():
    with pytest.raises(ValueError):
        router.choose_model("unknown")


def test_wrap_call_primary(monkeypatch):
    called = {}

    def fake_call(model_id, prompt, **kw):
        called["model"] = model_id
        return "ok"

    monkeypatch.setattr(router, "_call", fake_call)
    assert router.wrap_call("rag", "hi") == "ok"
    assert called["model"] == "gemini_flash_25"


def test_wrap_call_fallback(monkeypatch):
    seq = iter([Exception("fail"), "success"])
    called = []

    def fake_call(model_id, prompt, **kw):
        result = next(seq)
        if isinstance(result, Exception):
            raise result
        called.append(model_id)
        return result

    monkeypatch.setattr(router, "_call", fake_call)
    assert router.wrap_call("rag", "hi") == "success"
    assert called == ["claude_sonnet_35"]


def test_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "routing_rules:\n  foo:\n    primary: bar\n    fallback: baz\n"
    )
    monkeypatch.setenv("LLM_CFG", str(cfg))
    router._load_cfg.cache_clear()
    monkeypatch.setattr(router, "_call", lambda mid, prompt, **kw: mid)
    assert router.wrap_call("foo", "x") == "bar"

