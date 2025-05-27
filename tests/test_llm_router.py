import os
import sys
from pathlib import Path
import importlib

import pytest

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_router import router


@pytest.fixture(autouse=True)
def clear_cache():
    router._load_cfg.cache_clear()
    yield
    router._load_cfg.cache_clear()


def make_cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """defaults:\n  max_retry: 2\nrouting_rules:\n  foo:\n    primary: m1\n    fallback: m2\n  long>10k:\n    primary: big\n    fallback: small\n"""
    )
    return cfg


def test_choose_model_basic(tmp_path, monkeypatch):
    cfg_path = make_cfg(tmp_path)
    monkeypatch.setenv("LLM_CFG", str(cfg_path))
    assert router.choose_model("foo") == "m1"
    assert router.choose_model("long", tokens=20000) == "big"
    with pytest.raises(ValueError):
        router.choose_model("missing")


def test_env_override(tmp_path, monkeypatch):
    cfg_path = make_cfg(tmp_path)
    monkeypatch.setenv("LLM_CFG", str(cfg_path))
    assert router.choose_model("foo") == "m1"


def test_wrap_call_fallback(tmp_path, monkeypatch):
    cfg_path = make_cfg(tmp_path)
    monkeypatch.setenv("LLM_CFG", str(cfg_path))
    calls = []

    def fake_call(mid, prompt, **kw):
        calls.append(mid)
        if mid == "m1":
            raise RuntimeError("fail")
        return "ok"

    monkeypatch.setattr(router, "_call", fake_call)
    out = router.wrap_call("foo", "hi")
    assert out == "ok"
    assert calls == ["m1", "m2"]

