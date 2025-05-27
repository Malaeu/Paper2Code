import os
import yaml
from functools import lru_cache

CONFIG_PATH = os.environ.get("LLM_CFG", "llm_router/config.yaml")

@lru_cache
def _load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def choose_model(task: str, *, tokens: int = 0):
    cfg = _load_cfg()
    rules = cfg["routing_rules"]

    for pattern, rule in rules.items():
        name, *cond = pattern.split(">")
        if name in task:
            if cond and tokens <= int(cond[0]):   # long_doc
                continue
            return rule["primary"]
    raise ValueError("No routing rule found")

def wrap_call(task, prompt, **kw):
    m_id = choose_model(task, tokens=len(str(prompt))//4)
    try:
        return _call(m_id, prompt, **kw)
    except Exception as e:
        for _ in range(_load_cfg()["defaults"]["max_retry"]):
            m_id = _load_cfg()["routing_rules"][task]["fallback"]
            try:
                return _call(m_id, prompt, **kw)
            except Exception:
                continue
        raise e  # escalate

def _call(model_id, prompt, **kw):
    # Unified interface
    if model_id.startswith("gemini"):
        import google.generativeai as genai
        return genai.chat(model=model_id, messages=[prompt], **kw)
    elif model_id.startswith("claude"):
        import anthropic
        client = anthropic.Anthropic()
        return client.messages.create(model=model_id, messages=[{"role": "user", "content": prompt}], **kw)
    else:  # openai
        import openai
        return openai.ChatCompletion.create(model=model_id, messages=[{"role": "user", "content": prompt}], **kw)
