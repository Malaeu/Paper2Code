import os
from functools import lru_cache

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback if PyYAML missing
    yaml = None


def _parse_tokens(text: str) -> int:
    """Return integer token count from shorthand like '300k' or '1_000'."""
    text = text.replace('_', '').lower()
    mul = 1
    if text.endswith('k'):
        mul = 1000
        text = text[:-1]
    elif text.endswith('m'):
        mul = 1_000_000
        text = text[:-1]
    return int(float(text) * mul)


def _simple_yaml_load(text: str) -> dict:
    """Very small YAML loader supporting a subset of the syntax."""
    data: dict = {}
    stack = [(0, data)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith('#'):
            continue
        indent = len(raw) - len(raw.lstrip())
        key, _, val = raw.lstrip().partition(':')
        key = key.strip()
        val = val.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if val == '':
            new: dict = {}
            current[key] = new
            stack.append((indent + 2, new))
            continue
        if val.startswith('[') and val.endswith(']'):
            items = [v.strip() for v in val[1:-1].split(',') if v.strip()]
            current[key] = items
            continue
        try:
            current[key] = _parse_tokens(val)
        except Exception:
            if val.lower() in {'true', 'false'}:
                current[key] = val.lower() == 'true'
            else:
                current[key] = val
    return data


@lru_cache
def _load_cfg():
    path = os.environ.get("LLM_CFG", "llm_router/config.yaml")
    with open(path) as f:
        raw = f.read()
    if yaml:
        return yaml.safe_load(raw)
    return _simple_yaml_load(raw)

def choose_model(task: str, *, tokens: int = 0) -> str:
    cfg = _load_cfg()
    rules = cfg["routing_rules"]

    for pattern, rule in rules.items():
        name_part, *cond = pattern.split(">")
        names = name_part.split("|")
        if not any(n in task for n in names):
            continue
        if cond:
            limit = _parse_tokens(cond[0])
            if tokens <= limit:
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
