import os
from functools import lru_cache


def _get_config_path():
    return os.environ.get("LLM_CFG", os.path.join(os.path.dirname(__file__), "config.yaml"))

def _simple_yaml(text: str) -> dict:
    result = {}
    stack = [(result, -1)]
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.lstrip().partition(':')
        value = value.strip()
        if value == '' or value == '|':
            node = {}
        elif value.startswith('[') and value.endswith(']'):
            node = [v.strip() for v in value[1:-1].split(',') if v.strip()]
        else:
            num = value.replace('_', '')
            if num.isdigit():
                node = int(num)
            else:
                try:
                    node = float(num)
                except ValueError:
                    node = value
        while stack and indent <= stack[-1][1]:
            stack.pop()
        parent, _ = stack[-1]
        parent[key] = node
        if isinstance(node, dict):
            stack.append((node, indent))
    return result


@lru_cache
def _load_cfg(path: str):
    with open(path) as f:
        text = f.read()
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    if yaml is not None:
        return yaml.safe_load(text)
    return _simple_yaml(text)


def load_cfg():
    return _load_cfg(_get_config_path())

def _find_rule(task: str, *, tokens: int = 0):
    cfg = load_cfg()
    rules = cfg["routing_rules"]

    for pattern, rule in rules.items():
        name, *cond = pattern.split(">")
        names = name.split("|")
        if any(n in task for n in names):
            if cond:
                limit = cond[0]
                if limit.endswith("k"):
                    try:
                        limit_val = int(limit[:-1]) * 1000
                    except ValueError:
                        limit_val = int(limit)
                else:
                    limit_val = int(limit)
                if tokens <= limit_val:
                    continue
            return rule
    raise ValueError("No routing rule found")


def choose_model(task: str, *, tokens: int = 0):
    return _find_rule(task, tokens=tokens)["primary"]

def wrap_call(task, prompt, **kw):
    rule = _find_rule(task, tokens=len(str(prompt)) // 4)
    m_id = rule["primary"]
    try:
        return _call(m_id, prompt, **kw)
    except Exception as e:
        for _ in range(load_cfg()["defaults"]["max_retry"]):
            m_id = rule["fallback"]
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
