import os
try:
    import yaml
except Exception:
    yaml = None
from functools import lru_cache

CONFIG_PATH = os.environ.get("LLM_CFG", "llm_router/config.yaml")

def _simple_yaml(text: str) -> dict:
    data = {}
    stack = [data]
    indents = [0]
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        while indent < indents[-1]:
            stack.pop()
            indents.pop()
        key, _, val = line.lstrip().partition(":")
        val = val.strip()
        if not val:
            new = {}
            stack[-1][key] = new
            stack.append(new)
            indents.append(indent + 2)
        else:
            if val.startswith("[") and val.endswith("]"):
                items = [v.strip() for v in val[1:-1].split(",") if v.strip()]
                stack[-1][key] = items
            else:
                v = val.replace("_", "").lower()
                mult = 1
                if v.endswith("k"):
                    mult = 1000
                    v = v[:-1]
                elif v.endswith("m"):
                    mult = 1000000
                    v = v[:-1]
                try:
                    num = float(v)
                    if num.is_integer():
                        stack[-1][key] = int(num * mult)
                    else:
                        stack[-1][key] = num * mult
                    continue
                except ValueError:
                    pass
                stack[-1][key] = val
    return data

@lru_cache
def _load_cfg():
    with open(CONFIG_PATH) as f:
        text = f.read()
    if yaml is not None:
        return yaml.safe_load(text)
    return _simple_yaml(text)

def _parse_num(txt: str) -> int:
    txt = txt.replace("_", "").lower()
    mult = 1
    if txt.endswith("k"):
        mult = 1000
        txt = txt[:-1]
    elif txt.endswith("m"):
        mult = 1000000
        txt = txt[:-1]
    return int(float(txt) * mult)

def _find_rule(task: str, tokens: int = 0):
    cfg = _load_cfg()
    for pattern, rule in cfg["routing_rules"].items():
        name, *cond = pattern.split(">")
        names = name.split("|")
        if any(n and n in task for n in names):
            if cond:
                threshold = _parse_num(cond[0])
                if tokens <= threshold:
                    continue
            return rule
    return None


def choose_model(task: str, *, tokens: int = 0):
    rule = _find_rule(task, tokens)
    if not rule:
        raise ValueError("No routing rule found")
    return rule["primary"]

def wrap_call(task, prompt, **kw):
    tokens = len(str(prompt)) // 4
    rule = _find_rule(task, tokens)
    if not rule:
        raise ValueError("No routing rule found")
    m_id = rule["primary"]
    try:
        return _call(m_id, prompt, **kw)
    except Exception as e:
        for _ in range(_load_cfg()["defaults"]["max_retry"]):
            m_id = rule.get("fallback")
            if not m_id:
                break
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
