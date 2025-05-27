import os
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None
from functools import lru_cache

DEFAULT_CONFIG_PATH = "llm_router/config.yaml"

def _simple_yaml_load(text: str) -> dict:
    root: dict = {}
    stack = [(-1, root)]  # (indent, container)
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.lstrip().partition(":")
        key = key.strip()
        val = val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]
        if not val:
            new_dict = {}
            container[key] = new_dict
            stack.append((indent, new_dict))
        else:
            if val.startswith("[") and val.endswith("]"):
                val = [v.strip() for v in val[1:-1].split(",") if v.strip()]
            else:
                num = val.replace("_", "")
                if num.lower().endswith("k") and num[:-1].isdigit():
                    val = int(num[:-1]) * 1000
                elif num.replace(".", "", 1).isdigit():
                    val = float(num) if "." in num else int(num)
            container[key] = val
    return root


@lru_cache
def _load_cfg():
    path = os.environ.get("LLM_CFG", DEFAULT_CONFIG_PATH)
    with open(path) as f:
        text = f.read()
    if yaml is not None:
        return yaml.safe_load(text)
    return _simple_yaml_load(text)

def _parse_int(s: str) -> int:
    """Parse integers allowing underscores and k-suffix."""
    s = s.replace("_", "").lower()
    if s.endswith("k"):
        return int(s[:-1]) * 1000
    return int(s)


def _match_rule(task: str, tokens: int):
    cfg = _load_cfg()
    for pattern, rule in cfg["routing_rules"].items():
        name, *cond = pattern.split(">")
        if any(part in task for part in name.split("|")):
            if cond:
                limit = _parse_int(cond[0])
                if tokens <= limit:
                    continue
            return rule
    raise ValueError("No routing rule found")


def choose_model(task: str, *, tokens: int = 0):
    return _match_rule(task, tokens)["primary"]

def wrap_call(task, prompt, **kw):
    tokens = len(str(prompt)) // 4
    rule = _match_rule(task, tokens)
    primary = rule["primary"]
    fallback = rule.get("fallback")
    try:
        return _call(primary, prompt, **kw)
    except Exception as e:
        if not fallback:
            raise e
        for _ in range(_load_cfg()["defaults"]["max_retry"]):
            try:
                return _call(fallback, prompt, **kw)
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
