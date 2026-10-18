"""GPT-OSS tokenizer (o200k_harmony) from the GGUF, and the harmony chat format.

Reuses Strata's byte-level BPE (tools/strata_tokenizer.py) with gpt-oss's pre-tokenizer: llama.cpp's
LLAMA_VOCAB_PRE_TYPE_GPT4O, the original tokenizer.json pattern (Python's `regex` runs it as written).
"""
from __future__ import annotations

import datetime
import json
import pathlib
import sys

import regex

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402

GPT4O_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class OssTokenizer(Tokenizer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._re = regex.compile(GPT4O_PATTERN)
        self.pre = "gpt-4o"

    def id(self, tok: str) -> int:
        return self.ids[tok]


def load(gguf_path: str) -> OssTokenizer:
    t = Tokenizer.from_gguf(gguf_path)
    o = OssTokenizer.__new__(OssTokenizer)
    o.__dict__.update(t.__dict__)
    o._re = regex.compile(GPT4O_PATTERN)
    o.pre = "gpt-4o"
    return o


# ---------------------------------------------------------------- harmony
def harmony_prompt(messages: list[dict], reasoning: str = "medium", tools: list[dict] | None = None,
                   date: str | None = None) -> str:
    """The prompt text for a chat in OpenAI's harmony format, ending where the assistant's answer starts.

    messages: OpenAI chat messages (system / developer / user / assistant / tool).  A system message becomes the
    developer instructions, as in the model card's reference template.  An assistant message's `reasoning_content`
    is kept only for the last assistant turn before a tool result (the format drops earlier reasoning).
    """
    date = date or datetime.date.today().isoformat()
    sys_text = ("You are ChatGPT, a large language model trained by OpenAI.\n"
                "Knowledge cutoff: 2024-06\n"
                f"Current date: {date}\n\n"
                f"Reasoning: {reasoning}\n\n"
                "# Valid channels: analysis, commentary, final. Channel must be included for every message.")
    if tools:
        sys_text += "\nCalls to these tools must go to the commentary channel: 'functions'."
    out = [f"<|start|>system<|message|>{sys_text}<|end|>"]
    dev = [m for m in messages if m.get("role") in ("system", "developer")]
    instr = "\n\n".join(_text(m.get("content")) for m in dev)
    if instr or tools:
        body = ""
        if instr:
            body += "# Instructions\n\n" + instr
        if tools:
            body += ("\n\n" if body else "") + "# Tools\n\n## functions\n\nnamespace functions {\n\n"
            for t in tools:
                f = t.get("function", t)
                if f.get("description"):
                    body += "// " + f["description"].replace("\n", "\n// ") + "\n"
                body += "type " + f["name"] + " = (" + _ts_params(f.get("parameters")) + ") => any;\n\n"
            body += "} // namespace functions"
        out.append(f"<|start|>developer<|message|>{body}<|end|>")
    rest = [m for m in messages if m.get("role") not in ("system", "developer")]
    # the last assistant turn that is followed by tool results keeps its reasoning
    last_user = max((i for i, m in enumerate(rest) if m.get("role") == "user"), default=-1)
    for i, m in enumerate(rest):
        role = m.get("role")
        if role == "user":
            out.append(f"<|start|>user<|message|>{_text(m.get('content'))}<|end|>")
        elif role == "assistant":
            rc = m.get("reasoning_content") or m.get("reasoning")
            if rc and i > last_user:
                out.append(f"<|start|>assistant<|channel|>analysis<|message|>{rc}<|end|>")
            for tc in m.get("tool_calls") or []:
                f = tc.get("function", {})
                args = f.get("arguments", "{}")
                if not isinstance(args, str):
                    args = json.dumps(args)
                out.append(f"<|start|>assistant to=functions.{f.get('name')}<|channel|>commentary json"
                           f"<|message|>{args}<|call|>")
            c = _text(m.get("content"))
            if c:
                out.append(f"<|start|>assistant<|channel|>final<|message|>{c}<|end|>")
        elif role == "tool":
            name = m.get("name") or _tool_name(rest, m.get("tool_call_id"))
            out.append(f"<|start|>functions.{name} to=assistant<|channel|>commentary<|message|>"
                       f"{_text(m.get('content'))}<|end|>")
    out.append("<|start|>assistant")
    return "".join(out)


def _tool_name(msgs, call_id):
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            if tc.get("id") == call_id:
                return tc.get("function", {}).get("name", "tool")
    return "tool"


def _text(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") in ("text", "input_text"))


def _ts_type(s: dict) -> str:
    if not isinstance(s, dict):
        return "any"
    if "enum" in s:
        return " | ".join(json.dumps(v) for v in s["enum"])
    t = s.get("type")
    if t == "string":
        return "string"
    if t in ("number", "integer"):
        return "number"
    if t == "boolean":
        return "boolean"
    if t == "array":
        return _ts_type(s.get("items", {})) + "[]"
    if t == "object":
        return "{\n" + _ts_props(s) + "}"
    return "any"


def _ts_props(s: dict) -> str:
    req = set(s.get("required", []))
    out = ""
    for k, v in (s.get("properties") or {}).items():
        if v.get("description"):
            out += "// " + v["description"].replace("\n", "\n// ") + "\n"
        out += k + ("" if k in req else "?") + ": " + _ts_type(v)
        if "default" in v:
            out += ", // default: " + json.dumps(v["default"])
        out += ",\n"
    return out


def _ts_params(p) -> str:
    if not p or not p.get("properties"):
        return ""
    return "_: {\n" + _ts_props(p) + "}"


if __name__ == "__main__":
    tk = load(sys.argv[1])
    for s in ["Hello, world!", "The quick brown fox's 12345 jumps.\n\n  def f(x):\n\treturn x",
              harmony_prompt([{"role": "user", "content": "What is 2+2?"}])]:
        ids = tk.encode(s, parse_special=True)
        print(len(ids), ids[:40])
        assert tk.decode(ids) == s, (tk.decode(ids), s)
    print("ok")
