"""strata-oss server: an OpenAI-compatible API for gpt-oss on the strata-oss engine.

    python oss/server.py --model gpt-oss-120b-MXFP4.gguf [--port 8096] [--host 127.0.0.1] [--api-key KEY]
                         [-- extra engine args, e.g. --ctx 65536 --profile oss120.prof]

Standard library only.  Endpoints: GET /v1/models, POST /v1/chat/completions (stream or not; `reasoning_content` for
the analysis channel, `tool_calls` for functions.* calls; `reasoning_effort` low|medium|high), POST /v1/completions
(raw prompt), GET /health.  One answer at a time (the engine holds one conversation in its KV cache and reuses the
longest common prefix of the next prompt); requests queue.  A client that disconnects cancels its answer.

Never expose it beyond 127.0.0.1 without --api-key.
"""
from __future__ import annotations

import argparse
import codecs
import json
import pathlib
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import oss_tokenizer  # noqa: E402

START, END, MESSAGE, CHANNEL, CONSTRAIN, RETURN, CALL = 200006, 200007, 200008, 200005, 200003, 200002, 200012
STOP_IDS = [RETURN, CALL, 199999]


class Engine:
    """The strata-oss --serve process: JSON lines in, JSON lines out."""

    def __init__(self, binary: str, model: str, extra: list[str]):
        cmd = [binary, "--model", model, "--serve"] + extra
        print("strata-oss server: starting", " ".join(cmd), file=sys.stderr, flush=True)
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lock = threading.Lock()
        self.events: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        ev = self.events.get()
        if ev.get("type") != "ready":
            raise RuntimeError("engine did not start: %r" % ev)
        self.n_ctx = ev["n_ctx"]

    def _read(self):
        for line in self.p.stdout:
            line = line.strip()
            if line.startswith("{"):
                try:
                    self.events.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
        self.events.put({"type": "exit"})

    def send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def generate(self, tokens, params, cancelled):
        """Yields token ids, then the final 'done' event (a dict)."""
        with self.lock:
            while not self.events.empty():
                self.events.get_nowait()
            self.send(dict(params, op="generate", tokens=tokens))
            sent_cancel = False
            while True:
                if cancelled() and not sent_cancel:
                    self.send({"op": "cancel"})
                    sent_cancel = True
                try:
                    ev = self.events.get(timeout=0.2)
                except queue.Empty:
                    continue
                t = ev.get("type")
                if t == "token":
                    yield ev["id"]
                elif t in ("done", "error", "exit"):
                    yield ev
                    return


class Harmony:
    """Token-level parser of the assistant's harmony output (the prompt ends with '<|start|>assistant')."""

    def __init__(self, tk):
        self.tk = tk
        self.state = "header"
        self.header: list[int] = []
        self.channel = ""
        self.recipient = None
        self.dec = codecs.getincrementaldecoder("utf-8")("replace")
        self.calls: list[dict] = []

    def _parse_header(self):
        text = self.tk.decode([t for t in self.header if t < 199998]).strip()
        # e.g. "analysis", "final", "commentary to=functions.get_weather json", " to=functions.x" before the channel
        ch, rec = "", None
        for part in text.replace("<|channel|>", " ").split():
            if part.startswith("to="):
                rec = part[3:]
            elif not ch and part in ("analysis", "commentary", "final"):
                ch = part
        if CHANNEL in self.header:
            i = self.header.index(CHANNEL)
            after = self.tk.decode([t for t in self.header[i + 1:] if t < 199998]).split()
            if after:
                ch = after[0]
            for p in after[1:]:
                if p.startswith("to="):
                    rec = p[3:]
        self.channel, self.recipient = ch or "final", rec

    def feed(self, t: int):
        """Returns a list of (kind, text) with kind in reasoning | content | tool_start | tool_args."""
        out = []
        if self.state == "header":
            if t == MESSAGE:
                self._parse_header()
                self.state = "body"
                self.dec = codecs.getincrementaldecoder("utf-8")("replace")
                if self.recipient and self.recipient.startswith("functions."):
                    call = {"id": "call_" + uuid.uuid4().hex[:24], "name": self.recipient[len("functions."):],
                            "arguments": ""}
                    self.calls.append(call)
                    out.append(("tool_start", call))
            elif t == START:
                self.header = []
            else:
                self.header.append(t)
            return out
        # body
        if t in (END, RETURN, CALL):
            self.state = "header"
            self.header = []
            return out
        if t >= 199998:
            return out
        text = self.dec.decode(self.tk.token_bytes(t))
        if not text:
            return out
        if self.recipient and self.recipient.startswith("functions."):
            self.calls[-1]["arguments"] += text
            out.append(("tool_args", text))
        elif self.channel == "analysis" or (self.channel == "commentary" and not self.recipient):
            out.append(("reasoning", text))
        else:
            out.append(("content", text))
        return out


def make_handler(eng: Engine, tk, model_name: str, api_key: str | None, default_effort: str):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _auth(self):
            if not api_key:
                return True
            if self.headers.get("Authorization", "") == "Bearer " + api_key or self.headers.get("x-api-key") == api_key:
                return True
            self._json(401, {"error": {"message": "invalid api key", "type": "invalid_request_error"}})
            return False

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", ""):
                return self._json(200, {"status": "ok"})
            if not self._auth():
                return
            if self.path.startswith("/v1/models"):
                return self._json(200, {"object": "list", "data": [
                    {"id": model_name, "object": "model", "owned_by": "strata-oss", "context_length": eng.n_ctx}]})
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self._auth():
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": {"message": "bad json"}})
            if self.path.startswith("/v1/chat/completions"):
                prompt = oss_tokenizer.harmony_prompt(
                    req.get("messages", []), reasoning=req.get("reasoning_effort") or default_effort,
                    tools=req.get("tools"))
                return self._run(req, tk.encode(prompt, parse_special=True), chat=True)
            if self.path.startswith("/v1/completions"):
                return self._run(req, tk.encode(req.get("prompt", ""), parse_special=True), chat=False)
            self._json(404, {"error": {"message": "not found"}})

        def _run(self, req, tokens, chat):
            stream = bool(req.get("stream"))
            max_tokens = int(req.get("max_completion_tokens") or req.get("max_tokens") or 4096)
            max_tokens = max(1, min(max_tokens, eng.n_ctx - len(tokens) - 1))
            params = {"max_tokens": max_tokens, "temperature": float(req.get("temperature", 1.0)),
                      "top_p": float(req.get("top_p", 1.0)), "top_k": int(req.get("top_k", 0)),
                      "min_p": float(req.get("min_p", 0.0)), "stop": STOP_IDS if chat else [RETURN, 199999]}
            if "seed" in req:
                params["seed"] = int(req["seed"])
            rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
            created = int(time.time())
            hp = Harmony(tk)
            gone = {"v": False}
            content, reasoning, n_gen, done = [], [], 0, {}
            raw_dec = codecs.getincrementaldecoder("utf-8")("replace")

            def chunk(delta, finish=None):
                if chat:
                    obj = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                else:
                    obj = {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                           "choices": [{"index": 0, "text": delta.get("content", ""), "finish_reason": finish}]}
                return obj

            def send_sse(obj):
                try:
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    gone["v"] = True

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                if chat:
                    send_sse(chunk({"role": "assistant", "content": ""}))
            for ev in eng.generate(tokens, params, lambda: gone["v"]):
                if isinstance(ev, dict):
                    done = ev
                    break
                n_gen += 1
                if not chat:
                    if ev in (RETURN, 199999):
                        continue
                    text = raw_dec.decode(tk.token_bytes(ev))
                    content.append(text)
                    if stream and text:
                        send_sse(chunk({"content": text}))
                    continue
                for kind, val in hp.feed(ev):
                    if kind == "reasoning":
                        reasoning.append(val)
                        if stream:
                            send_sse(chunk({"reasoning_content": val}))
                    elif kind == "content":
                        content.append(val)
                        if stream:
                            send_sse(chunk({"content": val}))
                    elif kind == "tool_start" and stream:
                        i = len(hp.calls) - 1
                        send_sse(chunk({"tool_calls": [{"index": i, "id": val["id"], "type": "function",
                                                        "function": {"name": val["name"], "arguments": ""}}]}))
                    elif kind == "tool_args" and stream:
                        send_sse(chunk({"tool_calls": [{"index": len(hp.calls) - 1, "function": {"arguments": val}}]}))
            reason = done.get("reason", "stop")
            if done.get("type") == "error":
                if stream:
                    send_sse({"error": {"message": done.get("error")}})
                    self._end_stream()
                    return
                return self._json(500, {"error": {"message": done.get("error")}})
            finish = "tool_calls" if hp.calls else ("length" if reason in ("length", "context") else "stop")
            usage = {"prompt_tokens": len(tokens), "completion_tokens": n_gen, "total_tokens": len(tokens) + n_gen,
                     "prompt_tokens_details": {"cached_tokens": done.get("n_reused", 0)}}
            tg = done.get("t_gen") or 0
            print(f"strata-oss server: {len(tokens)} prompt tokens ({done.get('n_reused', 0)} reused, "
                  f"{done.get('t_prompt', 0):.2f} s), {n_gen} generated ({n_gen / tg if tg else 0:.1f} tok/s), "
                  f"{reason}", file=sys.stderr, flush=True)
            if stream:
                send_sse(dict(chunk({}, finish), usage=usage))
                self._end_stream()
                return
            if chat:
                msg = {"role": "assistant", "content": "".join(content) or (None if hp.calls else "")}
                if reasoning:
                    msg["reasoning_content"] = "".join(reasoning)
                if hp.calls:
                    msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": c["arguments"]}}
                                         for c in hp.calls]
                return self._json(200, {"id": rid, "object": "chat.completion", "created": created,
                                        "model": model_name, "usage": usage,
                                        "choices": [{"index": 0, "message": msg, "finish_reason": finish}]})
            return self._json(200, {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                                    "usage": usage,
                                    "choices": [{"index": 0, "text": "".join(content), "finish_reason": finish}]})

        def _end_stream(self):
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True

    return H


def main():
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", default=str(here.parent / "build-oss" / "strata-oss"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--api-key")
    ap.add_argument("--name", default=None, help="model id the API reports (default: the GGUF's file name)")
    ap.add_argument("--reasoning", default="medium", choices=["low", "medium", "high"])
    a, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.api_key:
        sys.exit("strata-oss server: refusing to listen on %s without --api-key" % a.host)
    tk = oss_tokenizer.load(a.model)
    eng = Engine(a.engine, a.model, extra)
    name = a.name or pathlib.Path(a.model).stem
    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(eng, tk, name, a.api_key, a.reasoning))
    print(f"strata-oss server: http://{a.host}:{a.port}/v1 ({name}, context {eng.n_ctx})", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            eng.send({"op": "save_profile"})
            eng.p.stdin.close()
            eng.p.wait(timeout=30)
        except Exception:
            eng.p.kill()


if __name__ == "__main__":
    main()
