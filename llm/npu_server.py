#!/usr/bin/env python3
"""Ollama- and OpenAI-compatible HTTP API for the NPU LLM (npu_llm.py).

Point any Ollama client (Open WebUI, `ollama`-python, curl ...) or OpenAI
client at this server instead of Ollama. It serves one model, named
"qwen2.5-npu:0.5b", whose matmuls run on the Ryzen AI NPU.

Endpoints:
    Ollama:  GET  /  /api/version  /api/tags  /api/ps
             POST /api/chat  /api/generate  /api/show
    OpenAI:  GET  /v1/models
             POST /v1/chat/completions  /v1/completions

The default port is 11435, since a real Ollama usually owns 11434.
Requests are served one at a time; the NPU runs a single generation.

Usage:
    .venv/bin/python npu_server.py [--host 127.0.0.1] [--port 11435]
"""

import argparse
import json
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import npu_llm

MODEL_NAME = "qwen2.5-npu:0.5b"
DEFAULT_MAX_TOKENS = 512
MAX_TOKENS_LIMIT = 4096

STATE = {}  # tok, net, rng, lock, temperature -- filled in main()


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def text_of(content):
    """OpenAI allows content as a list of parts; keep the text ones."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def clean_messages(messages):
    return [{"role": m.get("role", "user"), "content": text_of(m.get("content"))}
            for m in messages]


def model_entry():
    return {
        "name": MODEL_NAME, "model": MODEL_NAME, "modified_at": now_iso(),
        "size": 988_000_000, "digest": "npu-" + npu_llm.MODEL_ID.replace("/", "-"),
        "details": {"format": "safetensors", "family": "qwen2", "families": ["qwen2"],
                    "parameter_size": "494M", "quantization_level": "BF16"},
    }


def run(messages, max_tokens, temperature):
    """Generator of text pieces for one request, holding the NPU lock.

    Returns the stats dict from npu_llm.stream_generate when exhausted.
    """
    info = {}
    with STATE["lock"]:
        yield from npu_llm.stream_generate(
            STATE["net"], STATE["tok"], messages, max_tokens, temperature, STATE["rng"], info)
    return info


def collect(gen):
    """Drain a run() generator: (full_text, info)."""
    parts = []
    while True:
        try:
            parts.append(next(gen))
        except StopIteration as stop:
            return "".join(parts), stop.value


def stream(gen, on_piece):
    """Call on_piece for each piece of a run() generator, return its info."""
    while True:
        try:
            on_piece(next(gen))
        except StopIteration as stop:
            return stop.value


def ollama_final(info, t_start):
    ns = lambda s: int(s * 1e9)
    return {
        "done": True, "done_reason": info["finish"],
        "total_duration": ns(time.perf_counter() - t_start),
        "load_duration": 0,
        "prompt_eval_count": info["prompt_tokens"], "prompt_eval_duration": ns(info["prefill_s"]),
        "eval_count": len(info["tokens"]), "eval_duration": ns(info["decode_s"]),
    }


def openai_usage(info):
    p, c = info["prompt_tokens"], len(info["tokens"])
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


class Handler(BaseHTTPRequestHandler):
    server_version = "npu-llm/0.1"

    # ---- plumbing -----------------------------------------------------------

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}", file=sys.stderr)

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def start_stream(self, content_type):
        # No Content-Length: the body streams until the connection closes.
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True

    def write_line(self, obj, sse=False):
        data = json.dumps(obj)
        self.wfile.write((f"data: {data}\n\n" if sse else data + "\n").encode())
        self.wfile.flush()

    def read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def params(self, req, openai):
        """(max_tokens, temperature) from an Ollama or OpenAI request."""
        opts = req.get("options") or {}
        if openai:
            max_tokens = req.get("max_completion_tokens") or req.get("max_tokens")
            temperature = req.get("temperature")
        else:
            max_tokens = opts.get("num_predict", req.get("num_predict"))
            temperature = opts.get("temperature", req.get("temperature"))
        if max_tokens is None or max_tokens < 0:
            max_tokens = DEFAULT_MAX_TOKENS
        max_tokens = min(int(max_tokens), MAX_TOKENS_LIMIT)
        if temperature is None:
            temperature = STATE["temperature"]
        return max_tokens, float(temperature)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    # ---- routes -------------------------------------------------------------

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "":
            body = b"Ollama is running (npu-llm)"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/version":
            self.send_json({"version": "0.6.0-npu"})
        elif path in ("/api/tags", "/api/ps"):
            self.send_json({"models": [model_entry()]})
        elif path == "/v1/models":
            self.send_json({"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "npu"}]})
        else:
            self.send_json({"error": f"not found: {path}"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            req = self.read_json()
        except (ValueError, json.JSONDecodeError) as e:
            return self.send_json({"error": f"bad JSON: {e}"}, 400)
        try:
            if path == "/api/chat":
                self.ollama(req, chat=True)
            elif path == "/api/generate":
                self.ollama(req, chat=False)
            elif path == "/api/show":
                e = model_entry()
                self.send_json({"details": e["details"], "model_info": {
                    "general.architecture": "qwen2", "general.basename": npu_llm.MODEL_ID},
                    "capabilities": ["completion"], "modified_at": e["modified_at"]})
            elif path == "/v1/chat/completions":
                self.openai(req, chat=True)
            elif path == "/v1/completions":
                self.openai(req, chat=False)
            else:
                self.send_json({"error": f"not found: {path}"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away mid-stream

    def ollama(self, req, chat):
        t_start = time.perf_counter()
        max_tokens, temperature = self.params(req, openai=False)
        if chat:
            messages = clean_messages(req.get("messages") or [])
        else:
            messages = []
            if req.get("system"):
                messages.append({"role": "system", "content": req["system"]})
            messages.append({"role": "user", "content": req.get("prompt", "")})
        if not any(m["content"] for m in messages):
            # Ollama answers an empty request by "loading" the model.
            return self.send_json({"model": MODEL_NAME, "created_at": now_iso(), "done": True,
                                   "done_reason": "load",
                                   **({"message": {"role": "assistant", "content": ""}} if chat
                                      else {"response": ""})})

        def chunk(text):
            base = {"model": MODEL_NAME, "created_at": now_iso()}
            if chat:
                base["message"] = {"role": "assistant", "content": text}
            else:
                base["response"] = text
            return base

        gen = run(messages, max_tokens, temperature)
        if req.get("stream", True):
            self.start_stream("application/x-ndjson")
            info = stream(gen, lambda piece: self.write_line({**chunk(piece), "done": False}))
            self.write_line({**chunk(""), **ollama_final(info, t_start)})
        else:
            text, info = collect(gen)
            self.send_json({**chunk(text), **ollama_final(info, t_start)})

    def openai(self, req, chat):
        max_tokens, temperature = self.params(req, openai=True)
        if chat:
            messages = clean_messages(req.get("messages") or [])
        else:
            prompt = req.get("prompt", "")
            messages = [{"role": "user", "content": prompt if isinstance(prompt, str) else "".join(prompt)}]
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        finish = lambda info: "stop" if info["finish"] == "stop" else "length"

        gen = run(messages, max_tokens, temperature)
        if req.get("stream"):
            self.start_stream("text/event-stream")

            def delta(piece, first=[True]):
                if chat:
                    d = {"content": piece}
                    if first[0]:
                        d["role"], first[0] = "assistant", False
                    choice = {"index": 0, "delta": d, "finish_reason": None}
                else:
                    choice = {"index": 0, "text": piece, "finish_reason": None}
                self.write_line({"id": rid, "object": obj + ".chunk" if chat else obj,
                                 "created": created, "model": MODEL_NAME, "choices": [choice]},
                                sse=True)

            info = stream(gen, delta)
            end = ({"index": 0, "delta": {}, "finish_reason": finish(info)} if chat
                   else {"index": 0, "text": "", "finish_reason": finish(info)})
            last = {"id": rid, "object": obj + ".chunk" if chat else obj, "created": created,
                    "model": MODEL_NAME, "choices": [end]}
            if (req.get("stream_options") or {}).get("include_usage"):
                last["usage"] = openai_usage(info)
            self.write_line(last, sse=True)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            text, info = collect(gen)
            choice = ({"index": 0, "message": {"role": "assistant", "content": text},
                       "finish_reason": finish(info)} if chat
                      else {"index": 0, "text": text, "finish_reason": finish(info)})
            self.send_json({"id": rid, "object": obj, "created": created, "model": MODEL_NAME,
                            "choices": [choice], "usage": openai_usage(info)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on (0.0.0.0 to allow other machines)")
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="default when a request doesn't set one (0 = greedy)")
    args = ap.parse_args()

    tok, _, net = npu_llm.load()
    STATE.update(tok=tok, net=net, rng=np.random.default_rng(), lock=threading.Lock(),
                 temperature=args.temperature)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {MODEL_NAME} on http://{args.host}:{args.port} "
          f"(Ollama API at /api, OpenAI API at /v1)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
