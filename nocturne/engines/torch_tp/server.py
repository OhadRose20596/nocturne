"""HTTP on rank 0, 127.0.0.1 only: /health, /v1/chat/completions (stream, OpenAI chunk format, llama-server style
`timings` in the last chunk), /metrics (per rank memory, all-reduce counters, loaded bytes). One generation at a time.
Body fields read: messages, max_tokens, stream, ignore_eos (decode exactly max_tokens, for the fixed-token runs).
A request that loses the link to rank 1 ends with finish_reason "error" and an `error` field (503 when not
streaming); the next request reconnects."""
import http.server
import itertools
import json
import threading
import time

from .engine import LinkLost, Rank0, chat_ids


class ChatHandler(http.server.BaseHTTPRequestHandler):
    engine: Rank0 = None
    tokenizer = None
    lock = threading.Lock()
    model_name = "qwen2"
    counter = itertools.count(1)

    def log_message(self, *a):
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/metrics":
            with self.lock:
                self._json(200, self.engine.metrics())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        messages = req.get("messages") or []
        max_tokens = int(req.get("max_tokens") or 256)
        ignore_eos = bool(req.get("ignore_eos", False))
        stream = bool(req.get("stream", False))
        ids = chat_ids(self.tokenizer, messages)
        rid = f"req-{next(self.counter)}"
        cid = f"chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())

        def chunk(delta: dict, finish=None, extra=None) -> bytes:
            d = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": self.model_name,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if extra:
                d.update(extra)
            return b"data: " + json.dumps(d).encode() + b"\n\n"

        with self.lock:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(chunk({"role": "assistant", "content": ""}))
                self.wfile.flush()
                pending: list[int] = []

                def on_token(t: int):
                    if t in self.engine.cfg.eos_token_ids:
                        return
                    pending.append(t)
                    text = self.tokenizer.decode(pending, skip_special_tokens=True)
                    if text and not text.endswith("�"):        # wait for a complete UTF-8 sequence
                        self.wfile.write(chunk({"content": text}))
                        self.wfile.flush()
                        pending.clear()
                try:
                    res = self.engine.generate(ids, max_tokens, on_token=on_token, ignore_eos=ignore_eos, rid=rid)
                except LinkLost as e:
                    self.wfile.write(chunk({}, "error", {"error": f"link to rank 1 lost: {e}"}))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                tail = self.tokenizer.decode(pending, skip_special_tokens=True) if pending else ""
                self.wfile.write(chunk({"content": tail} if tail else {}, res["finish_reason"],
                                       {"timings": res["timings"], "frames": res["frames"]}))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                try:
                    res = self.engine.generate(ids, max_tokens, ignore_eos=ignore_eos, rid=rid)
                except LinkLost as e:
                    self._json(503, {"error": f"link to rank 1 lost: {e}"})
                    return
                text = self.tokenizer.decode([t for t in res["tokens"] if t not in self.engine.cfg.eos_token_ids], skip_special_tokens=True)
                self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": self.model_name,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": res["finish_reason"]}],
                                 "usage": {"prompt_tokens": len(ids), "completion_tokens": len(res["tokens"]),
                                           "total_tokens": len(ids) + len(res["tokens"])},
                                 "timings": res["timings"], "frames": res["frames"]})


def serve(engine: Rank0, tokenizer, host: str = "127.0.0.1", port: int = 8092, model_name: str = "qwen2") -> http.server.ThreadingHTTPServer:
    if host != "127.0.0.1":
        raise ValueError("the HTTP side binds 127.0.0.1 only; the bench talks to it locally")
    handler = type("Bound", (ChatHandler,), {"engine": engine, "tokenizer": tokenizer, "model_name": model_name,
                                             "lock": threading.Lock(), "counter": itertools.count(1)})
    srv = http.server.ThreadingHTTPServer((host, port), handler)
    print(f"[torch_tp] serving http://{host}:{port} (world {engine.world})", flush=True)
    return srv
