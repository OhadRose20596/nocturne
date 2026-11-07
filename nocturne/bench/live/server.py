"""HTTP server for the live page: `/` is page.html, `/events` is a Server-Sent Events stream, `/state` one JSON
snapshot. Binds 127.0.0.1 only; nothing here opens a port to the network."""
import http.server
import json
import socketserver
import threading
import time
from pathlib import Path

from .parsers import format_sse

PAGE = Path(__file__).with_name("page.html")


class Handler(http.server.BaseHTTPRequestHandler):
    state = None            # set by serve()
    period = 0.25

    def log_message(self, *a):       # quiet
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state":
            body = json.dumps(self.state.snapshot(time.time()) if self.state else {"mode": "waiting"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/events"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            seq = 0
            try:
                while True:
                    now = time.time()
                    if self.state is None:            # server up, no run attached yet
                        self.wfile.write(format_sse("state", {"mode": "waiting", "now": now}))
                        self.wfile.flush()
                        time.sleep(self.period)
                        continue
                    self.wfile.write(format_sse("state", self.state.snapshot(now)))
                    msgs = self.state.messages_since(seq)
                    if msgs:
                        seq = msgs[-1]["seq"]
                        self.wfile.write(format_sse("msgs", msgs))
                    self.wfile.flush()
                    time.sleep(self.period)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.send_error(404)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        # A browser that reloads or closes the tab resets the socket mid-request; not worth a traceback in the run output.
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def serve(state, port: int = 8091, host: str = "127.0.0.1") -> tuple[Server, str]:
    """Start serving in a daemon thread. Returns (server, url). port 0 picks a free port."""
    handler = type("BoundHandler", (Handler,), {"state": state})
    try:
        srv = Server((host, port), handler)
    except OSError as e:
        if port == 0:
            raise
        # another live view (or a stale one) holds the port: take a free one rather than abort the run
        print(f"[live] port {port} busy ({e.strerror}); using a free port", flush=True)
        srv = Server((host, 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://{host}:{srv.server_address[1]}/"
