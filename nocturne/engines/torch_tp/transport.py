"""Our TCP transport: one connection, length prefixed frames, bf16 payload, TCP_NODELAY, large socket buffers.
Frame: 16 byte header (u32 payload length, u8 kind, u8 dtype code, u16 ndim, u32 aux0, u32 aux1) + dims (u32 each) +
raw bytes. For two ranks an all-reduce is one frame each way plus a local add: both sides send first and then
receive. Frames up to SMALL_FRAME are sent inline on the caller's thread (both kernels buffer SOCK_BUF, so two such
frames in flight cannot deadlock); larger frames (prefill) go through a thread. recv(into=...) fills a caller's
buffer when the payload length matches, so the decode path makes no copies. Frame and byte counters per direction
are read by the bench (comm.rpc_commands_per_step = all-reduce frames per decode step).
Security: the listening side accepts any peer; deferred by ADR-009, never bind 0.0.0.0 (CLAUDE.md rule 7)."""
import socket
import struct
import threading

HEADER = struct.Struct("<IBBHII")
KIND_STEP, KIND_ALLREDUCE, KIND_RESET, KIND_METRICS, KIND_METRICS_REPLY, KIND_SHUTDOWN, KIND_HELLO = range(1, 8)
KIND_NAMES = {KIND_STEP: "step", KIND_ALLREDUCE: "allreduce", KIND_RESET: "reset", KIND_METRICS: "metrics",
              KIND_METRICS_REPLY: "metrics_reply", KIND_SHUTDOWN: "shutdown", KIND_HELLO: "hello"}
DTYPES = {0: None, 1: "bfloat16", 2: "float32", 3: "int64", 4: "uint8"}
DTYPE_CODES = {v: k for k, v in DTYPES.items()}
SOCK_BUF = 8 << 20
SMALL_FRAME = 1 << 20


class Counters:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.sent_frames = self.recv_frames = 0
        self.sent_bytes = self.recv_bytes = 0
        self.allreduce_frames = 0      # all-reduce frames sent (one per all-reduce for two ranks)
        self.step_frames = 0
        self.reconnects = 0

    def add(self, sent: bool, kind: int, nbytes: int):
        with self.lock:
            if sent:
                self.sent_frames += 1; self.sent_bytes += nbytes
                if kind == KIND_ALLREDUCE:
                    self.allreduce_frames += 1
                elif kind == KIND_STEP:
                    self.step_frames += 1
            else:
                self.recv_frames += 1; self.recv_bytes += nbytes

    def snapshot(self) -> dict:
        with self.lock:
            return {"sent_frames": self.sent_frames, "recv_frames": self.recv_frames, "sent_bytes": self.sent_bytes,
                    "recv_bytes": self.recv_bytes, "allreduce_frames": self.allreduce_frames, "step_frames": self.step_frames,
                    "reconnects": self.reconnects}


def _tune(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, SOCK_BUF)
        except OSError:
            pass


def _recv_into(sock: socket.socket, view: memoryview) -> None:
    got, n = 0, len(view)
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed the connection")
        got += k


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    _recv_into(sock, memoryview(buf))
    return bytes(buf)


class Link:
    """One connected socket with framed send and receive."""

    def __init__(self, sock: socket.socket, counters: Counters | None = None):
        _tune(sock)
        self.sock = sock
        self.counters = counters or Counters()
        self._send_lock = threading.Lock()

    def send(self, kind: int, payload=b"", dtype: str | None = None, shape: tuple[int, ...] = (),
             aux0: int = 0, aux1: int = 0) -> None:
        n = len(payload)
        head = HEADER.pack(n, kind, DTYPE_CODES[dtype], len(shape), aux0, aux1) + struct.pack(f"<{len(shape)}I", *shape)
        with self._send_lock:
            self.sock.sendall(head + payload)
        self.counters.add(True, kind, len(head) + n)

    def recv(self, into: memoryview | None = None) -> tuple[int, bytes | memoryview, str | None, tuple[int, ...], int, int]:
        """`into`: a writable buffer; when its length equals the payload length the payload is received into it and
        returned as that view (no copy), otherwise a new bytes object is returned."""
        head = _recv_exact(self.sock, HEADER.size)
        n, kind, dcode, ndim, aux0, aux1 = HEADER.unpack(head)
        shape = struct.unpack(f"<{ndim}I", _recv_exact(self.sock, 4 * ndim)) if ndim else ()
        if into is not None and len(into) == n:
            _recv_into(self.sock, into)
            payload = into
        else:
            payload = _recv_exact(self.sock, n) if n else b""
        self.counters.add(False, kind, HEADER.size + 4 * ndim + n)
        return kind, payload, DTYPES[dcode], shape, aux0, aux1

    def send_async(self, *a, **kw) -> threading.Thread:
        t = threading.Thread(target=self.send, args=a, kwargs=kw, daemon=True)
        t.start()
        return t

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def listen(bind: str, port: int, backlog: int = 1) -> socket.socket:
    if bind.startswith("0.0.0.0") or bind in ("", "::"):
        raise ValueError("refusing to bind 0.0.0.0: choose the address of the network on purpose (CLAUDE.md rule 7)")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind, port))
    srv.listen(backlog)
    print(f"[torch_tp] listening on {bind}:{port}; the port accepts any rank 0, security deferred by ADR-009", flush=True)
    return srv


def connect(host: str, port: int, timeout_s: float = 120.0, retry_s: float = 0.5) -> socket.socket:
    import time
    t0 = time.time()
    while True:
        try:
            return socket.create_connection((host, port), timeout=30)
        except OSError:
            if time.time() - t0 > timeout_s:
                raise
            time.sleep(retry_s)
