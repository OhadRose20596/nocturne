"""Our TCP transport: one connection, length prefixed frames, bf16 payload, TCP_NODELAY, large socket buffers.
Frame: 16 byte header (u32 payload length, u8 kind, u8 dtype code, u16 ndim, u32 aux0, u32 aux1) + dims (u32 each) +
raw bytes. For two ranks an all-reduce is one frame each way plus a local add: both sides send first and then
receive. A frame is sent inline on the caller's thread when it is at most SMALL_FRAME and when it fits half the send
buffer the kernel really granted, together with everything sent on the link and not yet answered (Link.fits_inline). Other frames (prefill) go
through a thread. The kernel may grant far less than SOCK_BUF: Linux caps the request at net.core.wmem_max without an
error (425984 bytes on gtx, 8 MiB on the Macs), so the granted size is read back and kept on the Link. recv(into=...) fills a caller's
buffer when the payload length matches, so the decode path makes no copies. Frame and byte counters per direction
are read by the bench (comm.rpc_commands_per_step = all-reduce frames per decode step).
Session 10 added two frame kinds, both on the ctl link (coordinator to one rank), neither is a data frame:
CLOCK (clock.py): aux1 is the operation (0 request, 1 reply, 2 done, 3 begin), aux0 the round number. A reply carries
  16 bytes: the rank's wall clock and monotonic clock in ns (two little endian int64).
WEIGHTS (shipping.py): aux0 = operation | slice index << 8. Operations: 1 MANIFEST (JSON, coordinator to rank), 2 NEED
  (JSON list of slice indexes, rank to coordinator), 3 CHUNK (raw bytes of one slice, at most CHUNK bytes, aux1 = CRC32
  of the payload), 4 END (end of one slice), 5 ACK (rank to coordinator, aux1 = 0 accepted, 1 rejected; JSON reason),
  6 DONE (no more slices). One slice at a time, in manifest order; the rank answers every END with an ACK.
In TOKEN and HIDDEN frames aux1 holds the flags in bits 0 to 7. When the run records a timeline, bits 8 to 23 carry
the request number; without the recorder they are zero and the frames are byte for byte what they were before.
Session 10 part 3 (overlap, docs/ENGINE_DESIGN.md section 11), all inside existing fields, all zero with the options off:
  TOKEN, HIDDEN aux1: flag bit 1 = no head result for this frame (a prefill chunk that is not the last), flag bit 2 =
    first frame of an interleaved pair (the next frame on the link is its partner); bits 24 to 31 = the slot.
  HEAD aux0: bits 0 to 23 = index of the best token, bits 24 to 31 = the slot.
A send waits and goes on when the kernel has no buffer space for a moment (ENOBUFS; seen on the macOS loopback while
GBs of weights were in flight to two ranks). The bytes and their order on the wire do not change.
Security: the listening side accepts any peer; deferred by ADR-009, never bind 0.0.0.0 (CLAUDE.md rule 7)."""
import errno
import socket
import struct
import threading
import time

HEADER = struct.Struct("<IBBHII")
KIND_STEP, KIND_ALLREDUCE, KIND_RESET, KIND_METRICS, KIND_METRICS_REPLY, KIND_SHUTDOWN, KIND_HELLO = range(1, 8)
# session 6 (hybrid placement): token ids to the first stage, hidden state between stages, (value, index) or local
# logits from a head rank, the plan, readiness, and a rank's fatal error
KIND_TOKEN, KIND_HIDDEN, KIND_HEAD, KIND_PLAN, KIND_READY, KIND_ERROR = range(8, 14)
KIND_CLOCK, KIND_WEIGHTS = 14, 15
KIND_NAMES = {KIND_STEP: "step", KIND_ALLREDUCE: "allreduce", KIND_RESET: "reset", KIND_METRICS: "metrics",
              KIND_METRICS_REPLY: "metrics_reply", KIND_SHUTDOWN: "shutdown", KIND_HELLO: "hello", KIND_TOKEN: "token",
              KIND_HIDDEN: "hidden", KIND_HEAD: "head", KIND_PLAN: "plan", KIND_READY: "ready", KIND_ERROR: "error",
              KIND_CLOCK: "clock", KIND_WEIGHTS: "weights"}
DATA_KINDS = (KIND_STEP, KIND_ALLREDUCE, KIND_TOKEN, KIND_HIDDEN, KIND_HEAD)
DTYPES = {0: None, 1: "bfloat16", 2: "float32", 3: "int64", 4: "uint8"}
DTYPE_CODES = {v: k for k, v in DTYPES.items()}
SOCK_BUF = 8 << 20
SMALL_FRAME = 1 << 20


def frame_bytes(payload_len: int, ndim: int = 0) -> int:
    """Bytes of one frame on the wire (header, dims, payload)."""
    return HEADER.size + 4 * ndim + payload_len


class Phase:
    """Process wide flag: True while the current step is a decode step (one token). Set by the step's owner."""
    decode = False


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
        self.send_waits = 0            # times a send waited for kernel buffer space (ENOBUFS)
        self.decode = {"sent_frames": 0, "sent_bytes": 0, "recv_frames": 0, "recv_bytes": 0}
        self.by_kind: dict[str, int] = {}

    def add(self, sent: bool, kind: int, nbytes: int, decode: bool = False):
        with self.lock:
            if sent:
                self.sent_frames += 1; self.sent_bytes += nbytes
                if kind == KIND_ALLREDUCE:
                    self.allreduce_frames += 1
                elif kind == KIND_STEP:
                    self.step_frames += 1
                k = KIND_NAMES.get(kind, str(kind))
                self.by_kind[k] = self.by_kind.get(k, 0) + 1
            else:
                self.recv_frames += 1; self.recv_bytes += nbytes
            if decode and kind in DATA_KINDS:
                d = "sent" if sent else "recv"
                self.decode[f"{d}_frames"] += 1
                self.decode[f"{d}_bytes"] += nbytes

    def snapshot(self) -> dict:
        with self.lock:
            return {"sent_frames": self.sent_frames, "recv_frames": self.recv_frames, "sent_bytes": self.sent_bytes,
                    "recv_bytes": self.recv_bytes, "allreduce_frames": self.allreduce_frames, "step_frames": self.step_frames,
                    "reconnects": self.reconnects, "send_waits": self.send_waits, "decode": dict(self.decode),
                    "by_kind_sent": dict(self.by_kind)}


def _tune(sock: socket.socket) -> tuple[int, int]:
    """Ask for SOCK_BUF in both directions. Returns what the kernel granted (send, receive), as it reports them."""
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, SOCK_BUF)
        except OSError:
            pass
    return sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF), sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)


ENOBUFS_WAIT_S = 30.0


def _send_all(sock: socket.socket, data, counters=None) -> None:
    """sock.sendall, except that ENOBUFS is waited out: send what fits, sleep 5 ms, go on from the same byte. sendall
    cannot be retried, because it does not say how much it sent. Gives up after ENOBUFS_WAIT_S without progress."""
    view = memoryview(data)
    sent, since = 0, None
    while sent < len(view):
        try:
            sent += sock.send(view[sent:])
            since = None
        except OSError as e:
            if e.errno != errno.ENOBUFS:
                raise
            now = time.monotonic()
            since = now if since is None else since
            if now - since > ENOBUFS_WAIT_S:
                raise
            if counters is not None:
                counters.send_waits += 1
            time.sleep(0.005)


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

    def __init__(self, sock: socket.socket, counters: Counters | None = None, phase: Phase | None = None):
        self.sndbuf, self.rcvbuf = _tune(sock)
        self.sock = sock
        self.counters = counters or Counters()
        self.phase = phase
        self.last_sent = self.last_recv = 0        # wire bytes of the latest frame, read by the recorder
        self._send_lock = threading.Lock()

    def send(self, kind: int, payload=b"", dtype: str | None = None, shape: tuple[int, ...] = (),
             aux0: int = 0, aux1: int = 0, decode: bool | None = None) -> None:
        """`decode`: is this a frame of a decode step (one token)? None asks the process wide Phase flag, as before. With
        several requests in flight the caller says it, because the flag is only right for the step being computed."""
        n = len(payload)
        head = HEADER.pack(n, kind, DTYPE_CODES[dtype], len(shape), aux0, aux1) + struct.pack(f"<{len(shape)}I", *shape)
        with self._send_lock:
            _send_all(self.sock, head + payload, self.counters)
        self.last_sent = len(head) + n
        self.counters.add(True, kind, len(head) + n, bool(self.phase and self.phase.decode) if decode is None else decode)

    def recv(self, into: memoryview | None = None, decode=None) -> tuple[int, bytes | memoryview, str | None, tuple[int, ...], int, int]:
        """`into`: a writable buffer; when its length equals the payload length the payload is received into it and
        returned as that view (no copy), otherwise a new bytes object is returned.
        `decode`: a function (kind, aux0, aux1) -> bool that says whether the frame belongs to a decode step, for a
        receiver that knows it from the frame's slot. None keeps the rule below."""
        head = _recv_exact(self.sock, HEADER.size)
        n, kind, dcode, ndim, aux0, aux1 = HEADER.unpack(head)
        shape = struct.unpack(f"<{ndim}I", _recv_exact(self.sock, 4 * ndim)) if ndim else ()
        if into is not None and len(into) == n:
            _recv_into(self.sock, into)
            payload = into
        else:
            payload = _recv_exact(self.sock, n) if n else b""
        # a received data frame is a decode frame when it carries one token (shape[0] == 1); HEAD frames follow the phase
        if decode is not None:
            dec = bool(decode(kind, aux0, aux1))
        else:
            dec = (shape[0] == 1) if shape and kind != KIND_HEAD else bool(self.phase and self.phase.decode)
        self.last_recv = HEADER.size + 4 * ndim + n
        self.counters.add(False, kind, self.last_recv, dec)
        return kind, payload, DTYPES[dcode], shape, aux0, aux1

    def fits_inline(self, nbytes: int, outstanding: int = 0) -> bool:
        """May a frame of nbytes be sent on the caller's thread, when the peer does the same and reads afterwards?
        `outstanding`: bytes of the frames already sent on this link, inline or on a helper thread, whose answer has not
        been received. The frame must be small, and all of it must fit half the granted send buffer (Linux reports twice
        the usable size)."""
        return nbytes <= SMALL_FRAME and outstanding + nbytes <= self.sndbuf // 2

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
