"""Clock alignment between the coordinator and one rank (session 10, A2). docs/MEASUREMENT.md section 2.
The coordinator sends ROUNDS requests on the ctl link; the rank answers each with its wall clock and its monotonic clock.
The round with the smallest round trip is kept: offset = remote minus local minus half the round trip (the SNTP formula
with a zero server hold time). Two offsets come out: monotonic (aligns recorder events) and wall (compared with the OS
time sync). The error is at most half the round trip. Measured at the start and at the end of a run; an event time is
converted with the offset interpolated between the two, which removes a constant drift."""
import struct
import time

from . import transport

ROUNDS = 16
REQ, REPLY, DONE, BEGIN = 0, 1, 2, 3
_PAIR = struct.Struct("<qq")
DISAGREE_NS = 2_000_000


def stamp() -> bytes:
    return _PAIR.pack(time.time_ns(), time.monotonic_ns())


def estimate(samples: list[tuple[int, int, int, int, int]]) -> dict:
    """samples: (local monotonic at send, local monotonic at receive, local wall at send, remote wall, remote monotonic),
    all ns. Returns the offsets (remote minus local) of the round with the smallest round trip."""
    t0, t1, w0, rw, rm = min(samples, key=lambda s: s[1] - s[0])
    rtt = t1 - t0
    rtts = sorted(s[1] - s[0] for s in samples)
    return {"offset_mono_ns": rm - (t0 + rtt // 2), "offset_wall_ns": rw - (w0 + rtt // 2), "rtt_ns": rtt,
            "rtt_median_ns": rtts[len(rtts) // 2], "rounds": len(samples), "at_mono_ns": t0 + rtt // 2, "at_wall_ns": w0 + rtt // 2}


def measure(link: transport.Link, rounds: int = ROUNDS) -> dict:
    """Coordinator side. The rank must be in answer()."""
    samples = []
    for i in range(rounds):
        w0, t0 = time.time_ns(), time.monotonic_ns()
        link.send(transport.KIND_CLOCK, aux0=i, aux1=REQ)
        kind, payload, *_ = link.recv()
        t1 = time.monotonic_ns()
        if kind != transport.KIND_CLOCK:
            raise RuntimeError(f"expected a clock reply, got {transport.KIND_NAMES.get(kind, kind)}")
        rw, rm = _PAIR.unpack(bytes(payload))
        samples.append((t0, t1, w0, rw, rm))
    link.send(transport.KIND_CLOCK, aux1=DONE)
    return estimate(samples)


def answer(link: transport.Link) -> None:
    """Rank side: answer requests until DONE."""
    while True:
        kind, _, _, _, aux0, aux1 = link.recv()
        if kind != transport.KIND_CLOCK:
            raise RuntimeError(f"expected a clock frame, got {transport.KIND_NAMES.get(kind, kind)}")
        if aux1 == DONE:
            return
        link.send(transport.KIND_CLOCK, stamp(), "uint8", aux0=aux0, aux1=REPLY)


ZERO = {"offset_mono_ns": 0, "offset_wall_ns": 0, "rtt_ns": 0, "rtt_median_ns": 0, "rounds": 0, "at_mono_ns": 0, "at_wall_ns": 0}


def offset_at(t_remote_ns: int, start: dict, end: dict | None) -> float:
    """Monotonic offset (remote minus coordinator) at a remote time, interpolated between the two measurements."""
    o0 = start["offset_mono_ns"]
    if not end or end["at_mono_ns"] == start["at_mono_ns"]:
        return o0
    o1 = end["offset_mono_ns"]
    r0, r1 = start["at_mono_ns"] + o0, end["at_mono_ns"] + o1
    return o0 + (o1 - o0) * (t_remote_ns - r0) / (r1 - r0)


def to_coordinator(t_remote_ns: int, start: dict, end: dict | None = None) -> float:
    """A rank's monotonic time on the coordinator's monotonic clock."""
    return t_remote_ns - offset_at(t_remote_ns, start, end)


def drift(start: dict, end: dict | None) -> dict | None:
    """Change of the offset between the two measurements, and as parts per million of the time between them."""
    if not end:
        return None
    d, span = end["offset_mono_ns"] - start["offset_mono_ns"], end["at_mono_ns"] - start["at_mono_ns"]
    return {"drift_ns": d, "span_s": span / 1e9, "ppm": d / span * 1e6 if span else None,
            "wall_minus_mono_change_ns": (end["offset_wall_ns"] - end["offset_mono_ns"]) - (start["offset_wall_ns"] - start["offset_mono_ns"])}


def compare_with_os(engine_wall_offset_ns: int, coord_sntp_offset_s: float | None, rank_sntp_offset_s: float | None,
                    limit_ns: int = DISAGREE_NS) -> dict:
    """The engine measures rank wall minus coordinator wall. The OS tool measured server minus host on each host, so
    it predicts (server - coordinator) - (server - rank). Reports the gap and whether it is over the limit."""
    if coord_sntp_offset_s is None or rank_sntp_offset_s is None:
        return {"os_predicted_ns": None, "gap_ns": None, "disagree": None}
    pred = round((coord_sntp_offset_s - rank_sntp_offset_s) * 1e9)
    gap = engine_wall_offset_ns - pred
    return {"os_predicted_ns": pred, "gap_ns": gap, "disagree": abs(gap) > limit_ns}
