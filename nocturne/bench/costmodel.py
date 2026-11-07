"""Cost model over records. Per decode step: ms_per_step = C + c1 * S * rtt + c2 * B / bw_bytes_per_ms.
S is blocking syncs per step (serial waits), rtt the in-decode ping p50 (the pre-run probe when none), B the wire bytes
per decode step, C the engine's compute per step from its one-machine run (BASELINE on the requester). fit() is the
free three-coefficient least squares over any records (stdlib only); fit_engine() fixes C from the baseline and
reports the fitted effective latency per sync, c1 * rtt, per engine. Not a success criterion (T1)."""


def ms_per_step(rec: dict) -> float | None:
    t = rec["timing"]
    if t.get("decode_ms_per_step_pooled"):
        return t["decode_ms_per_step_pooled"]
    tps = t.get("decode_tps_median")
    return (1000.0 / tps) if tps else None


def syncs_per_step(rec: dict) -> float | None:
    c = rec["comm"]
    return c.get("blocking_syncs_per_step") or c.get("rpc_commands_per_token") or c.get("msgs_per_token_median")


def rtt_ms(rec: dict) -> float | None:
    n = rec["network"]
    p = n.get("ping_decode_ms") or {}
    return p.get("p50") or n.get("rtt_ms_median")


def bytes_per_step(rec: dict) -> float | None:
    c = rec["comm"]
    return c.get("bytes_per_step_decode") or c.get("bytes_per_token") or c.get("bytes_per_token_median")


def features(rec: dict):
    t, S, B, rtt, bw = ms_per_step(rec), syncs_per_step(rec), bytes_per_step(rec), rtt_ms(rec), rec["network"].get("bw_mbps")
    if None in (t, S, B, rtt, bw) or t <= 0 or bw <= 0:
        return None
    bytes_per_ms = bw * 1e6 / 8 / 1000
    return [1.0, S * rtt, B / bytes_per_ms], t


def _solve(ata, atb):
    """Gaussian elimination for a small square system."""
    n = len(atb)
    m = [row[:] + [b] for row, b in zip(ata, atb)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(m[r][i]))
        if abs(m[piv][i]) < 1e-12:
            raise ValueError("singular design matrix; need variation in S*rtt and B/bw")
        m[i], m[piv] = m[piv], m[i]
        for r in range(n):
            if r != i:
                f = m[r][i] / m[i][i]
                m[r] = [a - f * b for a, b in zip(m[r], m[i])]
    return [m[i][n] / m[i][i] for i in range(n)]


def _lstsq(X, y):
    k = len(X[0])
    ata = [[sum(X[r][i] * X[r][j] for r in range(len(X))) for j in range(k)] for i in range(k)]
    atb = [sum(X[r][i] * y[r] for r in range(len(X))) for i in range(k)]
    c = _solve(ata, atb)
    pred = [sum(ci * xi for ci, xi in zip(c, x)) for x in X]
    ybar = sum(y) / len(y)
    ss_res = sum((a - b) ** 2 for a, b in zip(y, pred))
    ss_tot = sum((a - ybar) ** 2 for a in y) or 1e-12
    return c, 1 - ss_res / ss_tot


def fit(records: list[dict]) -> dict:
    rows = [f for f in (features(r) for r in records) if f]
    if len(rows) < 3:
        raise ValueError(f"need at least 3 usable records, got {len(rows)}")
    c, r2 = _lstsq([x for x, _ in rows], [v for _, v in rows])
    return {"c0_ms": c[0], "c1_per_msg_rtt": c[1], "c2_per_transfer": c[2], "r2": r2, "n": len(rows)}


def fit_engine(records: list[dict], baseline_ms_per_step: float) -> dict:
    """C fixed to the engine's one-machine ms per step; fits c1 and c2 on ms - C. Also the per record effective latency
    per sync, (ms - C - c2 * B / bw) / S, which is what one blocking wait costs on this network and engine."""
    rows = [(f, r) for r in records for f in [features(r)] if f]
    if len(rows) < 2:
        raise ValueError(f"need at least 2 usable records, got {len(rows)}")
    X = [[x[1], x[2]] for (x, _), _ in rows]
    y = [t - baseline_ms_per_step for (_, t), _ in rows]
    try:
        c, r2 = _lstsq(X, y)
    except ValueError:
        # one network only: no variation in B/bw against S*rtt; attribute everything above C to the syncs
        c1 = sum(yi / xi[0] for xi, yi in zip(X, y)) / len(y)
        c, r2 = [c1, 0.0], None
    per_sync = []
    for (x, t), r in rows:
        S, rtt = syncs_per_step(r), rtt_ms(r)
        per_sync.append({"id": r.get("id"), "ms_per_step": t, "S": S, "rtt_ms": rtt,
                         "latency_per_sync_ms": (t - baseline_ms_per_step - c[1] * x[2]) / S if S else None})
    rtts = [p["rtt_ms"] for p in per_sync if p["rtt_ms"]]
    return {"c0_ms_fixed": baseline_ms_per_step, "c1_per_sync_rtt": c[0], "c2_per_transfer": c[1], "r2": r2, "n": len(rows),
            "effective_latency_per_sync_ms": c[0] * (sum(rtts) / len(rtts)) if rtts else None, "per_record": per_sync}
