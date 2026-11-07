import pytest

from nocturne.bench import costmodel


def synth(S, B, rtt, bw, c=(30.0, 1.0, 1.0)):
    bytes_per_ms = bw * 1e6 / 8 / 1000
    ms = c[0] + c[1] * S * rtt + c[2] * B / bytes_per_ms
    return {"timing": {"decode_tps_median": 1000.0 / ms},
            "comm": {"rpc_commands_per_token": S, "bytes_per_token": B},
            "network": {"rtt_ms_median": rtt, "bw_mbps": bw}}


def test_fit_recovers_coefficients():
    recs = [synth(S, B, rtt, bw) for S in (1, 7) for B in (14336, 608256) for rtt in (2, 12) for bw in (250, 900)]
    f = costmodel.fit(recs)
    assert abs(f["c0_ms"] - 30.0) < 1e-6
    assert abs(f["c1_per_msg_rtt"] - 1.0) < 1e-6
    assert abs(f["c2_per_transfer"] - 1.0) < 1e-6
    assert f["r2"] > 0.999 and f["n"] == 16


def test_fit_skips_null_records():
    recs = [synth(7, 14336, 12, 250), synth(1, 608256, 2, 900), synth(7, 608256, 12, 900), synth(1, 14336, 2, 250)]
    recs.append({"timing": {"decode_tps_median": None}, "comm": {"rpc_commands_per_token": None,
                 "bytes_per_token": None}, "network": {"rtt_ms_median": None, "bw_mbps": None}})
    assert costmodel.fit(recs)["n"] == 4


def test_fit_needs_three():
    with pytest.raises(ValueError):
        costmodel.fit([synth(7, 14336, 12, 250)])


def test_fit_singular():
    recs = [synth(7, 14336, 12, 250)] * 5
    with pytest.raises(ValueError):
        costmodel.fit(recs)


def test_features_prefer_v3_fields_and_fit_engine():
    r = synth(7, 14336, 12, 250)
    assert costmodel.features(r)[1] == 1000.0 / r["timing"]["decode_tps_median"]
    r["timing"]["decode_ms_per_step_pooled"] = 120.0
    r["comm"]["blocking_syncs_per_step"] = 1.0
    r["network"]["ping_decode_ms"] = {"p50": 9.0, "p95": 20.0, "p99": 50.0, "n": 100}
    r["comm"]["bytes_per_step_decode"] = 40000.0
    x, t = costmodel.features(r)
    assert t == 120.0 and x[1] == 9.0 and abs(x[2] - 40000.0 / (250e6 / 8 / 1000)) < 1e-9
    # two TP engines on one network: C from the one-machine run, the rest attributed to the syncs
    def rec(ms, S, rtt=8.0, B=1e6, bw=300.0):
        return {"id": f"r{ms}", "timing": {"decode_ms_per_step_pooled": ms}, "comm": {"blocking_syncs_per_step": S, "bytes_per_step_decode": B},
                "network": {"ping_decode_ms": {"p50": rtt, "p95": rtt, "p99": rtt, "n": 1}, "bw_mbps": bw}}
    f = costmodel.fit_engine([rec(830.0, 57), rec(865.0, 57)], baseline_ms_per_step=70.0)
    assert f["c0_ms_fixed"] == 70.0 and f["n"] == 2 and f["r2"] is None
    assert abs(f["effective_latency_per_sync_ms"] - ((830 + 865) / 2 - 70) / 57) < 1e-6
    assert abs(f["per_record"][0]["latency_per_sync_ms"] - (830 - 70) / 57) < 1e-9
    # with variation in B/bw against S*rtt the two coefficients separate
    recs = [rec(70 + 10 * 57 * 8 / 8 * rtt / 8 + 2.0 * B / (bw * 1e6 / 8 / 1000), 57, rtt, B, bw) for rtt in (4, 8, 16) for B in (1e5, 1e6) for bw in (100, 300)]
    f = costmodel.fit_engine(recs, 70.0)
    assert abs(f["c1_per_sync_rtt"] - 10 / 8) < 1e-6 and abs(f["c2_per_transfer"] - 2.0) < 1e-6 and f["r2"] > 0.999
    with pytest.raises(ValueError):
        costmodel.fit_engine([rec(830.0, 57)], 70.0)
