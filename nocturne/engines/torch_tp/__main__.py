"""CLI. worker: rank >= 1 service bound to a chosen address (never 0.0.0.0). serve: rank 0 with HTTP on 127.0.0.1.
compare: greedy output against transformers generate (stage 1 and 2 of the B1t plan).
  python -m nocturne.engines.torch_tp worker --bind <addr> --port 50053 --model <dir> --rank 1 --world 2
  python -m nocturne.engines.torch_tp serve --model <dir> --world 2 --peer <addr>:50053 --port 8092 --max-seq-len 4096
  python -m nocturne.engines.torch_tp compare --model <dir> --world 1 --n 20
  python -m nocturne.engines.torch_tp.microbench listen|run ...   (transport exchange benchmark, no model)"""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--bind", required=True); w.add_argument("--port", type=int, default=50053)
    w.add_argument("--model", required=True); w.add_argument("--rank", type=int, default=1); w.add_argument("--world", type=int, default=2)
    w.add_argument("--device")
    s = sub.add_parser("serve")
    s.add_argument("--model", required=True); s.add_argument("--world", type=int, default=1); s.add_argument("--peer")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8092)
    s.add_argument("--max-seq-len", type=int, default=4096); s.add_argument("--device")
    s.add_argument("--trace-allreduce", action="store_true", help="per stage timing of every all-reduce, summary in /metrics")
    s.add_argument("--reduce-dtype", choices=["bfloat16", "float32"], default="bfloat16",
                   help="where the two partials are summed: bf16 add, or fp32 add rounded once (both ranks follow)")
    s.add_argument("--progress-every", type=int, default=16, help="a decode log line every N tokens (0: none)")
    c = sub.add_parser("compare")
    c.add_argument("--model", required=True); c.add_argument("--world", type=int, default=1); c.add_argument("--peer")
    c.add_argument("--n", type=int, default=20); c.add_argument("--max-tokens", type=int, default=64)
    c.add_argument("--max-seq-len", type=int, default=4096); c.add_argument("--device")
    a = ap.parse_args(argv)
    if a.cmd == "worker":
        from .engine import serve_rank
        serve_rank(a.model, a.rank, a.world, a.bind, a.port, a.device)
        return 0
    if a.cmd == "serve":
        from transformers import AutoTokenizer
        from .engine import Rank0
        from .server import serve
        eng = Rank0(a.model, a.world, a.max_seq_len, a.device, a.peer, trace_allreduce=a.trace_allreduce,
                    reduce_dtype=a.reduce_dtype, progress_every=a.progress_every)
        tok = AutoTokenizer.from_pretrained(a.model)
        srv = serve(eng, tok, a.host, a.port, model_name=a.model.rstrip("/").rsplit("/", 1)[-1])
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            eng.shutdown()
        return 0
    if a.cmd == "compare":
        from .compare import run
        return run(a.model, a.world, a.peer, a.n, a.max_tokens, a.device, a.max_seq_len)


if __name__ == "__main__":
    sys.exit(main())
