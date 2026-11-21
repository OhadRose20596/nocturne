"""Mutation check (V2): break the engine code on purpose, one change at a time, and see which test fails first.
Usage (on a scratch branch, then throw the branch away): python mutate.py <out.json>
Each mutation edits one file, runs a pytest selection, records the first failing test, and restores the file from git."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = str(Path.home() / "dev/nocturne-torch/bin/python")
E = "nocturne/engines/torch_tp/"
FAST = ["tests/bench/test_engine_gguf.py", "-k", "random_blocks or real_tensors_match_gguf_py or chunked"]
MODELS = "tests/bench/test_engine_gguf_models.py"

# name, file, old, new, pytest args
MUTATIONS = [
    ("Q4_K: swap the two halves of the scale and min unpacking", E + "quant.py",
     "scales = torch.cat([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)], dim=-1)",
     "scales = torch.cat([(md & 0x0F) | ((d >> 2) & 0x30), d & 0x3F], dim=-1)", FAST),
    ("Q8_0: drop the block scale", E + "quant.py",
     "return blocks[:, 2:].view(torch.int8).to(torch.float32) * _f16(blocks[:, :2])",
     "return blocks[:, 2:].view(torch.int8).to(torch.float32)", FAST),
    ("Q4_0: offset 7 instead of 8", E + "quant.py", "q = _nibbles(blocks[:, 2:], 1, 16).reshape(n, 32).to(torch.int8) - 8",
     "q = _nibbles(blocks[:, 2:], 1, 16).reshape(n, 32).to(torch.int8) - 7", FAST),
    ("Q5_0: high bit shifted by 3", E + "quant.py", "q = (ql | (bits << 4)) - 16", "q = (ql | (bits << 3)) - 16", FAST),
    ("Q5_K: high bit shifted by 3", E + "quant.py", "q = (_nibbles(blocks[:, 48:176], 4, 32) | (hi << 4)).to(torch.float32)",
     "q = (_nibbles(blocks[:, 48:176], 4, 32) | (hi << 3)).to(torch.float32)", FAST),
    ("Q4_K: add the min instead of subtracting it", E + "quant.py", "return (dd * q - dm).reshape(n, QK_K)",
     "return (dd * q + dm).reshape(n, QK_K)", FAST),
    ("Q6_K: bias 31 instead of 32", E + "quant.py", "q = ((ql | (hi << 4)).to(torch.int16) - 32).reshape(n, 16, 16).to(torch.float32)",
     "q = ((ql | (hi << 4)).to(torch.int16) - 31).reshape(n, 16, 16).to(torch.float32)", FAST),
    ("llama: skip the Q and K un-permute", E + "gguf_source.py", "return s.transpose(1, 2).reshape(-1)", "return s.reshape(-1)",
     [MODELS, "-k", "bf16_gguf_weights_equal or unpermute_agrees or permute_really"]),
    ("column slice of a row split layer: one block too many", E + "gguf_source.py", "data = data[:, c0 // bs:c1 // bs]",
     "data = data[:, c0 // bs:c1 // bs + 1]", [MODELS, "-k", "sharded_reads"]),
    ("planner: block alignment check off by one head", E + "plan.py", "(hb * d.head_dim) % u.o_block", "((hb + 1) * d.head_dim) % u.o_block",
     [MODELS, "-k", "rejects_unaligned or aligns_head"]),
    ("planner: MLP granule ignores the down_proj block", E + "plan.py", "Gm = G * u.down_block // math.gcd(G, u.down_block)", "Gm = G",
     [MODELS, "-k", "aligns_head"]),
    ("planner: no transient dequant buffer in the workspace", E + "plan.py", "ws += min(d.matrix_elems(nh, nkv, rows, vocab_rows), CHUNK_ELEMS) * CHUNK_WORKING_BYTES_PER_ELEM",
     "ws += 0", [MODELS, "-k", "transient"]),
    ("planner: count every layer at layer 0 quant type", E + "plan.py", "mean = lambda f: sum(f(i) for i in rng) / len(rng)",
     "mean = lambda f: f(l0)", [MODELS, "-k", "planned_weight_bytes_equal_loaded"]),
    ("qwen3: skip the per head q_norm and k_norm", E + "model.py", "if self.q_norm is not None:", "if False and self.q_norm is not None:",
     [MODELS, "-k", "matches_transformers and qwen3"]),
    ("llama: ignore the Llama 3 rope factors", E + "model.py", "if factors is not None:", "if False and factors is not None:",
     [MODELS, "-k", "matches_transformers and llama"]),
    ("qwen2: drop the q, k, v bias", E + "model.py", "self.wq, self.bq = g(\"self_attn.q_proj.weight\"), opt(\"self_attn.q_proj.bias\")",
     "self.wq, self.bq = g(\"self_attn.q_proj.weight\"), None", [MODELS, "-k", "matches_transformers and qwen2"]),
    ("GGUF config: head size from hidden // heads, ignoring attention.key_length", E + "gguf_source.py",
     'hd = int(self._num("attention.key_length", hidden // n_heads))', "hd = hidden // n_heads",
     ["tests/bench/test_engine_gguf.py", "-k", "name_map_covers"]),
]


def run(args):
    r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider", *args], cwd=REPO, capture_output=True, text=True)
    failed = [l for l in r.stdout.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")]
    return r.returncode, (failed[0] if failed else ""), r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


def main():
    out = []
    for name, rel, old, new, args in MUTATIONS:
        p = REPO / rel
        src = p.read_text()
        assert src.count(old) == 1, f"{name}: pattern found {src.count(old)} times in {rel}"
        p.write_text(src.replace(old, new))
        try:
            rc, first, tail = run(args)
        finally:
            subprocess.run(["git", "checkout", "--", rel], cwd=REPO, check=True)
        out.append({"mutation": name, "file": rel, "caught": rc != 0, "first_failure": first, "summary": tail})
        print(("CAUGHT " if rc else "MISSED ") + name + " -> " + first, flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    assert not subprocess.run(["git", "diff", "--quiet", "--", "nocturne"], cwd=REPO).returncode == 1, "tree not restored"


if __name__ == "__main__":
    main()
