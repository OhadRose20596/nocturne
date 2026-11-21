# Review pass 2 (fixes F1 to F9)

Scope: fixes.patch only. No pytest, no model load, no GPU. CPU snippets only.

## Per finding

- F1 fixed. `contents()` of a string array returns `list[str]` in this gguf-py (`gguf_reader.py`, ReaderField.contents), so `str(t)` is safe. `i` is a Python int. The result is still sorted and de-duplicated. Safetensors path untouched (config.py not in the patch). Added ids follow llama.cpp EOG-by-text; see New findings 1 and 2.
- F2 fixed. `plan` now builds `Dims.of(cfg, weight_meta(path))` and passes the Dims to `plan()` (accepts Dims: `d = cfg if isinstance(cfg, Dims)`) and to `format_plan`. `weight_meta` returns None for a directory, so safetensors behaves as before.
- F3 fixed. `linear` skips the copy when dense dtype equals x dtype, whole copy only up to 16M elements, else row chunks with `.to(x.dtype)`. Bias slice `bias[r0:r1]` is right, `torch.cat(dim=-1)` is right, dtype handling matches the old `weight()` path. Workspace term `min(elems, CHUNK) * 24` now bounds the dense chunk (at most 2 bytes bf16 per element plus the source view).
- F4 fixed (not in the requested list, but patched). Constant 14 to 24, matches the measured Q5_0 22.3. It raises every quant plan by up to 10 bytes per chunk element (about 168 MB at the cap); see New finding 4.
- F5 fixed. `round()` replaces `int()`. Random run of 200 mixed plans (Q4_K, Q5_K, Q6_K, Q8_0; L in 7, 13, 24, 28, 36; two ranks) ran without error. The new test asserts `abs(weights - want) <= 0.5`.
- F6 fixed. `mlp_row` uses `row(q.gate[i]) + row(q.up[i]) + down`. `QuantInfo.up` exists.
- F7 fixed. `load_tokenizer` takes a directory or a .gguf file; the default `tdir` (local_model_dir) can be a .gguf, which now works.
- F8 fixed. OOM during `.to(device)` sets `e.tensor_name`. `torch.cuda.OutOfMemoryError` exists on CPU-only torch too, so the except clause is safe.
- F9 fixed. All branches of `update_peers` checked by reading: new peer with known link gives `{wifi: ip}` or `{wired: ip}`; new peer with unknown link gives `{}`; existing `{...}` with items and unknown link: unchanged plus warning; existing `{}` and unknown link: stays `{}` plus warning; existing `{}` and known link: merged (merge_networks on empty items inserts at 0); no networks line: `{}` or `{key: ip}`; `networks: null` falls to the `else` branch (the for loop has no break) and is rewritten; block style mapping still exits with a message. Empty networks is safe for all consumers: run.py, preflight.py, calibrate_capture.py, adapters use `(w.get("networks") or {}).get(net) or w["bind"]`; llamacpp_rpc.py line 147 uses `worker.get("networks") or {}`.

## Bf16 Qwen2 regression check (question b)

- Planned bytes: loaded the old plan.py (git HEAD~1) next to the new one and ran Qwen2.5-7B shapes (hidden 3584, 28 layers, vocab 152064, untied) with weights (1,2), (1,1,1), (3,5). `weights`, `layer_bytes`, `vocab_bytes`, `workspace`, `total` are identical (True). Values stay `int`, because the non-quant Units are all ints, so `round` equals `int`.
- Tokens: no change in the safetensors path to config.py, model.py or loader.py (loader change is an added try/except). `linear` is only used for QuantLinear; `F.linear` is used for plain tensors.

## New findings

1. low. EOS by text can add an id that is not an end token for that model (a base model that has `<|im_end|>` or `<|eot_id|>` in its vocab, or a vocab where `<|end_of_text|>` appears as pad or bos in a converted model). Same rule as llama.cpp EOG, so the engine now stops where llama.cpp stops. Check: `GgufSource(p).config().eos_token_ids` on a Llama 3 base GGUF and on an Instruct GGUF, compare with generation_config.json.
2. low. The F1 test only checks the GGUF path when `~/models/gguf/Qwen2.5-1.5B-Instruct-Q8_0.gguf` exists. Otherwise it only reads a safetensors config (the `qwen3` bf16 branch), which the patch did not change. The tiny GGUF from `gguf_tiny` could be used: assert the ids of `<|im_end|>` and `<|endoftext|>` in its tokens are in `eos_token_ids`. Check: run the test on a machine without that file; it passes without exercising the new code.
3. low. docs/JOIN.md line 37 describes unknown link only for a peer that already has `networks`. The new behavior (new peer or empty `networks` stays `{}`, bind is used) is not written, and a new peer with unknown link prints no warning. Check: `ja.update_peers(PEERS, "newpeer", facts(ip, "unknown", ...), ...)` prints nothing.
4. low. CHUNK_WORKING_BYTES_PER_ELEM 24 is a global worst case (Q5_0), so Q8_0 and Q4_K plans are about 10 bytes per chunk element high (up to 168 MB at the cap). Safe direction, but a budget close to the limit may be refused. Check: planned workspace of a Q8_0 one-rank placement before and after.
5. low. The new byte test tolerates 0.5, not exact equality with the loader. A direct equality test against loaded bytes on a mixed file exists only for the tiny models (existing test), where layers have equal types. Check: build a tiny GGUF with `--tensor-type` overrides per layer and compare `planned["weights"]` with `nbytes` from `load_slices`.

Counts: high 0, medium 0, low 5.
