"""Own PyTorch tensor parallel engine for Qwen2 (bench cell B1t, engine B spike of ADR-007 amendment 2).
Two ranks over our own TCP transport; rank 0 serves HTTP on 127.0.0.1. See nocturne/bench/README.md, row B."""
