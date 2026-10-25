"""Placement algorithms. Pure functions: place(peers, manifest, policy) -> Placement.

Placement.rpc           "ip:port,ip:port" in device order.
Placement.tensor_split  [float, ...] in the same order as llama.cpp lists devices: RPC workers first, local device last.

v0 manual        the user types the split (scripts/04_run_two_machines.sh, 08_r0.sh, 10_capacity.sh)
v1 memory        legal, memory-feasible split from reported memory; output layer on the requester (O5)
v2 availability  adds availability history and recovery cost (O5)
"""
