# Nocturne. Current project context

Updated: 2026-10-25. This is a handoff note. It is not an ADR and not an experiment report. Older context is kept below.

## What changed on 2026-10-24 and 2026-10-25

1. Connectivity guide: docs/he/CONNECTIVITY.md. Wi-Fi stays the baseline. Wired links are diagnostic only. Internet without a VPN through port forwarding, UPnP or a relay on the tracker host. Recorded as ADR-009 (proposed). Security deferred by team decision.
2. Repository prepared for git. Scripts carry a two-line header and minimal comments. `.gitignore` keeps run evidence (`stderr.log`, `placement.log`) under `experiments/r0/runs` and `experiments/r3`, drops worker and smoke logs, and excludes the private working log `docs/he/SHARED_RESEARCH_AND_DECISIONS.md` and its HTML. Team decisions T1 to T3 are now in DECISIONS.md so the public repo is self-contained.
3. Documents aligned to v4: ARCHITECTURE.md sections 6 and 7, SETUP_GUIDE.md intro and section 9, README status, CLAUDE.md layout and next steps, env.example. Older operational docs (RUNBOOK_R0, GUIDE_R0_TO_R3, R0_PROTOCOL, STATUS, ONE_PAGER, PARTITIONING_DEEP_DIVE, BACKENDS) carry a status line that maps their internal names to O1 to O5. RESEARCH_PLAN.md moved to docs/archive/ (its R0 to R4 gates are referenced by ADR-007).
4. `scripts/08_r0.sh` accepts `LINK=thunderbolt|internet` and `OVERLAY=portforward|relay|tunnel`. `scripts/14_build_he_html.sh` skips private docs.
5. Git how-to for the team: docs/he/GIT_HOWTO.md.

## What changed on 2026-10-17

1. The supervisor's notes from 2026-10-11 are in docs/he/MEETING_CHEN_2026-10-11.md and as team decision T3 in the shared doc.
2. GOAL_AND_OBJECTIVES.md is now v4. The goal is one paragraph about peer-to-peer. Five short objectives come first: O1 single machine; O2 split by layers (PP); O3 scale out (more than two machines, internet, generic Hugging Face models); O4 performance study (PP vs TP, saturation, model to processor, trade-offs); O5 peer-to-peer application with liveness. The old v3.2 is in docs/archive/.
3. New documents, all proposed: FLOW.md (what happens when a model arrives, Mermaid diagram), PLATFORM.md (machines, software, GPU vs CPU and Metal vs CUDA as a review), BACKGROUND_AND_DESIGN.md (definitions, PP vs TP in practice, model format and manifest, the TP question).
4. Writing rule from the supervisor: short sentences, no long dash. Applies to new and rewritten Markdown.
5. Git was initialized in `nocturne/`. The first commit must run on a Mac. See docs/he/NEXT_STEPS_2026-10-17.md.
6. `scripts/10_capacity.sh` takes `SPLIT_MODE=tensor` to repeat the TP attempt with logs.

## Direction: T1, T2, T3

T1: latency is not a metric. T2: run a huge model over the internet on several home machines and succeed. T3: document structure and objectives per the supervisor. Full text in DECISIONS.md, "Team decisions".

## Open questions

- Q7 (new): why did `--split-mode tensor` fail on 32B between the two Macs. No log exists. The supervisor expects it to fit in memory. Repeat with evidence.
- From I017: quality protocol, interruption protocol, memory budget for long context. Not decided.
- ADR-007 (engine) is open. Decision due 15 Nov 2026.
- Supervisor to confirm v4, FLOW.md and the flow diagram.

## Documents not yet synced to v4 and T3

RESEARCH_PLAN.md, ARCHITECTURE.md, docs/he/GUIDE_R0_TO_R3.md, docs/he/ONE_PAGER_2026-10-11.md, CLAUDE.md "what to build next", the Word doc for the supervisor. Their technical evidence stands. Their objective numbering is the old one (O1 to O10). When they say O4 or O6, check the archive to map it.

## Earlier context (2026-10-11, Codex)

The project aims to enable a large model, beyond an individual participant's capacity under a declared memory policy, to run across volunteer consumer machines over the internet, complete a task, and deliver a high-quality answer. The "submit a task before sleep, receive an answer in the morning" example explains why interactive speed is not the priority. It is not a hard deadline and not the only use case. Network time and traffic are not optimization objectives. Existing timing and traffic measurements remain background evidence.

Priorities under T2: (1) capability and useful output, (2) reliability, (3) automatic planning. Memory planning must include weights, the total input plus generated context, KV cache, runtime buffers and headroom per machine.

Evidence: a two-Mac LAN pilot runs Qwen2.5-32B Q4 under the documented accelerator memory policy. It does not establish internet operation, long-context completion, fault recovery or better answer quality. R2 (the TP implementation experiment) has not been demonstrated. R3a/b/c have pilot evidence only. llama.cpp and layer splitting are the existing path. ADR-007 remains open. The requester coordinates its own job; the tracker has a discovery role.
