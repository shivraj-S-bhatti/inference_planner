# Heterogeneous Placement Report

| rank | plan | latency ms | energy J | transfer ms | SLA | assignments |
|---:|---|---:|---:|---:|---|---|
| 1 | plan_0001 | 909.31 | 605.67 | 1.78 | yes | retrieve_context->cpu_epyc_like, prefill_target->gpu_h100_like, draft_tokens->sram_dual_card_decode_accel, verify_tokens->gpu_h100_like, tool_call->cpu_epyc_like, postprocess_answer->cpu_epyc_like |

## Best Plan Explanation

- uses 3 device type(s): cpu_epyc_like, gpu_h100_like, sram_dual_card_decode_accel
- critical path 909.31 ms with 1.78 ms transfer overhead
- slowest phase is prefill_target on gpu_h100_like (compute)
- meets SLA

## Why This Plan Wins

The planner keeps compute-heavy target-model prefill and verification on the
GPU, moves network/CPU-heavy retrieval and tool execution to the CPU profile,
and places the small speculative draft phase on the SRAM-like accelerator. The
draft phase is memory-bandwidth sensitive and fits in the dual-card SRAM profile,
so the transfer overhead is lower than the latency saved by avoiding GPU decode
pressure.
