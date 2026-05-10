# Heterogeneous Inference Planner

This is a systems project for reasoning about the modern inference stack:
agent workloads are graphs, different phases have different bottlenecks, and the
best placement is often heterogeneous rather than "put everything on the biggest
GPU."

It models four things that matter in production inference:

- **Phase behavior:** prefill is usually compute-heavy, decode and speculative
  drafting are memory-bandwidth sensitive, and tool calls are network/CPU bound.
- **Hardware fit:** devices expose different compute, memory bandwidth, memory
  capacity, power, cost, and supported operation sets.
- **Graph placement:** moving a stage to another device can help, but only if the
  latency savings beat transfer and scheduling overhead.
- **SLA tradeoffs:** the planner reports latency, energy, dollar cost, bottleneck
  reasons, and a Pareto frontier instead of a single magical score.

The intent is to scale answers for: **where is the data stuck, and what
hardware should own that phase?**

Agent workloads should be represented as graphs, 
partitioned into schedulable units, 
placed on heterogeneous hardware (according to compute, memory, network, cost, SLA)

Inspired by Gimlet Labs blogs on prefill/decode disaggregation, speculative
decoding on SRAM-centric accelerators, and cost-aware graph optimization for AI
workloads.

This repo implements a POC of the idea:

1. Parse an agent workload graph.
2. Estimate each node's compute, memory, transfer, and residency pressure.
3. Enumerate feasible placements across GPU, CPU, and SRAM-like accelerator
   profiles.
4. Report Pareto-optimal plans and explain the bottleneck for each stage.
5. Sweep speculative decoding configurations to show when heterogeneous draft
   placement is worth the transfer overhead.

## Thoughts:

The useful performance question is no longer "can I make one operator faster?"
That still matters, but the operator is only one island in a much larger system.
For an agent request, the user does not experience a GEMM. They experience a
chain: retrieval, prefill, decode, tool execution, verification, post-processing,
and often another loop through the same graph. Each stage stresses a different
resource.

Prefill is the easy place to lie to yourself. It looks like the whole problem
because it is big and expensive. It is compute-heavy, it responds well to GPUs,
and it gives clean benchmark numbers. But decode is a different animal. Decode
is sequential and memory-bandwidth sensitive. The GPU may have ridiculous peak
FLOPs and still spend the request dragging weights and KV state through memory
one token at a time. Then tool calls show up and break the mental model again:
they are often network-bound, CPU-friendly, and dominated by scheduling and
external latency rather than matrix math.

That is why homogeneous placement is a trap. If every phase goes to the same
device, the system is optimized for the average phase, and no phase is average.
A single accelerator can be excellent at prefill and mediocre at draft decode. A
CPU can be terrible for transformer inference and still be the right place for
retrieval orchestration. An SRAM-centric accelerator can be unusable for a giant
target model and still be the best place for a small draft model whose latency is
dominated by memory movement.

The hard part is not saying "use heterogeneous hardware." The hard part is
deciding when moving a phase actually wins. Offloading has a cost. You pay for
network transfer, synchronization, queueing, extra failure modes, and operational
complexity. A phase should move only when the latency, energy, or cost advantage
beats those penalties. This repo makes that tradeoff explicit.

The planner treats an agent as a graph. Nodes represent phases such as retrieval,
prefill, draft, verify, decode, and tool execution. Edges carry dependencies and
data movement. Hardware profiles expose peak compute, memory bandwidth, memory
capacity, power, cost, network bandwidth, and supported operation types. The
planner estimates each node's compute pressure, resident model memory,
KV-cache/activation pressure, transfer overhead, and dominant bottleneck. It then
enumerates feasible placements and reports a Pareto frontier rather than
pretending there is one universal "best" result.

This is deliberately not a production scheduler. It is a thinking tool. A good
research prototype should let someone inspect the model, argue with the
assumptions, and change the numbers without needing a six-rack lab. The point is
to make performance claims falsifiable. If the planner says an SRAM-like device
should own draft decoding, the report should show why: lower memory latency,
acceptable transfer overhead, expected accepted tokens from speculation, energy
use, and where the critical path moved.

The project also includes a small kernel track. Not because a toy kernel proves a
cloud, but because kernel-level evidence keeps the planner honest. You cannot
reason about heterogeneous execution if kernels are just black boxes. The Triton
example fuses residual addition and RMSNorm, a common transformer-adjacent memory
operation. It is intentionally narrow: one kernel, one benchmark, one comparison
against eager PyTorch. That is enough to demonstrate the layer below the planner
without turning the repo into a CUDA shrine.

TensorRT is not the center of this project. TensorRT is excellent for graph-level
optimization, engine building, quantization, and deployment on NVIDIA targets. It
would be a useful baseline in a later extension. But it is not the right first
artifact for this thesis, because the thesis is about phase-aware heterogeneous
placement, not simply lowering one graph to one runtime. Straight CUDA is also
not the first move. CUDA would be more explicit, but it would bury the project in
boilerplate before the systems idea is visible. Triton is the right middle layer:
it exposes tiling, memory traffic, and fusion while staying small enough to read.

The north star is simple: show where the data is stuck, show which device should
own that phase, and show the evidence.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

phaseforge plan \
  --workload examples/coding_agent_workload.json \
  --hardware configs/hardware_profiles.json \
  --out reports/coding_agent_plan

phaseforge speculate \
  --hardware configs/hardware_profiles.json \
  --out reports/spec_decode_sweep

python -m unittest discover -s tests
```

The first command writes:

- `reports/coding_agent_plan.json`
- `reports/coding_agent_plan.csv`
- `reports/coding_agent_plan.md`

The speculative decoding sweep writes:

- `reports/spec_decode_sweep.json`
- `reports/spec_decode_sweep.csv`
- `reports/spec_decode_sweep.md`

Committed examples:

- [`docs/sample_coding_agent_plan.md`](docs/sample_coding_agent_plan.md)
- [`docs/sample_spec_decode_sweep.md`](docs/sample_spec_decode_sweep.md)

Optional kernel experiment:

```bash
python -m pip install -e ".[kernel]"
phaseforge kernel-bench --size 4096 --features 4096 --out reports/kernel_rmsnorm
```

If CUDA/Triton is unavailable, the command exits with an explanatory message
instead of pretending a CPU fallback is comparable.

## Example Output

The planner compares placements such as:

- all phases on GPU
- retrieval and tools on CPU, prefill/verify on GPU
- speculative draft/decode on an SRAM-like accelerator
- mixed plans that trade energy for lower latency

Each plan includes:

- critical-path latency
- estimated joules
- estimated dollars
- per-stage device assignment
- per-stage dominant bottleneck
- transfer latency across device boundaries
- feasibility failures when a phase exceeds device memory or unsupported ops

## What Makes This Non-Trivial

Most toy inference demos stop at "measure latency." This one separates why the
latency exists:

- prefill can be compute-bound even when total model memory is large
- decode can be memory-bound because weights/KV state dominate token-by-token
  execution
- speculative decoding only helps if draft latency, acceptance rate, and verify
  cost line up
- offloading a phase can lose if network transfer or synchronization dominates
- cheaper hardware can be worse for SLA and better for throughput-per-watt

That is the actual systems judgment layer.

## Repository Layout

```text
configs/
  hardware_profiles.json       # GPU, CPU, SRAM-like accelerator profiles
examples/
  coding_agent_workload.json   # agent graph with retrieval, prefill, draft, verify, decode, tools
reports/
  .gitkeep
src/phaseforge/
  cli.py                       # command line interface
  cost_model.py                # phase/resource cost estimates
  graph.py                     # workload loading and topological checks
  kernels.py                   # optional Triton fused residual + RMSNorm benchmark
  planner.py                   # placement enumeration and Pareto frontier
  reporting.py                 # json/csv/markdown artifacts
  spec_decode.py               # speculative decoding sweep
tests/
  test_cost_model.py
  test_graph.py
  test_planner.py
  test_spec_decode.py
```

## Resume Bullet

```latex
\resumeBullet{Built a heterogeneous inference planner that models agent workloads as phase graphs, estimates compute/memory/network bottlenecks per node, and chooses Pareto-optimal placements across GPU, CPU, and SRAM-like accelerators; includes speculative decoding sweeps, KV-cache pressure modeling, and reproducible latency/energy/cost reports.}
```

## References

- Gimlet Labs, ["Designing infrastructure for running efficient AI workloads"](https://gimletlabs.ai/blog/heterogeneous-ai-infrastructure)
- Gimlet Labs, ["Low-Latency Inference with Speculative Decoding on d-Matrix Corsair and GPU"](https://gimletlabs.ai/blog/low-latency-spec-decode-corsair)
- Gimlet Labs, ["Efficient and Scalable Agentic AI with Heterogeneous Systems"](https://arxiv.org/pdf/2507.19635)
