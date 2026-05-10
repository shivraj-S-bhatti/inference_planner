from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from phaseforge.graph import WorkNode


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    kind: str
    peak_tflops: float
    memory_bandwidth_tb_s: float
    memory_capacity_gb: float
    network_bandwidth_gb_s: float
    network_latency_ms: float
    power_watts: float
    hourly_cost_usd: float
    supported_ops: frozenset[str]


@dataclass(frozen=True)
class NodeEstimate:
    node: str
    device: str
    latency_ms: float
    energy_j: float
    cost_usd: float
    resident_gb: float
    working_set_gb: float
    bottleneck: str
    feasible: bool
    reason: str


def load_hardware(path: str | Path) -> list[HardwareProfile]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        HardwareProfile(
            name=raw["name"],
            kind=raw["kind"],
            peak_tflops=float(raw["peak_tflops"]),
            memory_bandwidth_tb_s=float(raw["memory_bandwidth_tb_s"]),
            memory_capacity_gb=float(raw["memory_capacity_gb"]),
            network_bandwidth_gb_s=float(raw["network_bandwidth_gb_s"]),
            network_latency_ms=float(raw["network_latency_ms"]),
            power_watts=float(raw["power_watts"]),
            hourly_cost_usd=float(raw["hourly_cost_usd"]),
            supported_ops=frozenset(raw["supported_ops"]),
        )
        for raw in payload["devices"]
    ]


def estimate_node(node: WorkNode, hardware: HardwareProfile) -> NodeEstimate:
    if node.op not in hardware.supported_ops:
        return NodeEstimate(
            node=node.name,
            device=hardware.name,
            latency_ms=float("inf"),
            energy_j=float("inf"),
            cost_usd=float("inf"),
            resident_gb=0.0,
            working_set_gb=0.0,
            bottleneck="unsupported",
            feasible=False,
            reason=f"{hardware.name} does not support op={node.op}",
        )

    resident_gb = model_resident_gb(node)
    working_set_gb = resident_gb + kv_cache_gb(node) + activation_gb(node)
    if working_set_gb > hardware.memory_capacity_gb:
        return NodeEstimate(
            node=node.name,
            device=hardware.name,
            latency_ms=float("inf"),
            energy_j=float("inf"),
            cost_usd=float("inf"),
            resident_gb=resident_gb,
            working_set_gb=working_set_gb,
            bottleneck="capacity",
            feasible=False,
            reason=(
                f"working set {working_set_gb:.2f}GB exceeds "
                f"{hardware.name} capacity {hardware.memory_capacity_gb:.2f}GB"
            ),
        )

    flops_t = phase_flops_t(node)
    memory_gb = phase_memory_gb(node)

    compute_util = utilization_for(node.op, hardware.kind, resource="compute")
    memory_util = utilization_for(node.op, hardware.kind, resource="memory")

    compute_ms = (flops_t / max(hardware.peak_tflops * compute_util, 1e-9)) * 1000.0
    memory_ms = (
        memory_gb / max(hardware.memory_bandwidth_tb_s * 1024.0 * memory_util, 1e-9)
    ) * 1000.0
    overhead_ms = op_overhead_ms(node.op, hardware.kind)
    latency_ms = max(compute_ms, memory_ms) + overhead_ms
    bottleneck = "compute" if compute_ms >= memory_ms else "memory_bandwidth"
    if node.op in {"tool", "retrieval", "postprocess"}:
        bottleneck = "network_or_cpu"

    seconds = latency_ms / 1000.0
    energy_j = hardware.power_watts * seconds
    cost_usd = hardware.hourly_cost_usd * seconds / 3600.0

    return NodeEstimate(
        node=node.name,
        device=hardware.name,
        latency_ms=latency_ms,
        energy_j=energy_j,
        cost_usd=cost_usd,
        resident_gb=resident_gb,
        working_set_gb=working_set_gb,
        bottleneck=bottleneck,
        feasible=True,
        reason="ok",
    )


def model_resident_gb(node: WorkNode) -> float:
    return node.model_billion_params * node.dtype_bytes / node.partitions


def kv_cache_gb(node: WorkNode) -> float:
    if node.model_billion_params <= 0 or node.op in {"retrieval", "tool", "postprocess"}:
        return 0.0
    bytes_total = (
        2
        * node.layers
        * node.hidden_size
        * max(node.total_tokens, 1)
        * node.batch_size
        * node.dtype_bytes
    )
    return bytes_total / 1e9 / node.partitions


def activation_gb(node: WorkNode) -> float:
    if node.op in {"retrieval", "tool", "postprocess"}:
        return max(node.transfer_gb, 0.001)
    return max(node.batch_size * node.input_tokens * node.hidden_size * node.dtype_bytes / 1e9, 0.001)


def phase_flops_t(node: WorkNode) -> float:
    params = node.model_billion_params * 1e9
    batch = node.batch_size
    if node.op == "prefill":
        return 2.0 * params * max(node.input_tokens, 1) * batch / 1e12 / node.partitions
    if node.op == "verify":
        return 2.0 * params * max(node.output_tokens, 1) * batch / 1e12 / node.partitions
    if node.op in {"decode", "draft"}:
        return 2.0 * params * max(node.output_tokens, 1) * batch / 1e12 / node.partitions
    if node.op in {"embedding", "rerank"}:
        return max(node.input_tokens, 1) * 0.002
    return max(node.input_tokens + node.output_tokens, 1) * 0.00002


def phase_memory_gb(node: WorkNode) -> float:
    resident = model_resident_gb(node)
    kv = kv_cache_gb(node)
    if node.op == "prefill":
        return resident + 0.35 * kv + activation_gb(node)
    if node.op == "verify":
        return resident + 0.20 * kv + activation_gb(node)
    if node.op in {"decode", "draft"}:
        # Autoregressive phases repeatedly touch weights and KV state token-by-token.
        return max(node.output_tokens, 1) * (resident + 0.08 * kv)
    if node.op in {"retrieval", "tool", "postprocess"}:
        return max(node.transfer_gb, 0.001)
    return resident + kv


def utilization_for(op: str, hardware_kind: str, resource: str) -> float:
    table = {
        ("prefill", "gpu", "compute"): 0.58,
        ("prefill", "gpu", "memory"): 0.62,
        ("verify", "gpu", "compute"): 0.50,
        ("verify", "gpu", "memory"): 0.55,
        ("decode", "gpu", "compute"): 0.18,
        ("decode", "gpu", "memory"): 0.42,
        ("draft", "gpu", "compute"): 0.18,
        ("draft", "gpu", "memory"): 0.42,
        ("draft", "sram_accelerator", "compute"): 0.25,
        ("draft", "sram_accelerator", "memory"): 0.70,
        ("decode", "sram_accelerator", "compute"): 0.25,
        ("decode", "sram_accelerator", "memory"): 0.70,
    }
    return table.get((op, hardware_kind, resource), 0.35 if resource == "compute" else 0.40)


def op_overhead_ms(op: str, hardware_kind: str) -> float:
    if op in {"tool", "retrieval"}:
        return 12.0 if hardware_kind == "cpu" else 20.0
    if hardware_kind == "sram_accelerator":
        return 0.08
    if hardware_kind == "gpu":
        return 0.12
    return 0.30


def transfer_latency_ms(size_gb: float, src: HardwareProfile, dst: HardwareProfile) -> float:
    if src.name == dst.name or size_gb <= 0:
        return 0.0
    bandwidth = min(src.network_bandwidth_gb_s, dst.network_bandwidth_gb_s)
    latency = src.network_latency_ms + dst.network_latency_ms
    return latency + (size_gb / max(bandwidth, 1e-9)) * 1000.0
