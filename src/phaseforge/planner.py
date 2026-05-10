from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from phaseforge.cost_model import (
    HardwareProfile,
    NodeEstimate,
    estimate_node,
    transfer_latency_ms,
)
from phaseforge.graph import Workload, topological_order


@dataclass(frozen=True)
class Plan:
    name: str
    assignments: dict[str, str]
    latency_ms: float
    energy_j: float
    cost_usd: float
    meets_sla: bool
    estimates: dict[str, NodeEstimate]
    transfer_ms: float
    explanation: tuple[str, ...]


def enumerate_plans(workload: Workload, hardware: list[HardwareProfile], limit: int = 512) -> list[Plan]:
    ordered = topological_order(workload)
    devices_by_name = {device.name: device for device in hardware}
    candidates: list[list[HardwareProfile]] = []
    for node in ordered:
        feasible_devices = [device for device in hardware if estimate_node(node, device).feasible]
        if not feasible_devices:
            raise ValueError(f"no feasible device for node {node.name}:{node.op}")
        candidates.append(feasible_devices)

    plans: list[Plan] = []
    for idx, choices in enumerate(product(*candidates)):
        if idx >= limit:
            break
        assignments = {node.name: device.name for node, device in zip(ordered, choices)}
        estimates = {
            node.name: estimate_node(node, devices_by_name[assignments[node.name]])
            for node in ordered
        }
        latency_ms, transfer_ms = critical_path_latency(workload, devices_by_name, assignments, estimates)
        energy_j = sum(estimate.energy_j for estimate in estimates.values())
        cost_usd = sum(estimate.cost_usd for estimate in estimates.values())
        explanation = explain_plan(workload, assignments, estimates, latency_ms, transfer_ms)
        plans.append(
            Plan(
                name=f"plan_{idx:04d}",
                assignments=assignments,
                latency_ms=latency_ms,
                energy_j=energy_j,
                cost_usd=cost_usd,
                meets_sla=latency_ms <= workload.sla_ms,
                estimates=estimates,
                transfer_ms=transfer_ms,
                explanation=tuple(explanation),
            )
        )

    return sorted(plans, key=lambda plan: (not plan.meets_sla, plan.latency_ms, plan.energy_j))


def critical_path_latency(
    workload: Workload,
    devices_by_name: dict[str, HardwareProfile],
    assignments: dict[str, str],
    estimates: dict[str, NodeEstimate],
) -> tuple[float, float]:
    by_name = workload.by_name()
    finish_ms: dict[str, float] = {}
    total_transfer = 0.0

    for node in topological_order(workload):
        dep_finish = 0.0
        for dep_name in node.deps:
            dep = by_name[dep_name]
            src = devices_by_name[assignments[dep_name]]
            dst = devices_by_name[assignments[node.name]]
            transfer = transfer_latency_ms(dep.transfer_gb, src, dst)
            total_transfer += transfer
            dep_finish = max(dep_finish, finish_ms[dep_name] + transfer)
        finish_ms[node.name] = dep_finish + estimates[node.name].latency_ms

    return max(finish_ms.values(), default=0.0), total_transfer


def pareto_frontier(plans: list[Plan]) -> list[Plan]:
    frontier: list[Plan] = []
    for plan in sorted(plans, key=lambda p: (p.latency_ms, p.energy_j, p.cost_usd)):
        dominated = False
        for other in frontier:
            if (
                other.latency_ms <= plan.latency_ms
                and other.energy_j <= plan.energy_j
                and other.cost_usd <= plan.cost_usd
                and (
                    other.latency_ms < plan.latency_ms
                    or other.energy_j < plan.energy_j
                    or other.cost_usd < plan.cost_usd
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(plan)
    return frontier


def explain_plan(
    workload: Workload,
    assignments: dict[str, str],
    estimates: dict[str, NodeEstimate],
    latency_ms: float,
    transfer_ms: float,
) -> list[str]:
    worst = max(estimates.values(), key=lambda estimate: estimate.latency_ms)
    devices = sorted(set(assignments.values()))
    return [
        f"uses {len(devices)} device type(s): {', '.join(devices)}",
        f"critical path {latency_ms:.2f} ms with {transfer_ms:.2f} ms transfer overhead",
        f"slowest phase is {worst.node} on {worst.device} ({worst.bottleneck})",
        "meets SLA" if latency_ms <= workload.sla_ms else "misses SLA",
    ]
