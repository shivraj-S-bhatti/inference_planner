from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class WorkNode:
    name: str
    op: str
    model_billion_params: float
    input_tokens: int
    output_tokens: int
    batch_size: int
    dtype_bytes: int
    transfer_gb: float
    deps: tuple[str, ...]
    layers: int = 32
    hidden_size: int = 4096
    partitions: int = 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Workload:
    name: str
    sla_ms: float
    nodes: tuple[WorkNode, ...]

    def by_name(self) -> dict[str, WorkNode]:
        return {node.name: node for node in self.nodes}


def load_workload(path: str | Path) -> Workload:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = []
    for raw in payload["nodes"]:
        nodes.append(
            WorkNode(
                name=raw["name"],
                op=raw["op"],
                model_billion_params=float(raw.get("model_billion_params", 0.0)),
                input_tokens=int(raw.get("input_tokens", 0)),
                output_tokens=int(raw.get("output_tokens", 0)),
                batch_size=int(raw.get("batch_size", 1)),
                dtype_bytes=int(raw.get("dtype_bytes", 2)),
                transfer_gb=float(raw.get("transfer_gb", 0.0)),
                deps=tuple(raw.get("deps", [])),
                layers=int(raw.get("layers", 32)),
                hidden_size=int(raw.get("hidden_size", 4096)),
                partitions=max(1, int(raw.get("partitions", 1))),
            )
        )
    workload = Workload(name=payload["name"], sla_ms=float(payload["sla_ms"]), nodes=tuple(nodes))
    topological_order(workload)
    return workload


def topological_order(workload: Workload) -> list[WorkNode]:
    by_name = workload.by_name()
    missing = {
        dep
        for node in workload.nodes
        for dep in node.deps
        if dep not in by_name
    }
    if missing:
        raise ValueError(f"unknown dependencies: {sorted(missing)}")

    temporary: set[str] = set()
    permanent: set[str] = set()
    ordered: list[WorkNode] = []

    def visit(node: WorkNode) -> None:
        if node.name in permanent:
            return
        if node.name in temporary:
            raise ValueError(f"cycle detected at {node.name}")
        temporary.add(node.name)
        for dep in node.deps:
            visit(by_name[dep])
        temporary.remove(node.name)
        permanent.add(node.name)
        ordered.append(node)

    for node in workload.nodes:
        visit(node)
    return ordered


def dependents(workload: Workload) -> dict[str, list[str]]:
    result = {node.name: [] for node in workload.nodes}
    for node in workload.nodes:
        for dep in node.deps:
            result[dep].append(node.name)
    return result


def node_names(nodes: Iterable[WorkNode]) -> list[str]:
    return [node.name for node in nodes]
