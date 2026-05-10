from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from phaseforge.planner import Plan
from phaseforge.spec_decode import SpecDecodeResult


def write_plan_reports(plans: Iterable[Plan], out_prefix: str | Path) -> None:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plans = list(plans)

    json_payload = [plan_to_dict(plan) for plan in plans]
    prefix.with_suffix(".json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "latency_ms",
                "energy_j",
                "cost_usd",
                "meets_sla",
                "transfer_ms",
                "assignments",
            ],
        )
        writer.writeheader()
        for plan in plans:
            writer.writerow(
                {
                    "name": plan.name,
                    "latency_ms": f"{plan.latency_ms:.4f}",
                    "energy_j": f"{plan.energy_j:.4f}",
                    "cost_usd": f"{plan.cost_usd:.8f}",
                    "meets_sla": plan.meets_sla,
                    "transfer_ms": f"{plan.transfer_ms:.4f}",
                    "assignments": json.dumps(plan.assignments, sort_keys=True),
                }
            )

    prefix.with_suffix(".md").write_text(render_plan_markdown(plans), encoding="utf-8")


def write_spec_reports(rows: Iterable[SpecDecodeResult], out_prefix: str | Path) -> None:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    prefix.with_suffix(".json").write_text(
        json.dumps([row.__dict__ for row in rows], indent=2),
        encoding="utf-8",
    )

    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SpecDecodeResult.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    prefix.with_suffix(".md").write_text(render_spec_markdown(rows), encoding="utf-8")


def plan_to_dict(plan: Plan) -> dict[str, object]:
    return {
        "name": plan.name,
        "assignments": plan.assignments,
        "latency_ms": plan.latency_ms,
        "energy_j": plan.energy_j,
        "cost_usd": plan.cost_usd,
        "meets_sla": plan.meets_sla,
        "transfer_ms": plan.transfer_ms,
        "explanation": list(plan.explanation),
        "nodes": {
            name: {
                "device": estimate.device,
                "latency_ms": estimate.latency_ms,
                "energy_j": estimate.energy_j,
                "resident_gb": estimate.resident_gb,
                "working_set_gb": estimate.working_set_gb,
                "bottleneck": estimate.bottleneck,
            }
            for name, estimate in plan.estimates.items()
        },
    }


def render_plan_markdown(plans: list[Plan]) -> str:
    lines = [
        "# Heterogeneous Placement Report",
        "",
        "| rank | plan | latency ms | energy J | transfer ms | SLA | assignments |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for rank, plan in enumerate(plans[:10], start=1):
        assignments = ", ".join(f"{node}->{device}" for node, device in plan.assignments.items())
        lines.append(
            f"| {rank} | {plan.name} | {plan.latency_ms:.2f} | {plan.energy_j:.2f} | "
            f"{plan.transfer_ms:.2f} | {'yes' if plan.meets_sla else 'no'} | {assignments} |"
        )
    lines.extend(["", "## Best Plan Explanation", ""])
    if plans:
        lines.extend(f"- {item}" for item in plans[0].explanation)
    lines.append("")
    return "\n".join(lines)


def render_spec_markdown(rows: list[SpecDecodeResult]) -> str:
    lines = [
        "# Speculative Decoding Sweep",
        "",
        "| rank | scenario | draft | accept p | expected accepted | latency ms | tok/s | joules |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows[:20], start=1):
        lines.append(
            f"| {rank} | {row.scenario} | {row.draft_tokens} | "
            f"{row.per_token_acceptance:.2f} | {row.expected_accepted_tokens:.2f} | "
            f"{row.request_latency_ms:.2f} | {row.tokens_per_second:.2f} | {row.joules:.2f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if rows:
        lines.append(f"- fastest scenario: {rows[0].scenario}")
        lines.append(f"- why: {rows[0].explanation}")
    lines.append("")
    return "\n".join(lines)
