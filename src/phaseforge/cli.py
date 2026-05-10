from __future__ import annotations

import argparse

from phaseforge.cost_model import load_hardware
from phaseforge.graph import load_workload
from phaseforge.kernels import run_rmsnorm_benchmark
from phaseforge.planner import enumerate_plans, pareto_frontier
from phaseforge.reporting import write_plan_reports, write_spec_reports
from phaseforge.spec_decode import speculative_sweep


def main() -> None:
    parser = argparse.ArgumentParser(prog="phaseforge")
    subcommands = parser.add_subparsers(dest="command", required=True)

    plan = subcommands.add_parser("plan", help="plan heterogeneous placement for a workload graph")
    plan.add_argument("--workload", required=True)
    plan.add_argument("--hardware", required=True)
    plan.add_argument("--out", required=True)
    plan.add_argument("--limit", type=int, default=512)

    speculate = subcommands.add_parser("speculate", help="sweep speculative decoding placements")
    speculate.add_argument("--hardware", required=True)
    speculate.add_argument("--out", required=True)

    kernel = subcommands.add_parser("kernel-bench", help="benchmark a fused Triton transformer-adjacent kernel")
    kernel.add_argument("--size", type=int, default=4096)
    kernel.add_argument("--features", type=int, default=4096)
    kernel.add_argument("--out", required=True)

    args = parser.parse_args()

    if args.command == "plan":
        workload = load_workload(args.workload)
        hardware = load_hardware(args.hardware)
        plans = pareto_frontier(enumerate_plans(workload, hardware, limit=args.limit))
        write_plan_reports(plans, args.out)
        print(f"wrote {len(plans)} Pareto plans to {args.out}.[json|csv|md]")
    elif args.command == "speculate":
        hardware = load_hardware(args.hardware)
        by_kind = {device.kind: device for device in hardware}
        rows = speculative_sweep(gpu=by_kind["gpu"], sram=by_kind["sram_accelerator"])
        write_spec_reports(rows, args.out)
        print(f"wrote {len(rows)} speculative decoding rows to {args.out}.[json|csv|md]")
    elif args.command == "kernel-bench":
        results = run_rmsnorm_benchmark(args.size, args.features, args.out)
        print(f"wrote {len(results)} kernel benchmark rows to {args.out}.[json|md]")


if __name__ == "__main__":
    main()
