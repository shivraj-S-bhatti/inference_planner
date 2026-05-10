from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time


@dataclass(frozen=True)
class KernelBenchResult:
    backend: str
    size: int
    features: int
    median_ms: float
    gb_read_write: float
    effective_gb_s: float
    max_abs_error: float


def run_rmsnorm_benchmark(size: int, features: int, out_prefix: str | Path) -> list[KernelBenchResult]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for kernel-bench. Install with: pip install -e '.[kernel]'") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; skipping Triton kernel benchmark.")

    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("Triton is required for kernel-bench. Install with: pip install -e '.[kernel]'") from exc

    @triton.jit
    def fused_residual_rmsnorm(x, residual, weight, out, n_cols: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < n_cols
        base = row * n_cols + offsets
        vals = tl.load(x + base, mask=mask, other=0.0) + tl.load(residual + base, mask=mask, other=0.0)
        mean_square = tl.sum(vals * vals, axis=0) / n_cols
        inv_rms = tl.rsqrt(mean_square + eps)
        scaled = vals * inv_rms * tl.load(weight + offsets, mask=mask, other=0.0)
        tl.store(out + base, scaled, mask=mask)

    torch.manual_seed(7)
    x = torch.randn((size, features), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn((features,), device="cuda", dtype=torch.float16)
    out = torch.empty_like(x)

    def eager() -> torch.Tensor:
        y = x + residual
        rms = torch.sqrt(torch.mean(y.float() * y.float(), dim=-1, keepdim=True) + 1e-5)
        return (y.float() / rms * weight.float()).to(torch.float16)

    block = triton.next_power_of_2(features)
    if block > 131072:
        raise RuntimeError("features too large for single-block Triton RMSNorm demo")

    def triton_kernel() -> torch.Tensor:
        fused_residual_rmsnorm[(size,)](x, residual, weight, out, features, 1e-5, block)
        return out

    eager_out = eager()
    triton_out = triton_kernel()
    torch.cuda.synchronize()
    max_abs_error = float(torch.max(torch.abs(eager_out.float() - triton_out.float())).item())

    results = [
        measure("torch_eager", eager, size, features, max_abs_error),
        measure("triton_fused", triton_kernel, size, features, max_abs_error),
    ]

    write_kernel_report(results, out_prefix)
    return results


def measure(name: str, fn, size: int, features: int, max_abs_error: float) -> KernelBenchResult:
    import torch

    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(50):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    median_ms = timings[len(timings) // 2]

    # x read + residual read + weight read amortized + output write.
    bytes_moved = (size * features * 2 * 3) + (features * 2)
    gb_moved = bytes_moved / 1e9
    effective_gb_s = gb_moved / (median_ms / 1000.0)
    return KernelBenchResult(
        backend=name,
        size=size,
        features=features,
        median_ms=median_ms,
        gb_read_write=gb_moved,
        effective_gb_s=effective_gb_s,
        max_abs_error=max_abs_error,
    )


def write_kernel_report(results: list[KernelBenchResult], out_prefix: str | Path) -> None:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps([result.__dict__ for result in results], indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Triton Kernel Benchmark",
        "",
        "| backend | median ms | effective GB/s | max abs error |",
        "|---|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.backend} | {result.median_ms:.4f} | "
            f"{result.effective_gb_s:.2f} | {result.max_abs_error:.6f} |"
        )
    lines.append("")
    prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
