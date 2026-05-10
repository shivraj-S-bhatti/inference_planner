from __future__ import annotations

from dataclasses import dataclass

from phaseforge.cost_model import HardwareProfile


@dataclass(frozen=True)
class SpecDecodeResult:
    scenario: str
    draft_tokens: int
    per_token_acceptance: float
    expected_accepted_tokens: float
    request_latency_ms: float
    tokens_per_second: float
    joules: float
    explanation: str


def expected_accepted_tokens(draft_tokens: int, per_token_acceptance: float) -> float:
    if draft_tokens <= 0:
        return 1.0
    p = max(0.0, min(per_token_acceptance, 0.999999))
    return sum(p ** i for i in range(1, draft_tokens + 1))


def speculative_sweep(
    gpu: HardwareProfile,
    sram: HardwareProfile,
    input_tokens: int = 8000,
    output_tokens: int = 1000,
    draft_lengths: tuple[int, ...] = (4, 8, 16, 32),
    acceptance_rates: tuple[float, ...] = (0.90, 0.93, 0.96),
) -> list[SpecDecodeResult]:
    rows: list[SpecDecodeResult] = []
    prefill_ms = input_tokens * 0.028
    gpu_decode_ms = output_tokens * 5.5

    rows.append(
        SpecDecodeResult(
            scenario="gpu_prefill_gpu_decode",
            draft_tokens=0,
            per_token_acceptance=1.0,
            expected_accepted_tokens=1.0,
            request_latency_ms=prefill_ms + gpu_decode_ms,
            tokens_per_second=output_tokens / ((prefill_ms + gpu_decode_ms) / 1000.0),
            joules=gpu.power_watts * (prefill_ms + gpu_decode_ms) / 1000.0,
            explanation="baseline prefill/decode on GPU; decode is memory-bandwidth limited",
        )
    )

    for draft_len in draft_lengths:
        for acceptance in acceptance_rates:
            expected = expected_accepted_tokens(draft_len, acceptance)
            verify_ms = 0.85 * draft_len
            gpu_draft_ms = 1.65 * draft_len
            sram_draft_ms = 0.22 * draft_len
            transfer_ms = 0.18

            gpu_cycle_ms = gpu_draft_ms + verify_ms
            sram_cycle_ms = sram_draft_ms + verify_ms + transfer_ms

            gpu_latency = prefill_ms + output_tokens * gpu_cycle_ms / expected
            sram_latency = prefill_ms + output_tokens * sram_cycle_ms / expected

            rows.append(
                SpecDecodeResult(
                    scenario="gpu_spec_decode",
                    draft_tokens=draft_len,
                    per_token_acceptance=acceptance,
                    expected_accepted_tokens=expected,
                    request_latency_ms=gpu_latency,
                    tokens_per_second=output_tokens / (gpu_latency / 1000.0),
                    joules=gpu.power_watts * gpu_latency / 1000.0,
                    explanation="draft and verify on GPU; faster than naive decode when acceptance is high",
                )
            )
            rows.append(
                SpecDecodeResult(
                    scenario="hetero_sram_draft_gpu_verify",
                    draft_tokens=draft_len,
                    per_token_acceptance=acceptance,
                    expected_accepted_tokens=expected,
                    request_latency_ms=sram_latency,
                    tokens_per_second=output_tokens / (sram_latency / 1000.0),
                    joules=(
                        gpu.power_watts * (prefill_ms + output_tokens * verify_ms / expected) / 1000.0
                        + sram.power_watts * (output_tokens * sram_draft_ms / expected) / 1000.0
                    ),
                    explanation="draft on SRAM-like accelerator, prefill/verify on GPU; wins when transfer is small",
                )
            )
    return sorted(rows, key=lambda row: row.request_latency_ms)
