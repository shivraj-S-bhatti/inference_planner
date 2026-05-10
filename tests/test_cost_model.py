import unittest

from phaseforge.cost_model import HardwareProfile, estimate_node
from phaseforge.graph import WorkNode


def gpu() -> HardwareProfile:
    return HardwareProfile(
        name="gpu",
        kind="gpu",
        peak_tflops=1000,
        memory_bandwidth_tb_s=3,
        memory_capacity_gb=80,
        network_bandwidth_gb_s=100,
        network_latency_ms=0.01,
        power_watts=700,
        hourly_cost_usd=4,
        supported_ops=frozenset({"prefill", "decode"}),
    )


class CostModelTests(unittest.TestCase):
    def test_decode_gets_slower_with_more_output_tokens(self):
        small = WorkNode("decode_small", "decode", 7, 100, 10, 1, 2, 0.0, ())
        large = WorkNode("decode_large", "decode", 7, 100, 50, 1, 2, 0.0, ())

        self.assertGreater(estimate_node(large, gpu()).latency_ms, estimate_node(small, gpu()).latency_ms)

    def test_capacity_failure_is_explicit(self):
        tiny = HardwareProfile(
            name="tiny",
            kind="gpu",
            peak_tflops=1000,
            memory_bandwidth_tb_s=3,
            memory_capacity_gb=1,
            network_bandwidth_gb_s=100,
            network_latency_ms=0.01,
            power_watts=700,
            hourly_cost_usd=4,
            supported_ops=frozenset({"prefill"}),
        )
        node = WorkNode("prefill", "prefill", 70, 1000, 1, 1, 2, 0.0, ())

        estimate = estimate_node(node, tiny)
        self.assertFalse(estimate.feasible)
        self.assertEqual(estimate.bottleneck, "capacity")
