import unittest

from phaseforge.cost_model import HardwareProfile
from phaseforge.spec_decode import expected_accepted_tokens, speculative_sweep


class SpecDecodeTests(unittest.TestCase):
    def test_expected_accepted_tokens_increases_with_draft_length(self):
        self.assertGreater(expected_accepted_tokens(8, 0.9), expected_accepted_tokens(4, 0.9))

    def test_speculative_sweep_includes_heterogeneous_case(self):
        gpu = HardwareProfile("gpu", "gpu", 1000, 3, 80, 100, 0.01, 700, 4, frozenset())
        sram = HardwareProfile("sram", "sram_accelerator", 100, 150, 2, 100, 0.01, 150, 1, frozenset())

        rows = speculative_sweep(gpu, sram, draft_lengths=(4,), acceptance_rates=(0.95,))

        self.assertEqual(
            {row.scenario for row in rows},
            {
                "gpu_prefill_gpu_decode",
                "gpu_spec_decode",
                "hetero_sram_draft_gpu_verify",
            },
        )
