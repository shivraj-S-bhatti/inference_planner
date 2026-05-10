import unittest

from phaseforge.cost_model import HardwareProfile
from phaseforge.graph import WorkNode, Workload
from phaseforge.planner import enumerate_plans, pareto_frontier


class PlannerTests(unittest.TestCase):
    def test_planner_finds_mixed_assignment(self):
        workload = Workload(
            name="toy",
            sla_ms=1000,
            nodes=(
                WorkNode("prefill", "prefill", 1, 100, 1, 1, 2, 0.01, ()),
                WorkNode("tool", "tool", 0, 10, 10, 1, 2, 0.01, ("prefill",)),
            ),
        )
        gpu = HardwareProfile("gpu", "gpu", 1000, 3, 80, 100, 0.01, 700, 4, frozenset({"prefill"}))
        cpu = HardwareProfile("cpu", "cpu", 5, 0.4, 256, 100, 0.01, 200, 1, frozenset({"tool"}))

        plans = pareto_frontier(enumerate_plans(workload, [gpu, cpu]))

        self.assertTrue(plans)
        self.assertEqual(plans[0].assignments["prefill"], "gpu")
        self.assertEqual(plans[0].assignments["tool"], "cpu")

    def test_pareto_removes_dominated_plan(self):
        workload = Workload(
            name="toy",
            sla_ms=1000,
            nodes=(WorkNode("a", "tool", 0, 10, 10, 1, 2, 0.0, ()),),
        )
        fast = HardwareProfile("fast", "cpu", 10, 1, 256, 100, 0.01, 100, 1, frozenset({"tool"}))
        slow = HardwareProfile("slow", "cpu", 1, 0.1, 256, 100, 0.01, 500, 5, frozenset({"tool"}))

        frontier = pareto_frontier(enumerate_plans(workload, [fast, slow]))

        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0].assignments["a"], "fast")
