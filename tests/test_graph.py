import unittest

from phaseforge.graph import WorkNode, Workload, node_names, topological_order


class GraphTests(unittest.TestCase):
    def test_topological_order_respects_dependencies(self):
        workload = Workload(
            name="toy",
            sla_ms=100,
            nodes=(
                WorkNode("b", "tool", 0, 0, 0, 1, 2, 0.0, ("a",)),
                WorkNode("a", "retrieval", 0, 0, 0, 1, 2, 0.0, ()),
            ),
        )

        self.assertEqual(node_names(topological_order(workload)), ["a", "b"])

    def test_cycle_detection(self):
        workload = Workload(
            name="bad",
            sla_ms=100,
            nodes=(
                WorkNode("a", "tool", 0, 0, 0, 1, 2, 0.0, ("b",)),
                WorkNode("b", "tool", 0, 0, 0, 1, 2, 0.0, ("a",)),
            ),
        )

        with self.assertRaisesRegex(ValueError, "cycle"):
            topological_order(workload)
