import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class V5PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arms = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((HERE / "out" / "gold_v5" / "gold_v5_manifest.json").read_text(encoding="utf-8"))

    def test_final_arm_is_slot_only_and_meta_hybrid(self):
        arm = self.arms["slot_meta_agent"]
        self.assertEqual(arm["tag_weights"], {"clm": 1.0, "sparse": 0.0})
        self.assertEqual(arm["router"], "rules")
        self.assertNotIn("qtags", arm)
        self.assertEqual(arm["meta_strategy"], "hybrid")
        self.assertTrue(arm["require_meta_before_submit"])

    def test_submit_requires_successful_meta_results(self):
        source = (HERE / "agent_tools.py").read_text(encoding="utf-8")
        self.assertIn('call.get("channel") == f"meta:{arm.get(\'meta_strategy\', \'hybrid\')}"', source)
        self.assertIn('bool(call.get("returned"))', source)

    def test_v5_data_gates(self):
        for split, expected in (("train", 337), ("test", 90)):
            stats = self.manifest[split]
            self.assertEqual(stats["n"], expected)
            self.assertEqual(stats["n_partial"], 0)
            self.assertEqual(stats["n_not_answerable"], 0)
            self.assertEqual(stats["n_groups_gt_10"], 0)
        self.assertEqual(self.manifest["train_test_qid_overlap"], 0)

    def test_known_invalid_questions_were_rewritten(self):
        path = Path(self.manifest["test"]["path"])
        with path.open(encoding="utf-8") as stream:
            rows = {row["qid"]: row for row in map(json.loads, stream)}
        self.assertNotIn("오타", rows["v3-offline-0222"]["q"])
        self.assertNotIn("행동 리스크", rows["v3-offline-0320"]["q"])
        self.assertIn("일반형", rows["v3-offline-0349"]["q"])

    def test_all_rows_are_reachable_and_within_submit_cap(self):
        for split in ("train", "test"):
            path = Path(self.manifest[split]["path"])
            with path.open(encoding="utf-8") as stream:
                for row in map(json.loads, stream):
                    self.assertEqual(row["status"], "ok")
                    self.assertFalse(row.get("c3_partial"))
                    self.assertNotEqual(row.get("task_type"), "not_answerable")
                    self.assertGreater(len(row["groups"]), 0)
                    self.assertLessEqual(len(row["groups"]), 10)
                    for group in row["groups"]:
                        self.assertTrue(group["members"])
                        self.assertTrue(group["jos"])
                        self.assertNotIn("?", group["jos"])


if __name__ == "__main__":
    unittest.main()
