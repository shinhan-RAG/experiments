import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).with_name("gen_meta_v9.py")
SPEC = importlib.util.spec_from_file_location("gen_meta_v9", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BatchUniquenessTest(unittest.TestCase):
    def test_detects_same_narrow_ignoring_surface_form(self):
        rows = [
            {"narrow": "실종선고를 받으면 사망으로 보나요?", "terms": []},
            {"narrow": "실종 선고를 받으면 사망으로 보나", "terms": []},
        ]
        narrow, terms = MODULE.find_batch_conflicts(rows)
        self.assertEqual(narrow, {0, 1})
        self.assertFalse(terms)

    def test_different_numeric_anchor_is_not_conflict(self):
        rows = [
            {"narrow": "30일 한도는 얼마인가요?", "terms": []},
            {"narrow": "60일 한도는 얼마인가요?", "terms": []},
        ]
        narrow, _ = MODULE.find_batch_conflicts(rows)
        self.assertFalse(narrow)

    def test_terms_allow_partial_but_not_full_overlap(self):
        rows = [
            {"narrow": "", "terms": ["암 보장", "진단비"]},
            {"narrow": "", "terms": ["진단비", "암보장"]},
            {"narrow": "", "terms": ["진단비", "수술비"]},
        ]
        _, terms = MODULE.find_batch_conflicts(rows)
        self.assertEqual(terms, {0, 1})

    def test_fallback_blanks_later_conflicts(self):
        rows = [
            {"narrow": "실종선고를 받으면 사망으로 보나요?", "terms": ["실종 사망"]},
            {"narrow": "실종 선고를 받으면 사망으로 보나", "terms": ["실종사망"]},
        ]
        MODULE._fallback_conflicts(rows)
        self.assertTrue(rows[0]["narrow"])
        self.assertEqual(rows[1]["narrow"], "")
        self.assertEqual(rows[1]["terms"], [])
        self.assertTrue(rows[1]["narrow_conflict"])
        self.assertTrue(rows[1]["terms_full_conflict"])

    def test_prompt_states_batch_contract(self):
        prompt = MODULE.build_batch_prompt([{"text": "충분히 긴 보험 약관 본문입니다."}])
        self.assertIn("의미가 같은 질문이 없도록", prompt)
        self.assertIn("전체 집합이 같아서는 안 된다", prompt)

    def test_process_repairs_then_falls_back_if_conflict_remains(self):
        chunks = [
            {"chunk_id": "c1", "text": "실종선고 사망 인정 기준에 관한 충분한 본문"},
            {"chunk_id": "c2", "text": "실종선고 사망 인정 기준에 관한 충분한 본문"},
        ]
        generated = [
            {"chunk_id": "c1", "ok": True, "wide": "사망 인정 기준은?",
             "narrow": "실종선고를 받으면 사망으로 보나요?", "terms": ["실종 사망"],
             "narrow_grounded": True},
            {"chunk_id": "c2", "ok": True, "wide": "사망 인정 기준은?",
             "narrow": "실종선고를 받으면 사망으로 보나요?", "terms": ["실종 사망"],
             "narrow_grounded": True},
        ]
        repair_items = [
            {"index": 1, "wide": "사망 인정 기준은?",
             "narrow": "실종선고를 받으면 사망으로 보나요?", "terms": ["실종 사망"]},
            {"index": 2, "wide": "사망 인정 기준은?",
             "narrow": "실종선고를 받으면 사망으로 보나요?", "terms": ["실종 사망"]},
        ]
        with patch.object(MODULE, "_generate_batch", return_value=generated), \
             patch.object(MODULE, "call_transport", return_value=(repair_items, None)):
            rows = MODULE.process_batch(chunks, "system", MODULE.Stats())
        self.assertTrue(rows[1]["repair_attempted"])
        self.assertEqual(rows[1]["narrow"], "")
        self.assertEqual(rows[1]["terms"], [])


if __name__ == "__main__":
    unittest.main()
