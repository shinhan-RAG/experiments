import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import unittest

from stabilize_universe_ids import make_id_map


class StableUniverseIdTest(unittest.TestCase):
    def test_reuses_exact_rows_and_hashes_novel_rows(self):
        reference = [
            {"element_id": "e00001", "char_start": 1, "char_end": 3, "text": "aa"},
            {"element_id": "e00002", "char_start": 4, "char_end": 6, "text": "bb"},
        ]
        rebuilt = [
            {"element_id": "e10000", "char_start": 4, "char_end": 6, "text": "bb"},
            {"element_id": "e10001", "char_start": 7, "char_end": 9, "text": "cc"},
        ]
        mapping, reused = make_id_map(reference, rebuilt, "e3n")
        self.assertEqual(mapping["e10000"], "e00002")
        self.assertTrue(mapping["e10001"].startswith("e3n"))
        self.assertEqual(reused, 1)

    def test_rejects_duplicate_reference_keys(self):
        rows = [
            {"element_id": "a", "char_start": 1, "char_end": 2, "text": "x"},
            {"element_id": "b", "char_start": 1, "char_end": 2, "text": "x"},
        ]
        with self.assertRaises(ValueError):
            make_id_map(rows, [], "n")


if __name__ == "__main__":
    unittest.main()
