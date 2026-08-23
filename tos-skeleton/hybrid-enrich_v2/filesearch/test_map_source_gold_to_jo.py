import unittest

from map_source_gold_to_jo import convert_rows, map_span


class SourceGoldToJoTests(unittest.TestCase):
    def setUp(self):
        self.units = [
            {"element_id": "j1", "char_start": 0, "char_end": 100},
            {"element_id": "j2", "char_start": 100, "char_end": 200},
            {"element_id": "jwide", "char_start": 0, "char_end": 200},
        ]

    def test_prefers_smallest_containing_jo(self):
        mapped = map_span(20, 80, self.units)
        self.assertEqual([("j1", 20, 80, "contained")],
                         [(row[0]["element_id"], *row[1:]) for row in mapped])

    def test_cross_boundary_is_split_conservatively(self):
        units = self.units[:2]
        mapped = map_span(80, 120, units)
        self.assertEqual([("j1", 80, 100), ("j2", 100, 120)],
                         [(row[0]["element_id"], row[1], row[2]) for row in mapped])

    def test_conversion_never_drops_unmapped_provenance(self):
        rows = [{"qid": "q1", "q": "question", "groups": [{"c0": 250, "c1": 260}],
                 "unmapped": ["source-line"]}]
        converted = convert_rows(rows, self.units)
        self.assertEqual("unmapped", converted[0]["status"])
        self.assertEqual(2, len(converted[0]["unmapped"]))


if __name__ == "__main__":
    unittest.main()
