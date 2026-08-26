import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import unittest

from build_universe import build


class RiderScopeBoundaryTest(unittest.TestCase):
    def _scopes(self, raw):
        return [(row["text"], row["contract_scope"])
                for row in build(raw, clean=True)[0]]

    def test_prefixed_rider_heading_sets_scope(self):
        scope = "(간편)암진단특약(무배당, 갱신형)"
        rows = self._scopes(f"{scope}\n\n제1조 정의\n\n보장 내용입니다.\n")
        self.assertTrue(rows)
        self.assertTrue(all(contract == scope for _, contract in rows))

    def test_non_prefixed_rider_heading_sets_scope(self):
        scope = "고혈압(원발성)약물치료특약(무배당, 해약환급금 미지급형)"
        rows = self._scopes(f"{scope}\n\n제1조 정의\n\n보장 내용입니다.\n")
        self.assertTrue(rows)
        self.assertTrue(all(contract == scope for _, contract in rows))

    def test_table_row_is_not_a_scope_boundary(self):
        table = "| 고혈압약물치료특약(무배당, 갱신형) | 가입 가능 |"
        rows = self._scopes(f"{table}\n\n일반 안내입니다.\n")
        self.assertTrue(rows)
        self.assertTrue(all(not contract for _, contract in rows))

    def test_repeated_heading_keeps_same_scope(self):
        scope = "고혈압약물치료특약(무배당, 갱신형)"
        rows = self._scopes(
            f"{scope}\n\n첫 내용입니다.\n\n{scope}\n\n둘째 내용입니다.\n")
        self.assertTrue(rows)
        self.assertTrue(all(contract == scope for _, contract in rows))


if __name__ == "__main__":
    unittest.main()
