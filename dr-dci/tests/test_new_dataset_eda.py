import importlib.util
import tempfile
import unittest
from pathlib import Path
import zipfile


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "new_dataset_eda.py"
SPEC = importlib.util.spec_from_file_location("new_dataset_eda", SCRIPT_PATH)
new_dataset_eda = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(new_dataset_eda)


class NewDatasetEdaTests(unittest.TestCase):
    def test_text_hash_collapses_whitespace(self):
        self.assertEqual(
            new_dataset_eda.sha256_text("  보험   약관    검색  "),
            new_dataset_eda.sha256_text("보험 약관 검색"),
        )

    def test_nested_scanner_marks_zip_without_central_directory_noncanonical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inner = root / "valid.zip"
            with zipfile.ZipFile(inner, "w") as archive:
                archive.writestr("sample.txt", "valid")
            outer = root / "outer.zip"
            with zipfile.ZipFile(outer, "w") as archive:
                archive.write(inner, "valid.zip")
                archive.writestr("truncated.zip", b"PK\\x03\\x04not-a-complete-zip")

            scanner = new_dataset_eda.NestedArchiveScanner(outer, "fixture")
            scanner.scan(lambda _info, _nested, _member: None)

            self.assertEqual(scanner.outer_report["regular_file_entries_expected"], 2)
            self.assertEqual(scanner.outer_report["regular_file_entries_verified_readable"], 2)
            self.assertEqual(
                [member["status"] for member in scanner.nested_members],
                ["passed", "noncanonical_no_central_directory"],
            )


if __name__ == "__main__":
    unittest.main()
