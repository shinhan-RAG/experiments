"""이동됨: gold_tools/finalize_reviewed_holdout.py — 하위호환 셔틀."""
import runpy
import sys
from pathlib import Path

_target = Path(__file__).parent / "gold_tools" / "finalize_reviewed_holdout.py"
if __name__ == "__main__":
    sys.argv[0] = str(_target)
    runpy.run_path(str(_target), run_name="__main__")
else:
    from gold_tools.finalize_reviewed_holdout import *  # noqa: F401,F403
