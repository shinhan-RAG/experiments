"""이동됨: retriever_rules/build_tags_u5_reference_graph.py — 하위호환 셔틀(정규식 검색기)."""
import runpy
import sys
from pathlib import Path

_target = Path(__file__).parent / "retriever_rules" / "build_tags_u5_reference_graph.py"
if __name__ == "__main__":
    sys.argv[0] = str(_target)
    runpy.run_path(str(_target), run_name="__main__")
else:
    from retriever_rules.build_tags_u5_reference_graph import *  # noqa: F401,F403
