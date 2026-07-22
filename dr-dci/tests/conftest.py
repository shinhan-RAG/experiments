import sys
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가 (src.* 임포트용)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
