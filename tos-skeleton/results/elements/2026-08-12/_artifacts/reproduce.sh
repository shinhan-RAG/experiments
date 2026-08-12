#!/usr/bin/env bash
# 최종 조합 재현 — 결정론 부분(LLM 0회)만으로 기준값 대조
set -euo pipefail
cd "$(dirname "$0")/../.."          # → tos-skeleton/semtag-schema
echo "[1/4] 입력 SHA 대조"
python3 - <<'PY'
import json,hashlib
from pathlib import Path
m=json.load(open("out/repro/MANIFEST.json",encoding="utf-8"))
def sha(p):
    p=Path(p); return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
bad=0
for f,s in m["code_sha256"].items():
    if sha(f)!=s: print(f"  ✗ {f}"); bad+=1
for k in ("element_태그",):
    d=m["최종조합"]["적재_2_5_태그"][k]
    if sha(d["file"])!=d["sha256"]: print(f"  ✗ {d['file']}"); bad+=1
u=m["최종조합"]["우주"]
if sha(u["file"])!=u["sha256"]: print(f"  ✗ {u['file']}"); bad+=1
print("  SHA 불일치" if bad else "  SHA 전부 일치")
PY
echo "[2/4] 재앵커링 (QA → P-section)"
python3 v2ds30_noah_adapter.py --full > /dev/null && echo "  완료"
echo "[3/4] 결정론 지표 — 태그 버전 2x2"
python3 v2ds34_tagversion.py --dataset v3 --core-only | tail -12
echo "[4/4] 범용/도메인 층 분리"
python3 v2ds43_layer_split.py | tail -12
echo
echo "기준값 대조: out/repro/MANIFEST.json 의 결정론_재현_기준값"
echo "에이전트 실측은 별도: python3 v2ds46_runner.py --execute --n -1 --pool all --reps 1 --workers 8 --resume"
